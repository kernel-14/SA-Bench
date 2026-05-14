# trainer/adaptation_trainer.py
"""
Adaptation Trainer for OLMoE-1B-7B-INSTRUCT.

Implements instruction tuning (SFT) and preference tuning (DPO/KTO) following
the paper's exact recipe (Section 2, Section 4.3, Appendix B).  All hyper‑
parameters are drawn from the configuration dictionary.

The trainer expects a MoETransformer loaded with the annealed pretraining
weights, a pre‑trained tokenizer, and the full YAML config.  It handles
FSDP wrapping, bfloat16 mixed‑precision, gradient clipping, and distributed
sampling internally.

Typical usage:
    trainer = AdaptationTrainer(model, tokenizer, config)
    trainer.sft_train(sft_dataset)   # saves SFT checkpoint
    trainer.dpo_train(dpo_dataset)   # saves final INSTRUCT checkpoint
"""

from __future__ import annotations

import logging
import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Sampler

# Custom project imports – these modules must be present in the PYTHONPATH.
from model.moe_transformer import MoETransformer
from trainer.utils import (
    get_optimizer_and_scheduler,   # not used for adaptation (constant LR)
    load_checkpoint,
    save_checkpoint,
    setup_fsdp_model,
)
from utils.logging_utils import init_wandb, log_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collation helpers
# ---------------------------------------------------------------------------
def _collate_sft(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of tokenized SFT examples.

    Each example is a dict with ``input_ids``, ``attention_mask``, ``labels``
    (all as Python lists of ints). The function pads all sequences to the
    maximum length present in the batch, truncating if necessary to the
    global ``max_seq_length`` (taken from the outer scope via closure or
    passed as a default – we capture it from the trainer instance).
    """
    # We'll use the trainer's max_seq_length later – for now we just pad
    # and then truncate to max_seq_length if needed.
    # The function will be a method of the trainer class for simplicity.
    raise NotImplementedError("Will be implemented as a method of AdaptationTrainer")
# We'll implement as a method of AdaptationTrainer so it has access to self.max_seq_length.


# ---------------------------------------------------------------------------
# AdaptationTrainer
# ---------------------------------------------------------------------------
class AdaptationTrainer:
    """
    Run SFT and DPO adaptation for OLMoE-1B-7B.

    Args:
        model:      Raw (un‑wrapped) MoETransformer with pretrained weights.
        tokenizer:  GPT‑NeoX tokenizer (from `allenai/OLMo‑1B‑0724‑hf`).
        config:     Full configuration dictionary (as loaded from config.yaml).
    """

    def __init__(
        self,
        model: MoETransformer,
        tokenizer,   # PreTrainedTokenizer
        config: Dict,
    ) -> None:
        # Distributed environment (assumes init_process_group already called)
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed process group must be initialised before creating "
                "AdaptationTrainer."
            )
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Core objects
        self.raw_model = model
        self.tokenizer = tokenizer
        self.config = config

        # Convenience shortcuts
        self.cfg_adapt = config["adaptation"]
        self.cfg_model = config["model"]
        self.cfg_pretrain = config["pretraining"]
        self.cfg_logging = config["logging"]
        self.cfg_fsdp = config.get("fsdp", {})

        # Sequence length used during adaptation
        self.max_seq_length = self.cfg_adapt["sft"].get(
            "max_seq_length", self.cfg_pretrain["seq_length"]
        )

        # Auxiliary loss control (MoE‑specific)
        self.use_load_balancing = self.cfg_adapt["sft"].get(
            "load_balancing_loss", False
        )  # Paper: disabled for both SFT and DPO
        self.use_router_z_loss = self.cfg_adapt["dpo"].get(
            "router_z_loss", False
        )  # Not mentioned, default False

        # Mixed precision dtype
        precision_str = self.cfg_adapt["sft"].get("precision", "bf16").lower()
        if precision_str == "bf16":
            self.amp_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported adaptation precision: {precision_str}")

        # Initialize W&B logging (once per process)
        init_wandb(self.cfg_logging)

        # Checkpoint directory
        self.checkpoint_dir = self.cfg_adapt.get("checkpoint_dir", "adapt_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Disable MoE auxiliary losses if configured
        # (We simply ignore them in the loss computation, but the paper says
        # "we do not use load balancing during adaptation".  The model's
        # forward still returns router logits; we discard them.)

        # Prepare a reference to the loss functions (imported if needed)
        from model.losses import load_balancing_loss, router_z_loss
        self._lb_loss_fn = load_balancing_loss
        self._rz_loss_fn = router_z_loss

        logger.info(
            "AdaptationTrainer initialised: rank=%d/%d, device=%s, max_seq_len=%d",
            self.rank,
            self.world_size,
            self.device,
            self.max_seq_length,
        )

    # ------------------------------------------------------------------
    # Collation helpers (as methods to access self.max_seq_length)
    # ------------------------------------------------------------------
    def _collate_sft(
        self, batch: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate and pad SFT examples to the longest sequence in the batch,
        then truncate to ``self.max_seq_length``.
        """
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Find longest sequence in the batch
        max_len = min(
            max(len(ex["input_ids"]) for ex in batch),
            self.max_seq_length,
        )

        input_ids_list, attn_mask_list, labels_list = [], [], []
        for ex in batch:
            ids = ex["input_ids"][:max_len]
            mask = ex["attention_mask"][:max_len] if "attention_mask" in ex else [1] * len(ids)
            labels = ex["labels"][:max_len]

            # Padding
            pad_size = max_len - len(ids)
            ids = ids + [pad_token_id] * pad_size
            mask = mask + [0] * pad_size
            labels = labels + [-100] * pad_size   # loss ignores padded tokens

            input_ids_list.append(ids)
            attn_mask_list.append(mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_list, dtype=torch.bool),
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

    def _collate_dpo(
        self, batch: List[Dict[str, List[int]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate DPO examples.  The dataset is expected to contain:

        prompt_input_ids, prompt_attention_mask,
        chosen_input_ids (response only), chosen_attention_mask,
        rejected_input_ids (response only), rejected_attention_mask.

        This function constructs full sequences for chosen and rejected,
        creates labels that mask the prompt, and pads to the batch‑max length.
        """
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        # Determine maximum total length (prompt + response) across batch
        max_total_len = 0
        for ex in batch:
            p_len = len(ex["prompt_input_ids"])
            c_len = len(ex["chosen_input_ids"])
            r_len = len(ex["rejected_input_ids"])
            total = max(p_len + c_len, p_len + r_len)
            if total > max_total_len:
                max_total_len = total
        max_total_len = min(max_total_len, self.max_seq_length)

        chosen_ids_batch, chosen_attn_batch, chosen_labels_batch = [], [], []
        rejected_ids_batch, rejected_attn_batch, rejected_labels_batch = [], [], []

        for ex in batch:
            p_ids = ex["prompt_input_ids"]
            p_mask = ex["prompt_attention_mask"]
            c_ids = ex["chosen_input_ids"]
            c_mask = ex["chosen_attention_mask"]
            r_ids = ex["rejected_input_ids"]
            r_mask = ex["rejected_attention_mask"]

            # Build full sequence for chosen
            full_c_ids = p_ids + c_ids
            full_c_mask = p_mask + c_mask
            full_c_labels = [-100] * len(p_ids) + c_ids

            # Truncate to max_total_len
            full_c_ids = full_c_ids[:max_total_len]
            full_c_mask = full_c_mask[:max_total_len]
            full_c_labels = full_c_labels[:max_total_len]

            # Build full sequence for rejected
            full_r_ids = p_ids + r_ids
            full_r_mask = p_mask + r_mask
            full_r_labels = [-100] * len(p_ids) + r_ids
            full_r_ids = full_r_ids[:max_total_len]
            full_r_mask = full_r_mask[:max_total_len]
            full_r_labels = full_r_labels[:max_total_len]

            chosen_ids_batch.append(full_c_ids)
            chosen_attn_batch.append(full_c_mask)
            chosen_labels_batch.append(full_c_labels)
            rejected_ids_batch.append(full_r_ids)
            rejected_attn_batch.append(full_r_mask)
            rejected_labels_batch.append(full_r_labels)

        # Pad to max_total_len (some may be shorter because the combined
        # length didn't reach max_total_len).
        def pad_seq(seq_list, pad_val):
            padded = []
            for seq in seq_list:
                pad_size = max_total_len - len(seq)
                padded.append(seq + [pad_val] * pad_size)
            return padded

        # Pad inputs and attention masks with 0, labels with -100
        chosen_ids_batch = pad_seq(chosen_ids_batch, pad_token_id)
        chosen_attn_batch = pad_seq(chosen_attn_batch, 0)
        chosen_labels_batch = pad_seq(chosen_labels_batch, -100)

        rejected_ids_batch = pad_seq(rejected_ids_batch, pad_token_id)
        rejected_attn_batch = pad_seq(rejected_attn_batch, 0)
        rejected_labels_batch = pad_seq(rejected_labels_batch, -100)

        return {
            "chosen_input_ids": torch.tensor(chosen_ids_batch, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_attn_batch, dtype=torch.bool),
            "chosen_labels": torch.tensor(chosen_labels_batch, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids_batch, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_attn_batch, dtype=torch.bool),
            "rejected_labels": torch.tensor(rejected_labels_batch, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Model and optimizer creation for a specific stage
    # ------------------------------------------------------------------
    def _create_fsdp_model(self) -> nn.Module:
        """
        Wrap the raw model with FSDP using the configuration.  This is called
        once at the beginning of each training stage (SFT, DPO policy).
        """
        # We pass the raw_model; setup_fsdp_model expects a MoETransformer.
        wrapped = setup_fsdp_model(self.raw_model, self.config)
        return wrapped

    def _create_optimizer(
        self, model: nn.Module, lr: float, weight_decay: float = 0.0
    ) -> AdamW:
        """
        AdamW optimizer with constant learning rate and no scheduler.
        """
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=(self.cfg_pretrain["adam_beta1"], self.cfg_pretrain["adam_beta2"]),
            weight_decay=weight_decay,
            eps=self.cfg_pretrain["adam_epsilon"],   # 1e-8 from config
        )
        return optimizer

    # ------------------------------------------------------------------
    # SFT Training
    # ------------------------------------------------------------------
    def sft_train(self, dataset: torch.utils.data.Dataset) -> None:
        """
        Run supervised instruction tuning using the given dataset.

        Saves the final SFT model to ``adapt_checkpoints/sft_model.pt``.
        """
        logger.info("Starting SFT training")
        self.raw_model.train()

        # Wrap with FSDP
        fsdp_model = self._create_fsdp_model()
        self.fsdp_model = fsdp_model  # store for later use

        # Hyperparameters from config
        sft_cfg = self.cfg_adapt["sft"]
        lr = sft_cfg["learning_rate"]        # 2.0e-5
        epochs = sft_cfg["epochs"]           # 2
        global_batch_size = sft_cfg["global_batch_size"]  # 128
        grad_accum = sft_cfg.get("gradient_accumulation_steps", 2)
        per_device_batch = sft_cfg.get("per_device_batch_size", 2)
        # (128 / 32 = 4 -> per_device 2 with accum 2)
        assert global_batch_size == self.world_size * per_device_batch * grad_accum, (
            f"global_batch_size mismatch: {global_batch_size} vs "
            f"world_size * per_device * accum "
            f"({self.world_size} * {per_device_batch} * {grad_accum})"
        )

        # Optimizer (no learning rate decay)
        optimizer = self._create_optimizer(fsdp_model, lr)

        # Distributed sampler and DataLoader
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=per_device_batch,
            sampler=sampler,
            collate_fn=self._collate_sft,
            drop_last=False,
            pin_memory=True,
        )

        total_steps = epochs * len(dataloader) // grad_accum
        logger.info(
            "SFT: %d epochs, %d steps/epoch, total steps ~%d",
            epochs,
            len(dataloader) // grad_accum,
            total_steps,
        )

        global_step = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device (each process gets its shard)
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Forward in bf16 autocast
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    # The model returns (logits, list_of_router_logits)
                    logits, _ = fsdp_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    # Shift for next‑token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Cross‑entropy with ignore_index = -100 (prompt tokens)
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / grad_accum

                # Backward
                loss.backward()

                # Step on gradient accumulation boundary
                if (batch_idx + 1) % grad_accum == 0:
                    # Gradient clipping
                    grad_norm = clip_grad_norm_(
                        fsdp_model.parameters(),
                        self.cfg_pretrain["gradient_clipping"],  # 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.cfg_logging.get("log_interval", 10) == 0:
                        metrics = {
                            "sft/loss": loss.item() * grad_accum,  # unscale
                            "sft/lr": lr,
                            "sft/grad_norm": grad_norm,
                        }
                        log_metrics(metrics, step=global_step)
                        if self.rank == 0:
                            logger.debug(
                                "SFT step %d: loss=%.4f, grad_norm=%.2f",
                                global_step,
                                loss.item() * grad_accum,
                                grad_norm,
                            )

        # Save SFT checkpoint
        save_path = os.path.join(self.checkpoint_dir, "sft_model.pt")
        save_checkpoint(
            fsdp_model=fsdp_model,
            optimizer=optimizer,
            scheduler=None,          # no scheduler used
            step=global_step,
            path=save_path,
            config=self.config,
        )
        if self.rank == 0:
            logger.info("SFT model saved to %s", save_path)

        # Clean up FSDP wrapper (will be re‑wrapped for DPO)
        self.fsdp_model = None
        del fsdp_model

    # ------------------------------------------------------------------
    # DPO Training
    # ------------------------------------------------------------------
    def dpo_train(self, dataset: torch.utils.data.Dataset) -> None:
        """
        Run Direct Preference Optimization.

        Expects a DPO dataset that provides prompt/response splits (see
        ``_collate_dpo``).  A frozen copy of the SFT model is used as the
        reference.

        Saves the final INSTRUCT model to ``adapt_checkpoints/instruct_model.pt``.
        """
        logger.info("Starting DPO training")

        # Load SFT checkpoint as reference model (un‑wrapped, frozen)
        sft_path = os.path.join(self.checkpoint_dir, "sft_model.pt")
        if not os.path.isfile(sft_path):
            raise FileNotFoundError(
                f"SFT checkpoint not found at {sft_path}. Run sft_train() first."
            )

        # Create reference model: instantiate new MoETransformer with same config
        ref_model = MoETransformer(self.cfg_model)
        # Load state dict from the saved checkpoint (full state dict)
        ckpt = torch.load(sft_path, map_location="cpu")
        ref_model.load_state_dict(ckpt["model"])
        ref_model.to(self.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        self.ref_model = ref_model

        # Policy model: we reuse the same raw_model but reload SFT weights
        # (Since raw_model already has pretrained weights, we re‑load SFT weights
        #  to start from the same policy as the reference.)
        policy_raw = MoETransformer(self.cfg_model)
        policy_raw.load_state_dict(ckpt["model"])
        policy_raw.to(self.device)
        policy_raw.train()
        self.raw_model = policy_raw   # update the stored raw model

        # Wrap policy with FSDP
        policy_fsdp = setup_fsdp_model(policy_raw, self.config)
        self.fsdp_model = policy_fsdp

        # Hyperparameters from config
        dpo_cfg = self.cfg_adapt["dpo"]
        lr = dpo_cfg["learning_rate"]           # 5.0e-7
        epochs = dpo_cfg["epochs"]              # 3
        global_batch_size = dpo_cfg["global_batch_size"]  # 32
        per_device_batch = dpo_cfg.get("per_device_batch_size", 1)
        grad_accum = dpo_cfg.get("gradient_accumulation_steps", 1)
        beta = dpo_cfg["dpo_beta"]              # 0.1

        assert global_batch_size == self.world_size * per_device_batch * grad_accum, (
            f"DPO global_batch_size mismatch: {global_batch_size} vs "
            f"world_size * per_device * accum "
            f"({self.world_size} * {per_device_batch} * {grad_accum})"
        )

        # Optimizer (constant LR)
        optimizer = self._create_optimizer(policy_fsdp, lr)

        # Distributed sampler and DataLoader
        sampler = DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=per_device_batch,
            sampler=sampler,
            collate_fn=self._collate_dpo,
            drop_last=False,
            pin_memory=True,
        )

        total_steps = epochs * len(dataloader) // grad_accum
        logger.info(
            "DPO: %d epochs, %d steps/epoch, total steps ~%d",
            epochs,
            len(dataloader) // grad_accum,
            total_steps,
        )

        global_step = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                # Move data to device
                chosen_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
                chosen_attn = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
                chosen_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
                rejected_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
                rejected_attn = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
                rejected_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    # Policy model forward for chosen and rejected
                    logits_chosen, _ = policy_fsdp(
                        input_ids=chosen_ids, attention_mask=chosen_attn
                    )
                    logits_rejected, _ = policy_fsdp(
                        input_ids=rejected_ids, attention_mask=rejected_attn
                    )

                    # Reference model forward (with no gradient)
                    with torch.no_grad():
                        ref_logits_chosen, _ = self.ref_model(
                            input_ids=chosen_ids, attention_mask=chosen_attn
                        )
                        ref_logits_rejected, _ = self.ref_model(
                            input_ids=rejected_ids, attention_mask=rejected_attn
                        )

                    # Compute sum of log‑probs over the response tokens
                    # (negative cross‑entropy sum)
                    def _sum_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        # Compute per‑token cross‑entropy loss with ignore_index=-100
                        per_token_loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            reduction='none',
                        ).view(shift_labels.shape)
                        # Sum over non‑ignored tokens, then negate to get sum log prob
                        mask = (shift_labels != -100)
                        sum_logp = -(per_token_loss * mask).sum(dim=-1)  # (batch_size,)
                        return sum_logp

                    policy_chosen_logp = _sum_log_probs(logits_chosen, chosen_labels)
                    policy_rejected_logp = _sum_log_probs(logits_rejected, rejected_labels)
                    ref_chosen_logp = _sum_log_probs(ref_logits_chosen, chosen_labels)
                    ref_rejected_logp = _sum_log_probs(ref_logits_rejected, rejected_labels)

                    # DPO loss
                    log_ratio_w = policy_chosen_logp - ref_chosen_logp
                    log_ratio_l = policy_rejected_logp - ref_rejected_logp
                    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()
                    loss = loss / grad_accum

                # Backward
                loss.backward()

                # Step logic
                if (batch_idx + 1) % grad_accum == 0:
                    grad_norm = clip_grad_norm_(
                        policy_fsdp.parameters(),
                        self.cfg_pretrain["gradient_clipping"],
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.cfg_logging.get("log_interval", 10) == 0:
                        metrics = {
                            "dpo/loss": loss.item() * grad_accum,
                            "dpo/lr": lr,
                            "dpo/grad_norm": grad_norm,
                            "dpo/policy_chosen_logp": policy_chosen_logp.mean().item(),
                            "dpo/policy_rejected_logp": policy_rejected_logp.mean().item(),
                            "dpo/ref_chosen_logp": ref_chosen_logp.mean().item(),
                            "dpo/ref_rejected_logp": ref_rejected_logp.mean().item(),
                        }
                        log_metrics(metrics, step=global_step)
                        if self.rank == 0:
                            logger.debug(
                                "DPO step %d: loss=%.4f, grad_norm=%.2f",
                                global_step,
                                loss.item() * grad_accum,
                                grad_norm,
                            )

        # Save final INSTRUCT checkpoint
        save_path = os.path.join(self.checkpoint_dir, "instruct_model.pt")
        save_checkpoint(
            fsdp_model=policy_fsdp,
            optimizer=optimizer,
            scheduler=None,
            step=global_step,
            path=save_path,
            config=self.config,
        )
        if self.rank == 0:
            logger.info("INSTRUCT model saved to %s", save_path)

        # Clean up
        self.fsdp_model = None
        del policy_fsdp
        del self.ref_model

