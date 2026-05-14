"""
OLMoE Adaptation: Instruction Tuning (SFT) and Preference Tuning (DPO)

From Section 2 and Appendix B of the paper:

Instruction Tuning (SFT):
- Dataset: Tulu 2 SFT Mix + No Robots + CodeFeedback + MetaMathQA + Daring Anteater
- 2 epochs, constant LR of 2e-5
- Global batch size 128 (4 H100 nodes x 8 GPUs x per-device batch 2 x 2 grad accum)
- Max sequence length 4096
- Token-level loss aggregation (not sample-level)
- NO load balancing loss during SFT (key finding from Section 4.3)

Preference Tuning (DPO):
- Dataset: UltraFeedback binarized (filtered for TruthfulQA contamination)
- 3 epochs, LR 5e-7, DPO beta 0.1
- Global batch size 32
- NO load balancing loss during DPO

Key findings from Section 4.3:
1. Not using load balancing loss during adaptation leads to better performance
2. Post-annealing checkpoint is better for adaptation than pre-annealing
3. DPO and KTO perform similarly, but DPO scores higher on AlpacaEval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""
    # Training
    num_epochs: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0

    # Batch
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    global_batch_size: int = 128  # 4 nodes x 8 GPUs x 2 per-device x 2 grad accum

    # Data
    max_seq_len: int = 4096

    # Key: NO load balancing loss during SFT
    use_load_balancing_loss: bool = False
    use_router_z_loss: bool = False

    # Loss aggregation: token-level (not sample-level)
    # From Muennighoff et al. (2024) GRIT: improves performance on long generative tasks
    token_level_loss: bool = True

    # Paths
    output_dir: str = "checkpoints/sft"


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""
    # Training
    num_epochs: int = 3
    learning_rate: float = 5e-7
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    grad_clip: float = 1.0

    # DPO-specific
    beta: float = 0.1  # DPO temperature parameter

    # Batch
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    global_batch_size: int = 32

    # Data
    max_seq_len: int = 4096

    # Key: NO load balancing loss during DPO
    use_load_balancing_loss: bool = False
    use_router_z_loss: bool = False

    # Paths
    output_dir: str = "checkpoints/dpo"


def compute_sft_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_level: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute SFT loss with token-level aggregation.

    Token-level loss: aggregate loss at the token level (not sample level).
    This improves performance on long generative tasks like AlpacaEval.
    From Muennighoff et al. (2024) GRIT.

    Args:
        model: OLMoE model
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] with -100 for tokens to ignore
        attention_mask: [batch, seq_len]
        token_level: If True, use token-level loss aggregation

    Returns:
        (loss, aux_loss)
    """
    # Temporarily disable auxiliary losses if configured
    original_lb = model.config.use_load_balancing_loss
    original_rz = model.config.use_router_z_loss

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # Restore original settings
    model.config.use_load_balancing_loss = original_lb
    model.config.use_router_z_loss = original_rz

    logits = outputs["logits"]
    aux_loss = outputs["aux_loss"]

    # Compute cross-entropy loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if token_level:
        # Token-level: sum over all non-ignored tokens, then normalize by count
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        loss = loss_fct(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1)
        )
        # Normalize by number of non-ignored tokens
        num_tokens = (shift_labels != -100).sum()
        if num_tokens > 0:
            loss = loss / num_tokens
    else:
        # Sample-level: mean over samples
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1)
        )

    return loss, aux_loss


def compute_dpo_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
    attention_mask_chosen: Optional[torch.Tensor] = None,
    attention_mask_rejected: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute DPO (Direct Preference Optimization) loss.

    From Rafailov et al. (2023):
    L_DPO = -E[log sigma(beta * (log pi(y_w|x) - log pi_ref(y_w|x))
                                - (log pi(y_l|x) - log pi_ref(y_l|x)))]

    where:
    - y_w: chosen (preferred) response
    - y_l: rejected response
    - pi: policy model
    - pi_ref: reference model
    - beta: temperature parameter (0.1 in OLMoE)

    Args:
        policy_model: The model being trained
        reference_model: The reference model (frozen SFT model)
        chosen_input_ids: [batch, seq_len] for chosen responses
        chosen_labels: [batch, seq_len] labels for chosen responses
        rejected_input_ids: [batch, seq_len] for rejected responses
        rejected_labels: [batch, seq_len] labels for rejected responses
        beta: DPO temperature
        attention_mask_chosen: Optional attention mask for chosen
        attention_mask_rejected: Optional attention mask for rejected

    Returns:
        (loss, metrics_dict)
    """
    # Get log probabilities from policy model
    policy_chosen_logps = get_log_probs(
        policy_model, chosen_input_ids, chosen_labels, attention_mask_chosen
    )
    policy_rejected_logps = get_log_probs(
        policy_model, rejected_input_ids, rejected_labels, attention_mask_rejected
    )

    # Get log probabilities from reference model (no gradient)
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(
            reference_model, chosen_input_ids, chosen_labels, attention_mask_chosen
        )
        ref_rejected_logps = get_log_probs(
            reference_model, rejected_input_ids, rejected_labels, attention_mask_rejected
        )

    # Compute DPO loss
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Compute metrics
    with torch.no_grad():
        reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        reward_margins = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "chosen_rewards": chosen_rewards.mean().item(),
        "rejected_rewards": rejected_rewards.mean().item(),
        "reward_accuracy": reward_accuracies.item(),
        "reward_margin": reward_margins.item(),
    }

    return loss, metrics


def get_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-sequence log probabilities.

    Args:
        model: Language model
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] with -100 for tokens to ignore
        attention_mask: Optional attention mask

    Returns:
        log_probs: [batch] per-sequence log probabilities
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs["logits"]

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for the actual tokens
    # Mask out ignored tokens
    mask = (shift_labels != -100).float()
    shift_labels_clamped = shift_labels.clamp(min=0)

    token_log_probs = log_probs.gather(
        -1, shift_labels_clamped.unsqueeze(-1)
    ).squeeze(-1)

    # Sum log probs over sequence (only non-ignored tokens)
    seq_log_probs = (token_log_probs * mask).sum(dim=-1)

    return seq_log_probs


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning of OLMoE."""

    def __init__(
        self,
        model: nn.Module,
        config: SFTConfig,
        train_dataloader,
        eval_dataloader=None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Disable auxiliary losses during SFT
        self.model.config.use_load_balancing_loss = config.use_load_balancing_loss
        self.model.config.use_router_z_loss = config.use_router_z_loss

        # Optimizer: AdamW with constant LR
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0

    def train(self):
        """Main SFT training loop."""
        logger.info("Starting SFT training...")
        logger.info(f"Load balancing loss: {self.config.use_load_balancing_loss}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in self.train_dataloader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                self.model.train()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss, aux_loss = compute_sft_loss(
                        self.model,
                        batch["input_ids"],
                        batch["labels"],
                        batch.get("attention_mask"),
                        token_level=self.config.token_level_loss,
                    )

                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.global_step += 1

                if self.global_step % 100 == 0:
                    logger.info(
                        f"Step {self.global_step} | Loss: {loss.item():.4f}"
                    )

        logger.info("SFT training complete!")


class DPOTrainer:
    """Trainer for Direct Preference Optimization of OLMoE."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: DPOConfig,
        train_dataloader,
        eval_dataloader=None,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Freeze reference model
        for param in reference_model.parameters():
            param.requires_grad = False

        # Disable auxiliary losses during DPO
        self.policy_model.config.use_load_balancing_loss = config.use_load_balancing_loss
        self.policy_model.config.use_router_z_loss = config.use_router_z_loss

        # Optimizer: AdamW with very small LR
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0

    def train(self):
        """Main DPO training loop."""
        logger.info("Starting DPO training...")
        logger.info(f"DPO beta: {self.config.beta}")
        logger.info(f"Load balancing loss: {self.config.use_load_balancing_loss}")

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in self.train_dataloader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                self.policy_model.train()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss, metrics = compute_dpo_loss(
                        self.policy_model,
                        self.reference_model,
                        batch["chosen_input_ids"],
                        batch["chosen_labels"],
                        batch["rejected_input_ids"],
                        batch["rejected_labels"],
                        beta=self.config.beta,
                        attention_mask_chosen=batch.get("chosen_attention_mask"),
                        attention_mask_rejected=batch.get("rejected_attention_mask"),
                    )

                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), self.config.grad_clip
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.global_step += 1

                if self.global_step % 100 == 0:
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Reward Acc: {metrics['reward_accuracy']:.3f} | "
                        f"Reward Margin: {metrics['reward_margin']:.3f}"
                    )

        logger.info("DPO training complete!")
