
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, StateDictType, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrapper_policy
import functools
import random
import numpy as np
import time
import logging
from tqdm import tqdm

from config import Config, ModelConfig, TrainingConfig
from model import OLMoE, TransformerBlock
from data import DataProcessor
from modules import compute_load_balancing_loss, compute_router_z_loss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_lr_scheduler(optimizer, train_config: TrainingConfig, total_training_steps: int):
    """
    Creates a learning rate scheduler with cosine decay and linear warmup,
    followed by a linear decay for annealing.
    """
    warmup_steps = train_config.warmup_steps
    peak_lr = train_config.learning_rate # 4.0e-4
    min_lr_cosine_decay = train_config.minimum_learning_rate # 4.0e-5
    annealing_tokens_billions = train_config.annealing_tokens_billions # 100.0
    annealing_min_lr = train_config.annealing_min_lr # 0.0

    # Calculate total steps for the entire training (including annealing)
    # This is a simplification; in practice, total_training_steps would be a more precise count
    # based on actual dataset size and epochs.
    
    # Calculate annealing steps based on tokens / global batch size tokens
    annealing_steps = int(annealing_tokens_billions * 1e9 / train_config.global_batch_size_tokens)
    
    # The main training steps are total_training_steps - annealing_steps (before annealing)
    main_training_steps = total_training_steps - annealing_steps

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay phase
        if current_step < main_training_steps:
            progress = (current_step - warmup_steps) / max(1, main_training_steps - warmup_steps)
            # Cosine decay from peak_lr down to min_lr_cosine_decay
            return (min_lr_cosine_decay / peak_lr) + (1.0 - (min_lr_cosine_decay / peak_lr)) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Annealing phase: linear decay from min_lr_cosine_decay to annealing_min_lr (0.0)
        else:
            annealing_progress = (current_step - main_training_steps) / max(1, annealing_steps)
            # The LR at the start of annealing is min_lr_cosine_decay. Decay this linearly to annealing_min_lr.
            return max(0.0, (1.0 - annealing_progress) * (min_lr_cosine_decay / peak_lr) + annealing_progress * (annealing_min_lr / peak_lr))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    cfg = Config()
    train_cfg = cfg.training
    model_cfg = cfg.model
    data_cfg = cfg.data

    # Initialize distributed training
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = "cpu"
        logging.warning("CUDA not available, running on CPU. Training will be very slow.")

    if rank == 0:
        logging.info(f"Starting training on {world_size} devices.")
        logging.info(f"Model config: {model_cfg}")
        logging.info(f"Training config: {train_cfg}")
        logging.info(f"Data config: {data_cfg}")

    set_seed(train_cfg.seed + rank)

    # Data Processor
    data_processor = DataProcessor(data_cfg)

    # Model
    model = OLMoE(model_cfg)
    model.to(device)

    # FSDP Mixed Precision
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16,
        reduce_dtype=torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16,
        buffer_dtype=torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16,
        keep_low_precision_fp32_ops=True # Paper states gradient_reduce_dtype FP32
    )

    # FSDP Wrapping Policy
    # Wrap TransformerBlock modules with FSDP
    auto_wrap_policy = functools.partial(
        transformer_auto_wrapper_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # FSDP setup
    # Paper mentions ZeRO via PyTorch FSDP. SHARD_GRAD_OP is similar to ZeRO-2/3.
    # We will use FULL_SHARD for maximal sharding as in ZeRO-3
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # Corresponds to ZeRO-3
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=device,
        param_init_fn=lambda module: module.to(device), # Ensure params are on device for init
        use_orig_params=True # Recommended for better compatibility with optimizers
    )
    
    if rank == 0:
        logging.info(f"FSDP Model initialized: {fsdp_model}")
        logging.info(f"Total model parameters: {sum(p.numel() for p in fsdp_model.parameters()) / 1e6:.2f}M")
        logging.info(f"Trainable model parameters: {sum(p.numel() for p in fsdp_model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(
        fsdp_model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        eps=train_cfg.adam_epsilon,
        weight_decay=train_cfg.weight_decay
    )
    # The paper explicitly states "we weight decay all parameters in OLMOE-1B-7B including embedding and RMSNorm."
    # So no special handling for `decay_rmsnorm_params` or `decay_embedding_params` is needed beyond
    # passing weight_decay to AdamW.

    # Total training steps calculation
    # Paper: 5.133T tokens total, global_batch_size_tokens (4M tokens).
    # One "step" is typically one optimization step (forward + backward + update)
    total_training_steps = int(train_cfg.pretraining_tokens_billions * 1e9 / train_cfg.global_batch_size_tokens)
    
    # Learning Rate Scheduler
    lr_scheduler = get_lr_scheduler(optimizer, train_cfg, total_training_steps)

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # Assuming -100 is ignore index for padding

    # Autocast for mixed precision
    scaler = amp.GradScaler(enabled=(train_cfg.mixed_precision == "fp16"))

    start_time = time.time()
    for step in range(total_training_steps):
        fsdp_model.train()
        optimizer.zero_grad()

        # Simulate data loading
        # In a real setup, this would be a proper DataLoader
        batch = next(data_processor.get_pretraining_dataloader(
            batch_size=train_cfg.global_batch_size_samples // world_size, # Per-device batch size
            seq_len=train_cfg.sequence_length,
            num_batches=total_training_steps # Just ensure it yields enough
        ))
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = (input_ids != 0).long().to(device) # Simple mask for padding

        with amp.autocast(enabled=(train_cfg.mixed_precision != "fp32")):
            logits, all_router_logits, all_expert_gate_probabilities = fsdp_model(input_ids, attention_mask)
            
            # Cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = ce_loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss = ce_loss

            # Auxiliary losses
            if train_cfg.use_load_balancing_loss and all_expert_gate_probabilities:
                load_balancing_loss = 0
                for probs in all_expert_gate_probabilities:
                    load_balancing_loss += compute_load_balancing_loss(
                        probs, model_cfg.num_experts, model_cfg.num_activated_experts
                    )
                load_balancing_loss *= train_cfg.load_balancing_loss_weight
                total_loss += load_balancing_loss
            
            if train_cfg.use_router_z_loss and all_router_logits:
                router_z_loss = 0
                for r_logits in all_router_logits:
                    router_z_loss += compute_router_z_loss(r_logits, model_cfg.num_experts)
                router_z_loss *= train_cfg.router_z_loss_weight
                total_loss += router_z_loss

        scaler.scale(total_loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), train_cfg.gradient_clipping)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        if rank == 0 and step % 100 == 0: # Log every 100 steps
            elapsed_time = time.time() - start_time
            tokens_processed = (step + 1) * train_cfg.global_batch_size_tokens
            tps_per_gpu = tokens_processed / (elapsed_time * world_size)
            
            logging.info(
                f"Step {step}/{total_training_steps} | "
                f"Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}) | "
                f"LR: {lr_scheduler.get_last_lr()[0]:.6f} | "
                f"Tokens/GPU/sec: {tps_per_gpu:.2f}"
            )
            # Placeholder for evaluation/checkpointing

    if rank == 0:
        logging.info("Training finished.")
        # Save final model checkpoint
        save_path = "final_olmoe_model.pt"
        # FSDP recommended way to save full state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = fsdp_model.state_dict()
        if rank == 0:
            torch.save(cpu_state, save_path)
            logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()

