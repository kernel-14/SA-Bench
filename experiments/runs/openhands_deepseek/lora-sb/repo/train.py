"""Training loops for LoRA-SB and baseline methods.

Supports:
- LoRA-SB: W = W0 + B @ R @ A, only R trainable.
  Uses optimal gradient approximation at each step.
- LoRA: Standard LoRA with trainable B and A.
- LoRA-XS: B and A fixed, R trainable (no gradient approximation).
- LoRA-Pro: LoRA with gradient optimization per step.
- rsLoRA, DoRA, PiSSA variants.
"""

import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm

from config import (
    ModelConfig,
    LoRAConfig,
    InitConfig,
    TrainingConfig,
    GLUE_NUM_LABELS,
)
from model import (
    replace_with_lora,
    get_lora_parameters,
    is_lora_sb_module,
    is_lora_pro_module,
    LoRASBLinear,
    LoRAProLinear,
    count_trainable_parameters,
)
from init import LoRASBInitializer, compute_lora_sb_init
from data import (
    load_glue_dataset,
    load_metamathqa,
    load_commonsense_dataset,
    create_init_dataloader,
)


def train_glue(
    model_name: str,
    task_name: str,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    init_config: Optional[InitConfig] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Train on a GLUE task.

    Args:
        model_name: Base model name (e.g. 'roberta-large').
        task_name: GLUE task name.
        lora_config: LoRA configuration.
        training_config: Training configuration.
        init_config: LoRA-SB initialization config (if None, uses default init).
        max_train_samples: Limit training samples.
        max_eval_samples: Limit eval samples.
        seed: Random seed.
        output_dir: Output directory for checkpoints.

    Returns:
        Best evaluation metric.
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, eval_dataset, _, data_collator = load_glue_dataset(
        task_name=task_name,
        tokenizer=tokenizer,
        max_seq_length=training_config.max_seq_length,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        seed=seed,
    )

    num_labels = GLUE_NUM_LABELS.get(task_name, 2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=dtype,
    ).to(device)

    lora_sb_inits = None
    if lora_config.use_lora_sb and init_config is not None:
        print("Computing LoRA-SB initialization...")
        init_loader = create_init_dataloader(
            train_dataset,
            num_samples=init_config.num_samples,
            batch_size=init_config.batch_size,
            collate_fn=data_collator,
            seed=init_config.random_seed,
        )

        def loss_fn(m, batch):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = m(**batch)
            return outputs.loss

        initializer = LoRASBInitializer(
            model=model,
            target_modules=lora_config.target_modules,
            rank=lora_config.rank,
            num_samples=init_config.num_samples,
            device=device,
            dtype=torch.float32,
        )
        lora_sb_inits = initializer.compute_update_approximation(init_loader, loss_fn)
        print(f"Initialization computed for {len(lora_sb_inits)} layers.")

    method = "lora_sb" if lora_config.use_lora_sb else ("lora_pro" if lora_config.use_optimal_gradient else "lora")
    init_method = "pissa" if "pissa" in method else "default"
    model = replace_with_lora(
        model,
        method=method,
        rank=lora_config.rank,
        alpha=lora_config.alpha,
        dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        init_method=init_method,
        lora_sb_inits=lora_sb_inits,
    )

    trainable_params = get_lora_parameters(model, method=method)
    num_trainable = count_trainable_parameters(model)
    print(f"Trainable parameters: {num_trainable:,}")

    optimizer = AdamW(
        [{"params": trainable_params, "weight_decay": training_config.weight_decay}],
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_epsilon,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_config.batch_size,
        collate_fn=data_collator,
    )

    num_update_steps_per_epoch = len(train_dataloader) // training_config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * training_config.num_epochs
    num_warmup_steps = int(total_training_steps * training_config.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=training_config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_metric = -float("inf") if task_name != "cola" else -1.0
    best_metric_name = "accuracy" if task_name not in ("cola", "stsb") else (
        "matthews_correlation" if task_name == "cola" else "pearson"
    )
    global_step = 0

    for epoch in range(training_config.num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / training_config.gradient_accumulation_steps
            loss.backward()

            if method == "lora_pro":
                for module in model.modules():
                    if isinstance(module, LoRAProLinear):
                        module.compute_optimal_gradients()

            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * training_config.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * training_config.gradient_accumulation_steps})

            if global_step > 0 and global_step % training_config.eval_steps == 0:
                eval_metric = evaluate_glue(model, eval_dataloader, task_name, device)
                print(f"Step {global_step}: {best_metric_name} = {eval_metric:.4f}")
                if eval_metric > best_metric:
                    best_metric = eval_metric
                    if output_dir:
                        model.save_pretrained(output_dir)
                model.train()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        eval_metric = evaluate_glue(model, eval_dataloader, task_name, device)
        print(f"Epoch {epoch+1} eval: {best_metric_name} = {eval_metric:.4f}")
        if eval_metric > best_metric:
            best_metric = eval_metric
            if output_dir:
                model.save_pretrained(output_dir)

    return best_metric


def train_math(
    model_name: str,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    init_config: Optional[InitConfig] = None,
    max_train_samples: int = 50000,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Train on MetaMathQA for arithmetic reasoning.

    Evaluates on GSM8K and MATH during training.

    Args:
        model_name: Base model name (e.g. 'mistralai/Mistral-7B-v0.1').
        lora_config: LoRA configuration.
        training_config: Training configuration.
        init_config: LoRA-SB initialization config.
        max_train_samples: Number of MetaMathQA samples.
        seed: Random seed.
        output_dir: Output directory.

    Returns:
        Tuple of best (gsm8k_score, math_score).
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, data_collator = load_metamathqa(
        tokenizer=tokenizer,
        max_seq_length=training_config.max_seq_length,
        max_train_samples=max_train_samples,
        seed=seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_sb_inits = None
    if lora_config.use_lora_sb and init_config is not None:
        print("Computing LoRA-SB initialization...")
        init_loader = create_init_dataloader(
            train_dataset,
            num_samples=init_config.num_samples,
            batch_size=init_config.batch_size,
            collate_fn=data_collator,
            seed=init_config.random_seed,
        )

        def loss_fn(m, batch):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = m(**batch)
            return outputs.loss

        initializer = LoRASBInitializer(
            model=model,
            target_modules=lora_config.target_modules,
            rank=lora_config.rank,
            num_samples=init_config.num_samples,
            device=device,
            dtype=torch.float32,
        )
        lora_sb_inits = initializer.compute_update_approximation(init_loader, loss_fn)
        print(f"Initialization computed for {len(lora_sb_inits)} layers.")

    method = "lora_sb" if lora_config.use_lora_sb else "lora"
    model = replace_with_lora(
        model,
        method=method,
        rank=lora_config.rank,
        alpha=lora_config.alpha,
        dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        lora_sb_inits=lora_sb_inits,
    )

    trainable_params = get_lora_parameters(model, method=method)
    num_trainable = count_trainable_parameters(model)
    print(f"Trainable parameters: {num_trainable:,}")

    optimizer = AdamW(
        [{"params": trainable_params, "weight_decay": training_config.weight_decay}],
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_epsilon,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
    num_update_steps_per_epoch = len(train_dataloader) // training_config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * training_config.num_epochs
    num_warmup_steps = int(total_training_steps * training_config.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=training_config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    global_step = 0

    for epoch in range(training_config.num_epochs):
        model.train()
        total_loss = 0.0
        accumulated_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / training_config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * training_config.gradient_accumulation_steps
            accumulated_steps += 1

            if accumulated_steps == training_config.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(trainable_params, training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accumulated_steps = 0
                global_step += 1

            progress_bar.set_postfix({"loss": loss.item() * training_config.gradient_accumulation_steps})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        print("Evaluating on GSM8K...")
        gsm8k_score = evaluate_math(model, tokenizer, "gsm8k", device)
        print(f"Evaluating on MATH...")
        math_score = evaluate_math(model, tokenizer, "math", device)
        print(f"GSM8K: {gsm8k_score:.2f}%, MATH: {math_score:.2f}%")

    return gsm8k_score, math_score


def train_commonsense(
    model_name: str,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    init_config: Optional[InitConfig] = None,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
    output_dir: Optional[str] = None,
):
    """Train on COMMONSENSE170K dataset.

    Args:
        model_name: Base model name.
        lora_config: LoRA configuration.
        training_config: Training configuration.
        init_config: LoRA-SB initialization config.
        max_train_samples: Limit training samples.
        seed: Random seed.
        output_dir: Output directory.

    Returns:
        Average accuracy across 8 commonsense tasks.
    """
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, data_collator = load_commonsense_dataset(
        tokenizer=tokenizer,
        max_seq_length=training_config.max_seq_length,
        max_train_samples=max_train_samples,
        seed=seed,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_sb_inits = None
    if lora_config.use_lora_sb and init_config is not None:
        print("Computing LoRA-SB initialization...")
        init_loader = create_init_dataloader(
            train_dataset,
            num_samples=init_config.num_samples,
            batch_size=init_config.batch_size,
            collate_fn=data_collator,
            seed=init_config.random_seed,
        )

        def loss_fn(m, batch):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = m(**batch)
            return outputs.loss

        initializer = LoRASBInitializer(
            model=model,
            target_modules=lora_config.target_modules,
            rank=lora_config.rank,
            num_samples=init_config.num_samples,
            device=device,
            dtype=torch.float32,
        )
        lora_sb_inits = initializer.compute_update_approximation(init_loader, loss_fn)
        print(f"Initialization computed for {len(lora_sb_inits)} layers.")

    method = "lora_sb" if lora_config.use_lora_sb else "lora"
    model = replace_with_lora(
        model,
        method=method,
        rank=lora_config.rank,
        alpha=lora_config.alpha,
        dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        lora_sb_inits=lora_sb_inits,
    )

    trainable_params = get_lora_parameters(model, method=method)
    num_trainable = count_trainable_parameters(model)
    print(f"Trainable parameters: {num_trainable:,}")

    optimizer = AdamW(
        [{"params": trainable_params, "weight_decay": training_config.weight_decay}],
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_epsilon,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    num_update_steps_per_epoch = len(train_dataloader) // training_config.gradient_accumulation_steps
    total_training_steps = num_update_steps_per_epoch * training_config.num_epochs
    num_warmup_steps = int(total_training_steps * training_config.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=training_config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    global_step = 0

    for epoch in range(training_config.num_epochs):
        model.train()
        total_loss = 0.0
        accumulated_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{training_config.num_epochs}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / training_config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * training_config.gradient_accumulation_steps
            accumulated_steps += 1

            if accumulated_steps == training_config.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(trainable_params, training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accumulated_steps = 0
                global_step += 1

            progress_bar.set_postfix({"loss": loss.item() * training_config.gradient_accumulation_steps})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        print("Evaluating on commonsense benchmarks...")
        from eval import evaluate_commonsense
        scores = evaluate_commonsense(model, tokenizer, device)
        avg_score = sum(scores.values()) / len(scores)
        print(f"Average accuracy: {avg_score:.2f}%")
        for task, score in scores.items():
            print(f"  {task}: {score:.2f}%")

    return avg_score


def evaluate_glue(model, eval_dataloader, task_name: str, device) -> float:
    """Evaluate on GLUE validation set.

    Returns the appropriate metric for the task.
    """
    from sklearn.metrics import matthews_corrcoef, accuracy_score
    from scipy.stats import pearsonr

    model.eval()
    all_preds = []
    all_labels = []
    is_regression = task_name == "stsb"

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)

            if is_regression:
                preds = outputs.logits.squeeze()
            else:
                preds = outputs.logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if task_name == "cola":
        return matthews_corrcoef(all_labels, all_preds)
    elif task_name == "stsb":
        return pearsonr(all_labels, all_preds)[0]
    else:
        return accuracy_score(all_labels, all_preds)


def evaluate_math(model, tokenizer, task: str, device) -> float:
    """Evaluate on GSM8K or MATH by generating answers and checking correctness."""
    from eval import evaluate_math_task
    return evaluate_math_task(model, tokenizer, task, device)


def main():
    """Example training script entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Train LoRA-SB on various tasks")
    parser.add_argument("--task_type", type=str, default="glue", choices=["glue", "math", "commonsense"])
    parser.add_argument("--task_name", type=str, default="mrpc", help="GLUE task name")
    parser.add_argument("--model_name", type=str, default="roberta-large")
    parser.add_argument("--method", type=str, default="lora_sb",
                        choices=["lora", "rslora", "dora", "lora_xs", "lora_sb", "lora_pro", "pissa"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--init_samples", type=int, default=50, help="Samples for LoRA-SB init")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    model_config = ModelConfig(name=args.model_name)
    lora_config = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        use_lora_sb=(args.method == "lora_sb"),
        use_optimal_gradient=(args.method in ("lora_sb", "lora_pro")),
    )
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    init_config = InitConfig(num_samples=args.init_samples)

    if args.task_type == "glue":
        result = train_glue(
            model_name=args.model_name,
            task_name=args.task_name,
            lora_config=lora_config,
            training_config=training_config,
            init_config=init_config if args.method == "lora_sb" else None,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"Best metric: {result:.4f}")
    elif args.task_type == "math":
        gsm8k, math_score = train_math(
            model_name=args.model_name,
            lora_config=lora_config,
            training_config=training_config,
            init_config=init_config if args.method == "lora_sb" else None,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"GSM8K: {gsm8k:.2f}%, MATH: {math_score:.2f}%")
    elif args.task_type == "commonsense":
        avg_score = train_commonsense(
            model_name=args.model_name,
            lora_config=lora_config,
            training_config=training_config,
            init_config=init_config if args.method == "lora_sb" else None,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"Average accuracy: {avg_score:.2f}%")


if __name__ == "__main__":
    main()
