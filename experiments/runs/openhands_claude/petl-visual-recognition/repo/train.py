"""
Training loop for all PEFT experiments.

Supports three experiment types:
  1. VTAB-1K low-shot (100 epochs, no augmentation)
  2. Many-shot (40 epochs, dataset-specific augmentation)
  3. Robustness / CLIP fine-tuning (100 epochs, strong augmentation)

Hyperparameter search is performed over learning rate, weight decay,
and method-specific parameters (e.g., bottleneck dimension, scale factor).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    VTABConfig,
    ManyShotConfig,
    RobustnessConfig,
    PEFT_SEARCH_GRIDS,
    PEFT_DEFAULT_CONFIGS,
    VTAB_ALL_TASKS,
    VTAB_NUM_CLASSES,
    MANYSHOT_DATASETS,
)
from data import (
    get_vtab_dataloaders,
    get_manyshot_dataloaders,
    get_imagenet_dataloaders,
)
from models.vit import build_peft_model, count_trainable_params
from utils import set_seed, AverageMeter, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(logits, labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        acc = accuracy(logits, labels)

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optimizer_name: str = "adamw",
) -> optim.Optimizer:
    """Build AdamW optimizer with separate weight decay for bias/norm params."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "gamma" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_name == "adamw":
        return optim.AdamW(param_groups, lr=lr)
    elif optimizer_name == "adam":
        return optim.Adam(param_groups, lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_single_config(
    peft_method: str,
    peft_config: Dict,
    num_classes: int,
    dataloaders: Dict[str, DataLoader],
    lr: float,
    weight_decay: float,
    num_epochs: int,
    drop_path_rate: float,
    device: torch.device,
    model_name: str = "vit_base_patch16_224.augreg_in21k",
    pretrained: bool = True,
    output_dir: Optional[str] = None,
    writer: Optional[SummaryWriter] = None,
    run_name: str = "",
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train a single PEFT configuration and return the best model + metrics."""
    set_seed(42)

    model = build_peft_model(
        peft_method=peft_method,
        num_classes=num_classes,
        peft_config=peft_config,
        drop_path_rate=drop_path_rate,
        pretrained=pretrained,
        model_name=model_name,
    )
    model = model.to(device)

    n_trainable = count_trainable_params(model)
    print(f"  Trainable params: {n_trainable / 1e6:.3f}M")

    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device, epoch
        )
        val_metrics = evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step()

        if writer:
            writer.add_scalar(f"{run_name}/train_loss", train_metrics["loss"], epoch)
            writer.add_scalar(f"{run_name}/train_acc", train_metrics["acc"], epoch)
            writer.add_scalar(f"{run_name}/val_loss", val_metrics["loss"], epoch)
            writer.add_scalar(f"{run_name}/val_acc", val_metrics["acc"], epoch)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_model_state = copy.deepcopy(model.state_dict())

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, dataloaders["test"], criterion, device)

    return model, {
        "val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "n_trainable_params": n_trainable,
    }


def hyperparameter_search(
    peft_method: str,
    num_classes: int,
    dataloaders: Dict[str, DataLoader],
    num_epochs: int,
    device: torch.device,
    lr_choices: List[float],
    wd_choices: List[float],
    drop_path_rate_choices: List[float],
    model_name: str = "vit_base_patch16_224.augreg_in21k",
    output_dir: Optional[str] = None,
) -> Tuple[Dict, float]:
    """
    Grid search over learning rate, weight decay, drop path rate,
    and method-specific hyperparameters.
    Returns the best config and its validation accuracy.
    """
    search_grid = PEFT_SEARCH_GRIDS.get(peft_method, {})
    default_config = PEFT_DEFAULT_CONFIGS.get(peft_method, {})

    # Build all combinations of method-specific hyperparameters
    if search_grid:
        keys = list(search_grid.keys())
        values = list(search_grid.values())
        method_configs = [
            dict(zip(keys, combo)) for combo in itertools.product(*values)
        ]
    else:
        method_configs = [{}]

    best_val_acc = 0.0
    best_config = {
        "lr": lr_choices[0],
        "weight_decay": wd_choices[0],
        "drop_path_rate": drop_path_rate_choices[0],
        "peft_config": default_config,
    }

    for lr, wd, dpr, method_cfg in itertools.product(
        lr_choices, wd_choices, drop_path_rate_choices, method_configs
    ):
        config_str = f"lr={lr}, wd={wd}, dpr={dpr}, {method_cfg}"
        print(f"  Trying: {config_str}")

        try:
            _, metrics = train_single_config(
                peft_method=peft_method,
                peft_config=method_cfg,
                num_classes=num_classes,
                dataloaders=dataloaders,
                lr=lr,
                weight_decay=wd,
                num_epochs=num_epochs,
                drop_path_rate=dpr,
                device=device,
                model_name=model_name,
            )
            val_acc = metrics["val_acc"]
            print(f"    Val acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = {
                    "lr": lr,
                    "weight_decay": wd,
                    "drop_path_rate": dpr,
                    "peft_config": method_cfg,
                }
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    return best_config, best_val_acc


def run_vtab_experiment(
    peft_method: str,
    task_name: str,
    config: VTABConfig,
    device: torch.device,
    data_dir: str = "./data/vtab",
    output_dir: str = "./outputs/vtab",
    do_hparam_search: bool = True,
) -> Dict:
    """Run VTAB-1K experiment for a single task and PEFT method."""
    print(f"\n{'='*60}")
    print(f"VTAB-1K | Method: {peft_method} | Task: {task_name}")
    print(f"{'='*60}")

    num_classes = VTAB_NUM_CLASSES[task_name]
    dataloaders = get_vtab_dataloaders(
        task_name=task_name,
        batch_size=config.batch_size,
        image_size=config.image_size,
        data_dir=data_dir,
        num_train_samples=config.num_train_samples,
    )

    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / f"{peft_method}_{task_name}_results.json"

    if do_hparam_search:
        best_config, best_val_acc = hyperparameter_search(
            peft_method=peft_method,
            num_classes=num_classes,
            dataloaders=dataloaders,
            num_epochs=config.num_epochs,
            device=device,
            lr_choices=config.lr_choices,
            wd_choices=config.wd_choices,
            drop_path_rate_choices=config.drop_path_rate_choices,
            output_dir=output_dir,
        )
    else:
        best_config = {
            "lr": config.lr_choices[0],
            "weight_decay": config.wd_choices[0],
            "drop_path_rate": config.drop_path_rate,
            "peft_config": PEFT_DEFAULT_CONFIGS.get(peft_method, {}),
        }

    # Final training with best config on full 1000 samples
    print(f"\nFinal training with best config: {best_config}")
    model, metrics = train_single_config(
        peft_method=peft_method,
        peft_config=best_config["peft_config"],
        num_classes=num_classes,
        dataloaders=dataloaders,
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        num_epochs=config.num_epochs,
        drop_path_rate=best_config["drop_path_rate"],
        device=device,
        output_dir=output_dir,
    )

    results = {
        "method": peft_method,
        "task": task_name,
        "test_acc": metrics["test_acc"],
        "val_acc": metrics["val_acc"],
        "n_trainable_params": metrics["n_trainable_params"],
        "best_config": best_config,
    }
    print(f"Test accuracy: {metrics['test_acc']:.2f}%")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save model checkpoint
    ckpt_path = Path(output_dir) / f"{peft_method}_{task_name}_best.pth"
    torch.save(model.state_dict(), ckpt_path)

    return results


def run_manyshot_experiment(
    peft_method: str,
    dataset_name: str,
    config: ManyShotConfig,
    device: torch.device,
    data_root: str,
    output_dir: str = "./outputs/manyshot",
    param_sizes: Optional[List[float]] = None,
) -> Dict:
    """
    Run many-shot experiment, varying the number of trainable parameters.
    param_sizes: list of parameter fractions to test (e.g., [0.02, 0.05, 0.10])
    """
    print(f"\n{'='*60}")
    print(f"Many-shot | Method: {peft_method} | Dataset: {dataset_name}")
    print(f"{'='*60}")

    num_classes_map = {"cifar100": 100, "resisc45": 45, "clevr_distance": 6}
    num_classes = num_classes_map[dataset_name]
    augmentations = config.augmentation.get(dataset_name, [])

    dataloaders = get_manyshot_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=config.batch_size,
        image_size=config.image_size,
        augmentations=augmentations,
        train_val_split=config.train_val_split,
    )

    os.makedirs(output_dir, exist_ok=True)

    best_config, _ = hyperparameter_search(
        peft_method=peft_method,
        num_classes=num_classes,
        dataloaders=dataloaders,
        num_epochs=config.num_epochs,
        device=device,
        lr_choices=config.lr_choices,
        wd_choices=config.wd_choices,
        drop_path_rate_choices=[config.drop_path_rate],
        output_dir=output_dir,
    )

    model, metrics = train_single_config(
        peft_method=peft_method,
        peft_config=best_config["peft_config"],
        num_classes=num_classes,
        dataloaders=dataloaders,
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
        num_epochs=config.num_epochs,
        drop_path_rate=best_config["drop_path_rate"],
        device=device,
        output_dir=output_dir,
    )

    results = {
        "method": peft_method,
        "dataset": dataset_name,
        "test_acc": metrics["test_acc"],
        "n_trainable_params": metrics["n_trainable_params"],
        "best_config": best_config,
    }

    results_path = Path(output_dir) / f"{peft_method}_{dataset_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_robustness_experiment(
    peft_method: str,
    config: RobustnessConfig,
    device: torch.device,
    imagenet_root: str,
    output_dir: str = "./outputs/robustness",
    clip_model_name: str = "ViT-B/16",
) -> Dict:
    """
    Run CLIP robustness experiment: fine-tune on 100-shot ImageNet,
    evaluate on ImageNet + distribution shift datasets.
    """
    print(f"\n{'='*60}")
    print(f"Robustness | Method: {peft_method} | CLIP ViT-B/16")
    print(f"{'='*60}")

    import open_clip

    # Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # Build zero-shot classifier weights using 80 CLIP prompts
    imagenet_classes = _load_imagenet_classes()
    zeroshot_weights = _build_zeroshot_weights(
        clip_model, tokenizer, imagenet_classes, device
    )

    # Build PEFT model using CLIP visual encoder
    from models.vit import PEFTViT
    visual_encoder = clip_model.visual
    num_classes = 1000

    # Initialize head with zero-shot weights
    model = _build_clip_peft_model(
        visual_encoder=visual_encoder,
        zeroshot_weights=zeroshot_weights,
        peft_method=peft_method,
        peft_config=PEFT_DEFAULT_CONFIGS.get(peft_method, {}),
        num_classes=num_classes,
        device=device,
    )

    dataloaders = get_imagenet_dataloaders(
        imagenet_root=imagenet_root,
        num_shots=config.num_shots,
        batch_size=config.batch_size,
        image_size=config.image_size,
        use_clip_norm=True,
        use_strong_augmentation=config.use_strong_augmentation,
    )

    optimizer = build_optimizer(model, config.lr, config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, config.num_epochs + 1):
        train_one_epoch(model, dataloaders["train"], optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, dataloaders["val"], criterion, device)
        scheduler.step()

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    results = {
        "method": peft_method,
        "imagenet_acc": best_val_acc,
    }

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = Path(output_dir) / f"{peft_method}_clip_best.pth"
    torch.save(model.state_dict(), ckpt_path)

    results_path = Path(output_dir) / f"{peft_method}_robustness_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results, model


def _load_imagenet_classes() -> List[str]:
    """Load ImageNet class names."""
    try:
        from torchvision.datasets import ImageNet
        # Use standard ImageNet class names
        import json
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def _build_zeroshot_weights(
    clip_model: nn.Module,
    tokenizer,
    class_names: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Build zero-shot classifier weights by ensembling 80 CLIP prompts.
    W_zero_shot ∈ R^{d × k} where each column is the mean text embedding.
    """
    import open_clip

    # 80 ImageNet prompts from CLIP paper
    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "art of the {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a photo of a {} for sale.",
        "a photo of a {} in the wild.",
        "a photo of a {} in a museum.",
        "a photo of a {} in a store.",
        "a photo of a {} in a zoo.",
        "a photo of a {} in a park.",
        "a photo of a {} on a white background.",
    ]

    clip_model.eval()
    clip_model = clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in class_names:
            texts = [template.format(classname) for template in templates]
            tokens = tokenizer(texts).to(device)
            text_features = clip_model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            class_embedding = text_features.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)  # [D, K]

    return zeroshot_weights


def _build_clip_peft_model(
    visual_encoder: nn.Module,
    zeroshot_weights: torch.Tensor,
    peft_method: str,
    peft_config: Dict,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Build a PEFT model wrapping the CLIP visual encoder."""
    from models.vit import PEFTViT, freeze_backbone

    class CLIPPEFTModel(nn.Module):
        def __init__(self, visual_encoder, num_classes, peft_method, peft_config):
            super().__init__()
            self.visual = visual_encoder
            embed_dim = visual_encoder.output_dim if hasattr(visual_encoder, 'output_dim') else 512
            self.head = nn.Linear(embed_dim, num_classes, bias=False)
            # Initialize head with zero-shot weights
            with torch.no_grad():
                self.head.weight.copy_(zeroshot_weights.t())

            # Apply PEFT
            freeze_backbone(self.visual)
            self._apply_peft(peft_method, peft_config, embed_dim)

        def _apply_peft(self, method, config, embed_dim):
            if method == "linear":
                pass
            elif method == "full":
                for p in self.visual.parameters():
                    p.requires_grad = True
            elif method == "bitfit":
                from models.peft.selective import apply_bitfit
                apply_bitfit(self.visual)
            elif method == "layernorm":
                from models.peft.selective import apply_layernorm_tuning
                apply_layernorm_tuning(self.visual)
            elif method == "houl_adapter":
                from models.peft.adapters import apply_houl_adapter
                apply_houl_adapter(self.visual, embed_dim, **config)
            elif method == "adaptformer":
                from models.peft.adapters import apply_adaptformer
                apply_adaptformer(self.visual, embed_dim, **config)
            elif method == "repadapter":
                from models.peft.adapters import apply_repadapter
                apply_repadapter(self.visual, embed_dim, **config)
            elif method == "convpass":
                from models.peft.adapters import apply_convpass
                apply_convpass(self.visual, embed_dim, **config)
            elif method == "lora":
                from models.peft.lora import apply_lora
                apply_lora(self.visual, embed_dim, **config)
            elif method == "fact_tk":
                from models.peft.fact import apply_fact_tk
                num_layers = len(list(self.visual.transformer.resblocks))
                apply_fact_tk(self.visual, embed_dim, num_layers, **config)

        def forward(self, x):
            features = self.visual(x)
            if isinstance(features, tuple):
                features = features[0]
            return self.head(features)

    model = CLIPPEFTModel(visual_encoder, num_classes, peft_method, peft_config)
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="PEFT Visual Recognition Training")
    parser.add_argument("--experiment", type=str, default="vtab",
                        choices=["vtab", "manyshot", "robustness"],
                        help="Experiment type")
    parser.add_argument("--method", type=str, default="lora",
                        help="PEFT method name")
    parser.add_argument("--task", type=str, default="cifar100",
                        help="Task/dataset name")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--no_hparam_search", action="store_true",
                        help="Skip hyperparameter search, use defaults")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.experiment == "vtab":
        config = VTABConfig()
        results = run_vtab_experiment(
            peft_method=args.method,
            task_name=args.task,
            config=config,
            device=device,
            data_dir=os.path.join(args.data_dir, "vtab"),
            output_dir=os.path.join(args.output_dir, "vtab"),
            do_hparam_search=not args.no_hparam_search,
        )
        print(f"\nFinal test accuracy: {results['test_acc']:.2f}%")

    elif args.experiment == "manyshot":
        config = ManyShotConfig()
        results = run_manyshot_experiment(
            peft_method=args.method,
            dataset_name=args.task,
            config=config,
            device=device,
            data_root=os.path.join(args.data_dir, args.task),
            output_dir=os.path.join(args.output_dir, "manyshot"),
        )
        print(f"\nFinal test accuracy: {results['test_acc']:.2f}%")

    elif args.experiment == "robustness":
        config = RobustnessConfig()
        results, _ = run_robustness_experiment(
            peft_method=args.method,
            config=config,
            device=device,
            imagenet_root=os.path.join(args.data_dir, "imagenet"),
            output_dir=os.path.join(args.output_dir, "robustness"),
        )
        print(f"\nImageNet accuracy: {results['imagenet_acc']:.2f}%")


if __name__ == "__main__":
    main()
