"""Training loop for neural operators.

Implements:
- NMAE metric (Equation 3)
- Standard training from scratch
- Pretraining on multi-physics datasets
- Fine-tuning with frozen core + trainable adapters
- Training statistics collection for Tables 1 & 2
"""

import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models import get_model, MultiPhysicsNO
from layers import Lift, Project
from data import load_pde_data, load_multiphysics_data, get_dataloader


# ============================================================================
#  Metric: Range-Normalized Mean Absolute Error (Equation 3)
# ============================================================================

def nmae(pred, target, eps=1e-8):
    """
    Compute NMAE as in Eq. (3):
        NMAE = |pred - target|_1,G / (max_G(target) - min_G(target) + eps)
    Averaged over batch and spatial grid G.
    """
    # pred, target: (B, out_channels, ...)
    batch_size = pred.shape[0]
    t_flat = target.reshape(batch_size, -1)
    p_flat = pred.reshape(batch_size, -1)
    range_t = t_flat.max(dim=-1).values - t_flat.min(dim=-1).values + eps
    mae = (p_flat - t_flat).abs().mean(dim=-1)
    return (mae / range_t).mean().item()


def mse(pred, target):
    return nn.functional.mse_loss(pred, target).item()


# ============================================================================
#  Single-problem training (from scratch)
# ============================================================================

def train_single(config, model=None):
    """
    Train a single neural operator on one PDE problem.

    Args:
        config: configuration dict
        model: optional pre-built model

    Returns:
        model: trained model
        metrics: dict with training statistics
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model_type = config.get('model_type', 'fno')

    if model is None:
        model_cfg = dict(config)
        model_cfg['in_channels'] = config['in_channels']
        model_cfg['out_channels'] = config['out_channels']
        model = get_model(model_type, model_cfg)
    model = model.to(device)

    # Data
    train_ds = load_pde_data(config, split='train')
    test_ds = load_pde_data(config, split='test')
    train_loader = get_dataloader(train_ds, config.get('batch_size', 16), shuffle=True)
    test_loader = get_dataloader(test_ds, config.get('batch_size', 16), shuffle=False)

    # Optimizer
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler_type = config.get('scheduler', 'step')
    n_epochs = config.get('n_epochs', 100)
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3, gamma=0.5)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        scheduler = None

    loss_fn = nn.MSELoss()

    best_mse = float('inf')
    best_nmae_val = float('inf')
    train_mse_history = []
    test_mse_history = []
    test_nmae_history = []
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []

        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            pred = model(batch_input)
            loss = loss_fn(pred, batch_target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        train_mse_history.append(avg_train_loss)

        if scheduler is not None:
            scheduler.step()

        # Evaluation
        model.eval()
        test_losses = []
        test_nmaes = []
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                pred = model(batch_input)
                test_losses.append(mse(pred, batch_target))
                test_nmaes.append(nmae(pred, batch_target))

        avg_test_mse = np.mean(test_losses)
        avg_test_nmae = np.mean(test_nmaes)
        test_mse_history.append(avg_test_mse)
        test_nmae_history.append(avg_test_nmae)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if avg_test_mse < best_mse:
            best_mse = avg_test_mse
            best_nmae_val = avg_test_nmae

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[scratch {model_type}] Epoch {epoch:3d}: "
                  f"Train MSE={avg_train_loss:.2e}, Test MSE={avg_test_mse:.2e}, "
                  f"NMAE={avg_test_nmae:.6f}, Time={epoch_time:.2f}s")

    avg_epoch_time = np.mean(epoch_times)
    n_params = sum(p.numel() for p in model.parameters())

    metrics = {
        'model_type': model_type,
        'mode': 'scratch',
        'best_mse': best_mse,
        'best_nmae': best_nmae_val,
        'avg_epoch_time': avg_epoch_time,
        'n_params': n_params,
        'final_train_mse': train_mse_history[-1],
        'final_test_mse': test_mse_history[-1],
        'final_test_nmae': test_nmae_history[-1],
        'test_mse_history': test_mse_history,
        'test_nmae_history': test_nmae_history,
    }
    return model, metrics


# ============================================================================
#  Multi-physics pretraining  (Section 3, Figure 1)
# ============================================================================

def pretrain_multiphysics(config):
    """
    Pretrain a MultiPhysicsNO on multiple PDE problems simultaneously.

    Args:
        config: dict with:
            - core_config: core body architecture
            - problems: dict {name: {in_channels, out_channels, ...}}
            - train_config: {lr, n_epochs, batch_size, ...}

    Returns:
        model: trained MultiPhysicsNO
        metrics: training statistics
    """
    device = torch.device(config['train_config'].get('device',
                         'cuda' if torch.cuda.is_available() else 'cpu'))
    core_config = config['core_config']
    problem_configs_raw = config['problems']

    # Build problem configs list for MultiPhysicsNO
    problem_configs = []
    for name, pcfg in problem_configs_raw.items():
        pc = dict(pcfg)
        pc['name'] = name
        pc['in_channels'] = pcfg['in_channels']
        pc['out_channels'] = pcfg['out_channels']
        problem_configs.append(pc)

    model = MultiPhysicsNO(core_config, problem_configs)
    model = model.to(device)

    # Build dataloaders for each problem
    dataloaders = {}
    test_dataloaders = {}
    for name, pcfg in problem_configs_raw.items():
        pcfg_full = dict(pcfg)
        pcfg_full['problem'] = pcfg.get('problem', name)
        pcfg_full['device'] = device
        train_ds = load_pde_data(pcfg_full, split='train')
        test_ds = load_pde_data(pcfg_full, split='test')
        dataloaders[name] = get_dataloader(train_ds, config['train_config'].get('batch_size', 16))
        test_dataloaders[name] = get_dataloader(test_ds, config['train_config'].get('batch_size', 16))

    train_cfg = config['train_config']
    lr = train_cfg.get('lr', 1e-3)
    n_epochs = train_cfg.get('n_epochs', 100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=train_cfg.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_mse = float('inf')
    best_nmae_val = float('inf')
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []

        # Round-robin through problems
        for name in dataloaders:
            for batch_input, batch_target in dataloaders[name]:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                optimizer.zero_grad()
                pred = model(batch_input, problem_name=name)
                loss = loss_fn(pred, batch_target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        scheduler.step()

        # Evaluate on all problems
        model.eval()
        all_mse = []
        all_nmae = []
        with torch.no_grad():
            for name in test_dataloaders:
                for batch_input, batch_target in test_dataloaders[name]:
                    batch_input = batch_input.to(device)
                    batch_target = batch_target.to(device)
                    pred = model(batch_input, problem_name=name)
                    all_mse.append(mse(pred, batch_target))
                    all_nmae.append(nmae(pred, batch_target))

        avg_mse = np.mean(all_mse)
        avg_nmae = np.mean(all_nmae)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_nmae_val = avg_nmae

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[pretrain] Epoch {epoch:3d}: Train MSE={avg_train_loss:.2e}, "
                  f"Test MSE={avg_mse:.2e}, NMAE={avg_nmae:.6f}, Time={epoch_time:.2f}s")

    avg_epoch_time = np.mean(epoch_times)
    n_params = sum(p.numel() for p in model.parameters())

    metrics = {
        'mode': 'pretrain',
        'best_mse': best_mse,
        'best_nmae': best_nmae_val,
        'avg_epoch_time': avg_epoch_time,
        'n_params': n_params,
    }
    return model, metrics


# ============================================================================
#  Fine-tuning (Section 3)  –  only adapters trained
# ============================================================================

def finetune(config, pretrained_model):
    """
    Fine-tune a pretrained MultiPhysicsNO on a new/different PDE problem.
    Only adapter parameters (lift + project for the new problem) are trained;
    core body parameters are frozen.

    Args:
        config: dict with new problem configuration
        pretrained_model: trained MultiPhysicsNO

    Returns:
        model: fine-tuned model
        metrics: training statistics
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    pretrained_model = pretrained_model.to(device)
    pretrained_model.freeze_core()

    # Add new problem adapters
    new_name = config['problem_name']
    hidden_channels = pretrained_model.hidden_channels
    lift_mode = config.get('lift_mode', 'mlp')

    pretrained_model.lifts[new_name] = Lift(
        config['in_channels'], hidden_channels, mode=lift_mode
    ).to(device)

    pretrained_model.projects[new_name] = Project(
        hidden_channels, config['out_channels'], mode=lift_mode
    ).to(device)

    # Only adapter params are trainable (core is frozen)
    trainable_params = (
        list(pretrained_model.lifts[new_name].parameters()) +
        list(pretrained_model.projects[new_name].parameters())
    )

    # Data
    train_ds = load_pde_data(config, split='train')
    test_ds = load_pde_data(config, split='test')
    train_loader = get_dataloader(train_ds, config.get('batch_size', 16))
    test_loader = get_dataloader(test_ds, config.get('batch_size', 16))

    lr = config.get('lr', 1e-3)
    n_epochs = config.get('n_epochs', 50)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr,
                                  weight_decay=config.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 3, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_mse = float('inf')
    best_nmae_val = float('inf')
    epoch_times = []

    for epoch in range(n_epochs):
        epoch_start = time.time()
        pretrained_model.train()
        train_losses = []

        for batch_input, batch_target in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            pred = pretrained_model(batch_input, problem_name=new_name)
            loss = loss_fn(pred, batch_target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        scheduler.step()

        pretrained_model.eval()
        test_mse_vals = []
        test_nmae_vals = []
        with torch.no_grad():
            for batch_input, batch_target in test_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                pred = pretrained_model(batch_input, problem_name=new_name)
                test_mse_vals.append(mse(pred, batch_target))
                test_nmae_vals.append(nmae(pred, batch_target))

        avg_test_mse = np.mean(test_mse_vals)
        avg_test_nmae = np.mean(test_nmae_vals)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if avg_test_mse < best_mse:
            best_mse = avg_test_mse
            best_nmae_val = avg_test_nmae

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"[finetune {new_name}] Epoch {epoch:3d}: "
                  f"Train MSE={avg_train_loss:.2e}, Test MSE={avg_test_mse:.2e}, "
                  f"NMAE={avg_test_nmae:.6f}, Time={epoch_time:.2f}s")

    avg_epoch_time = np.mean(epoch_times)
    metrics = {
        'mode': 'finetune',
        'best_mse': best_mse,
        'best_nmae': best_nmae_val,
        'avg_epoch_time': avg_epoch_time,
    }
    return pretrained_model, metrics


# ============================================================================
#  Experiment runner – produces Table 1 and Table 2 style results
# ============================================================================

def run_experiment_table1(config):
    """
    Run "Out-of-sample parameter values" experiments (Table 1).
    Compares pretrained + fine-tuned vs training from scratch on
    Burgers, Gray-Scott, and Navier-Stokes with varied parameters.
    """
    results = []
    problems = config.get('table1_problems', ['burgers', 'grayscott', 'navierstokes'])
    models_to_test = config.get('models', ['mamba_fno', 'perceiver_io_fno', 'fno', 'codano'])

    for problem in problems:
        problem_cfg = config['problem_configs'][problem]
        base_cfg = dict(problem_cfg)
        base_cfg['problem'] = problem

        # Varied-parameter versions for pretraining
        pretrain_problems = {}
        for pretrain_var in problem_cfg.get('pretrain_variants', [problem]):
            pc = dict(problem_cfg)
            pc['problem'] = pretrain_var
            pc['name'] = pretrain_var
            pretrain_problems[pretrain_var] = pc

        for model_type in models_to_test:
            core_cfg = dict(config['core_config'])
            core_cfg['model_type'] = model_type

            # ---- Pretraining + Fine-tuning ----
            pt_cfg = {
                'core_config': core_cfg,
                'problems': pretrain_problems,
                'train_config': config['pretrain_config'],
            }
            pretrained_model, pt_metrics = pretrain_multiphysics(pt_cfg)

            ft_cfg = dict(base_cfg)
            ft_cfg['problem_name'] = f'{problem}_ft'
            ft_cfg['n_epochs'] = config['finetune_config'].get('n_epochs', 50)
            ft_cfg['lr'] = config['finetune_config'].get('lr', 1e-3)
            _, ft_metrics = finetune(ft_cfg, pretrained_model)

            results.append({
                'model': model_type,
                'problem': problem,
                'mode': 'pretrained',
                'mse': ft_metrics['best_mse'],
                'nmae': ft_metrics['best_nmae'],
                'avg_epoch_time': ft_metrics['avg_epoch_time'],
            })

            # ---- From scratch ----
            scratch_cfg = dict(base_cfg)
            scratch_cfg['model_type'] = model_type
            scratch_cfg['n_epochs'] = config['scratch_config'].get('n_epochs', 100)
            scratch_cfg['lr'] = config['scratch_config'].get('lr', 1e-3)
            _, scratch_metrics = train_single(scratch_cfg)

            results.append({
                'model': model_type,
                'problem': problem,
                'mode': 'scratch',
                'mse': scratch_metrics['best_mse'],
                'nmae': scratch_metrics['best_nmae'],
                'avg_epoch_time': scratch_metrics['avg_epoch_time'],
            })

    return results


def run_experiment_table2(config):
    """
    Run "Input function set extension" and "Multi-physics learning" (Table 2).
    Tests heat equation extension with convection, and reaction-diffusion with
    advection. Also tests transfer from advection+Burgers to reaction-diffusion.
    """
    results = []

    # ---- Heat extension ----
    heat_base_cfg = dict(config['problem_configs']['heat'])
    heat_conv_cfg = dict(config['problem_configs']['heat_convection'])
    heat_conv_cfg['problem'] = 'heat_convection'

    # ---- RD extension ----
    rd_base_cfg = dict(config['problem_configs']['rd'])
    rd_adv_cfg = dict(config['problem_configs']['rd_advection'])

    # ---- Multi-physics transfer ----
    mp_source = config.get('multiphysics_source', ['advection', 'burgers'])
    mp_target = config.get('multiphysics_target', 'rd')

    extension_tasks = [
        {
            'name': 'heat_extension',
            'pretrain_on': heat_base_cfg,
            'finetune_on': heat_conv_cfg,
        },
        {
            'name': 'rd_extension',
            'pretrain_on': rd_base_cfg,
            'finetune_on': rd_adv_cfg,
        },
        {
            'name': 'multiphysics_transfer',
            'pretrain_on': {p: config['problem_configs'][p] for p in mp_source},
            'finetune_on': config['problem_configs'][mp_target],
        },
    ]

    models_to_test = config.get('table2_models',
                                ['mamba_fno', 'perceiver_io_fno', 'fno', 'codano'])

    for task in extension_tasks:
        for model_type in models_to_test:
            core_cfg = dict(config['core_config'])
            core_cfg['model_type'] = model_type

            # Pretraining
            if isinstance(task['pretrain_on'], dict):
                pretrain_problems = {}
                for pname, pcfg in task['pretrain_on'].items():
                    pc = dict(pcfg)
                    pc['name'] = pname
                    if 'problem' not in pc:
                        pc['problem'] = pname
                    pretrain_problems[pname] = pc
            else:
                pname = task['pretrain_on'].get('problem', task['name'])
                pc = dict(task['pretrain_on'])
                pc['name'] = pname
                pretrain_problems = {pname: pc}

            pt_cfg = {
                'core_config': core_cfg,
                'problems': pretrain_problems,
                'train_config': config['pretrain_config'],
            }
            pretrained_model, pt_metrics = pretrain_multiphysics(pt_cfg)

            # Fine-tuning
            ft_cfg = dict(task['finetune_on'])
            ft_cfg['problem_name'] = f"{task['name']}_ft"
            ft_cfg['n_epochs'] = config['finetune_config'].get('n_epochs', 50)
            ft_cfg['lr'] = config['finetune_config'].get('lr', 1e-3)
            _, ft_metrics = finetune(ft_cfg, pretrained_model)

            results.append({
                'task': task['name'],
                'model': model_type,
                'mode': 'pretrained',
                'mse': ft_metrics['best_mse'],
                'nmae': ft_metrics['best_nmae'],
                'avg_epoch_time': ft_metrics['avg_epoch_time'],
            })

            # From scratch (on finetune target)
            scratch_cfg = dict(task['finetune_on'])
            scratch_cfg['model_type'] = model_type
            scratch_cfg['n_epochs'] = config['scratch_config'].get('n_epochs', 100)
            scratch_cfg['lr'] = config['scratch_config'].get('lr', 1e-3)
            _, scratch_metrics = train_single(scratch_cfg)

            results.append({
                'task': task['name'],
                'model': model_type,
                'mode': 'scratch',
                'mse': scratch_metrics['best_mse'],
                'nmae': scratch_metrics['best_nmae'],
                'avg_epoch_time': scratch_metrics['avg_epoch_time'],
            })

    return results


# ============================================================================
#  Main training entry point (used by train.py and main.py)
# ============================================================================

def train(config):
    """
    Main training entry point.
    Config dict determines whether to run:
    - single problem training
    - multi-physics pretraining
    - fine-tuning
    - full experiment suite
    """
    mode = config.get('mode', 'train_single')

    if mode == 'train_single':
        return train_single(config)

    elif mode == 'pretrain':
        return pretrain_multiphysics(config)

    elif mode == 'finetune':
        pretrained = MultiPhysicsNO(
            config['core_config'],
            [{'name': config['finetune_config'].get('source_problem', 'source'),
              'in_channels': config['finetune_config'].get('in_channels', 2),
              'out_channels': config['finetune_config'].get('out_channels', 1)}]
        )
        ft_cfg = dict(config['finetune_config'])
        ft_cfg['problem_name'] = config.get('target_problem', 'target')
        return finetune(ft_cfg, pretrained)

    elif mode == 'table1':
        return run_experiment_table1(config)

    elif mode == 'table2':
        return run_experiment_table2(config)

    else:
        raise ValueError(f"Unknown mode: {mode}")
