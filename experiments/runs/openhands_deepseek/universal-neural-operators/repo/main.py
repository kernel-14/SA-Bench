"""Main entry point for reproducing the experiments from:
"Towards Universal Neural Operators through Multiphysics Pretraining"

Run modes:
  python main.py --mode train_single --problem burgers --model fno
  python main.py --mode table1
  python main.py --mode table2
"""

import argparse
import torch
from config import get_config, BURGERS_CONFIG, GRAYSCOTT_CONFIG, NAVIERSTOKES_CONFIG
from train import train


def main():
    parser = argparse.ArgumentParser(description='Universal Neural Operators')
    parser.add_argument('--mode', type=str, default='train_single',
                        choices=['train_single', 'pretrain', 'finetune',
                                 'table1', 'table2'],
                        help='Experiment mode')
    parser.add_argument('--problem', type=str, default='burgers',
                        choices=['burgers', 'grayscott', 'navierstokes',
                                 'heat', 'heat_convection', 'rd_advection', 'advection'],
                        help='PDE problem to solve')
    parser.add_argument('--model', type=str, default='fno',
                        choices=['fno', 'mamba_fno', 'local_attn_fno',
                                 'perceiver_io_fno', 'codano', 'swinv2_fno'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.mode == 'table1':
        print("=" * 60)
        print("Running Table 1 experiments: Out-of-sample parameter values")
        print("=" * 60)
        config = get_config('table1')
        results = train(config)
        print_results_table1(results)

    elif args.mode == 'table2':
        print("=" * 60)
        print("Running Table 2 experiments: Input extension & multi-physics")
        print("=" * 60)
        config = get_config('table2')
        results = train(config)
        print_results_table2(results)

    elif args.mode == 'train_single':
        problem_configs = {
            'burgers': BURGERS_CONFIG,
            'grayscott': GRAYSCOTT_CONFIG,
            'navierstokes': NAVIERSTOKES_CONFIG,
        }
        if args.problem not in problem_configs:
            raise ValueError(f"Unknown problem: {args.problem}")

        config = dict(problem_configs[args.problem])
        config['model_type'] = args.model
        config['mode'] = 'train_single'
        config['n_epochs'] = args.epochs
        config['batch_size'] = args.batch_size
        config['lr'] = args.lr
        config['device'] = args.device

        print(f"Training {args.model} on {args.problem} from scratch")
        _, metrics = train(config)
        print(f"\nBest MSE: {metrics['best_mse']:.6e}")
        print(f"Best NMAE: {metrics['best_nmae']:.6f}")
        print(f"Avg epoch time: {metrics['avg_epoch_time']:.2f}s")
        print(f"Model parameters: {metrics['n_params']}")

    else:
        print(f"Mode '{args.mode}' requires additional configuration.")
        print("Use the config.py module to build custom experiment configs.")


def print_results_table1(results):
    """Print results in Table 1 format."""
    print("\n" + "=" * 80)
    print("Table 1: Out-of-sample parameter values results")
    print("=" * 80)
    print(f"{'Model':<20} {'Mode':<12} {'MSE':<16} {'NMAE (%)':<12} {'Time (s)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<20} {r['mode']:<12} {r['mse']:<16.3e} "
              f"{r['nmae']*100:<12.4f} {r['avg_epoch_time']:<12.2f}")


def print_results_table2(results):
    """Print results in Table 2 format."""
    print("\n" + "=" * 80)
    print("Table 2: Input extension & multi-physics results")
    print("=" * 80)
    print(f"{'Task':<25} {'Model':<20} {'Mode':<12} {'MSE':<16} {'NMAE (%)':<12} {'Time (s)':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['task']:<25} {r['model']:<20} {r['mode']:<12} {r['mse']:<16.3e} "
              f"{r['nmae']*100:<12.4f} {r['avg_epoch_time']:<12.2f}")


if __name__ == '__main__':
    main()
