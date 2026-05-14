
import argparse
from config import Config
from train import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train SC-FNO models for various differential equations.")
    parser.add_argument('--model', type=str, default="SC-FNO",
                        choices=["FNO", "FNO-PINN", "SC-FNO", "SC-FNO-PINN"],
                        help="Type of FNO model to train.")
    parser.add_argument('--equation', type=str, default="ODE1",
                        choices=["ODE1", "ODE2", "PDE1", "PDE2", "PDE3", "PDE4", "PDE2_ZONED"],
                        help="Which differential equation to use for training.")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of epochs to train. Overrides config if specified.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for training. Overrides config if specified.")
    parser.add_argument('--num_train_samples', type=int, default=None,
                        help="Number of training samples to generate. Overrides config if specified.")
    
    args = parser.parse_args()

    config = Config()

    # Update config based on command line arguments if provided
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_train_samples is not None:
        config.num_train_samples = args.num_train_samples
        
    config.update_for_equation(args.equation) # Update equation specific configs

    trainer = Trainer(model_name=args.model, equation_name=args.equation, config=config)
    trainer.run()

if __name__ == "__main__":
    main()
