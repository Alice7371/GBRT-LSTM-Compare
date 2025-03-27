from trainers import LSTMTrainer
from visualization import visualization as viz
import torch
import argparse

def main():
    # Configure command line arguments
    parser = argparse.ArgumentParser(description='LSTM Model Training')
    parser.add_argument('--preprocess', type=str, default='raw',
                        choices=['raw', 'min_max', 'z_score', 'decimal_scaling'],
                        help='Preprocessing method (default: raw)')
    parser.add_argument('--station', type=str, required=True,
                        help='Station ID (e.g. 01013500)')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Sequence length (default: 30)')
    
    args = parser.parse_args()

    # Parameter grid configuration
    param_grid = {
        "hidden_size": [64, 128],
        "num_layers": [2, 3],
        "learning_rate": [1e-3, 1e-4],
        "batch_size": [32, 64],
        "epochs": [100],
        "sequence_length": [args.sequence_length]
    }

    try:
        # Initialize and run trainer
        trainer = LSTMTrainer(preprocess_method=args.preprocess)
        results = trainer.grid_search(args.station, param_grid)
        
        # Save and visualize results
        output_dir = f"results/lstm/{args.station}"
        viz.plot_training_times(results, param_grid, output_dir)
        viz.visualize_grid_search(results, output_dir)
        
        print(f"Training completed. Results saved to {output_dir}")

    except FileNotFoundError as e:
        print(f"Data error: {str(e)}")
    except torch.cuda.OutOfMemoryError:
        print("GPU memory error - Reduce batch size or sequence length")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
