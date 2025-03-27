from trainers import LSTMTrainer
from visualization import visualization as viz
from data_loader.dataloader import DataLoader
import torch

if __name__ == "__main__":
    # Example configuration
    station_ids = ["01013500", "01022500", "01030500"]  # 可修改需要分析的站点列表
    param_grid = {
        "hidden_size": [64, 128],
        "num_layers": [2, 3],
        "learning_rate": [1e-3, 1e-4],
        "batch_size": [32, 64],
        "epochs": [100],
        "sequence_length": [30]
    }

    try:
        # Initialize and run trainer
        trainer = LSTMTrainer(preprocess_method="z_score")
        for station_id in station_ids:
            grid_results = trainer.grid_search(station_id, param_grid)
            
            # Visualize results
            output_dir = f"results/lstm/{station_id}"
            viz.plot_training_times([r for r in grid_results], param_grid, output_dir)
            viz.visualize_grid_search(grid_results, output_dir)
        
        print("Training completed successfully. Check results/lstm directory for outputs.")
        
    except FileNotFoundError as e:
        print(f"Data loading failed: {str(e)}")
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory - Try reducing batch size or sequence length")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
