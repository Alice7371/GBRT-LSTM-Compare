import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import Dict, List

def plot_training_times(results: List[Dict], param_grid: Dict, save_path: str = None):
    plt.figure(figsize=(12, 8))
    times = [r["train_time"] for r in results]
    params = [f"HS:{p['hidden_size']}\nLY:{p['num_layers']}" for p in [r["params"] for r in results]]
    
    plt.bar(range(len(times)), times, tick_label=params)
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Hyperparameters")
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(os.path.join(save_path, "training_times.png"), bbox_inches="tight")
    plt.close()

def plot_loss_curves(results: List[Dict], save_path: str = None):
    plt.figure(figsize=(12, 8))
    for result in results:
        plt.plot(result["train_losses"], label=f'Fold {result["fold"]} Train')
        plt.plot(result["val_losses"], '--', label=f'Fold {result["fold"]} Val')
    
    plt.title("Training/Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "loss_curves.png"), bbox_inches="tight")
    plt.close()

def visualize_grid_search(grid_results: Dict, save_path: str = None):
    plt.figure(figsize=(12, 8))
    scores = [res["best_score"] for res in grid_results]
    params = [f"LR:{res['params']['learning_rate']:.0e}\nBS:{res['params']['batch_size']}" 
             for res in grid_results]
    
    plt.scatter(range(len(scores)), scores, c=scores, cmap='viridis', s=100)
    plt.colorbar(label="Validation Loss")
    plt.xticks(range(len(params)), params, rotation=45)
    plt.title("Grid Search Performance")
    plt.ylabel("Best Validation Loss")
    
    if save_path:
        plt.savefig(os.path.join(save_path, "grid_search.png"), bbox_inches="tight")
    plt.close()

# Existing functions remain unchanged
