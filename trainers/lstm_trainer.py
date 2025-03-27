import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import time
import itertools
import os
from data_loader.dataloader import RiverFlowDataset, create_data_loader

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)  # Remove last dimension to match target shape

class LSTMTrainer:
    def __init__(self, preprocess_method: str = "z_score"):
        self.preprocess_method = preprocess_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join("results", "lstm")
        os.makedirs(self.results_dir, exist_ok=True)

    def train_model(self, model: LSTMModel, train_loader: DataLoader, 
                   val_loader: DataLoader, params: Dict) -> Dict:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        train_losses, val_losses = [], []
        start_time = time.time()
        
        for epoch in range(params["epochs"]):
            model.train()
            total_batches = len(train_loader)
            epoch_start_time = time.time()
            
            for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward + backward + optimize
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate training speed metrics
                batch_time = time.time() - epoch_start_time
                samples_processed = inputs.size(0) * inputs.size(1)  # batch_size * seq_length
                speed = samples_processed / batch_time if batch_time > 0 else 0
                
                # Calculate progress and ETA
                progress = batch_idx / total_batches * 100
                remaining_batches = total_batches - batch_idx
                eta = (batch_time / batch_idx) * remaining_batches if batch_idx > 0 else 0
                
                # Real-time progress display
                print(f"\rEpoch [{epoch+1}/{params['epochs']}] "
                      f"Batch [{batch_idx}/{total_batches}] "
                      f"Speed: {speed:.2f} samples/sec "
                      f"ETA: {eta:.2f}s   ", end="")
            train_losses.append(loss.item())
            
            val_loss = self.evaluate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            
            # Print epoch summary with loss changes
            prev_val_loss = val_losses[-2] if len(val_losses) > 1 else float('inf')
            improvement = prev_val_loss - val_loss if len(val_losses) > 1 else 0
            direction = "↑" if val_loss > prev_val_loss else "↓"
            
            print(f"\nEpoch {epoch+1}/{params['epochs']} - "
                  f"Train Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_loss:.4f} ({direction}{abs(improvement):.4f})")
            
        # Final training summary
        best_val_loss = min(val_losses)
        total_time = time.time() - start_time
        print(f"\n🏆 Training completed in {total_time/60:.1f} minutes")
        print(f"🏅 Best validation loss: {best_val_loss:.4f}")
        
        return {
            "train_time": total_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss
        }

    def evaluate_model(self, model: LSTMModel, loader: DataLoader, 
                      criterion: nn.Module) -> float:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
        return total_loss / len(loader)

    def kfold_train(self, station_id: str, params: Dict, 
                   n_splits: int = 5) -> List[Dict]:
        dataset = RiverFlowDataset(
            station_id=station_id,
            preprocess_method=self.preprocess_method,
            seq_length=params["sequence_length"]
        )
        
        kfold = KFold(n_splits=n_splits, shuffle=True)
        results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = create_data_loader(train_subset, params["batch_size"])
            val_loader = create_data_loader(val_subset, params["batch_size"], shuffle=False)
            
            model = LSTMModel(
                input_size=dataset[0][0].shape[-1],
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"]
            )
            
            fold_results = self.train_model(model, train_loader, val_loader, params)
            fold_results["fold"] = fold
            results.append(fold_results)
            
            self.save_results(station_id, params, results)
            
        return results

    def grid_search(self, station_id: str, param_grid: Dict) -> Dict:
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        best_score = float("inf")
        best_params = None
        total_combos = len(param_combinations)
        
        print(f"\nStarting grid search with {total_combos} parameter combinations")
        
        for combo_idx, params in enumerate(param_combinations, 1):
            start_time = time.time()
            print(f"\n[{time.strftime('%H:%M:%S')}] Combination {combo_idx}/{total_combos} ")
            print("Parameters:", params)
            
            results = self.kfold_train(station_id, params)
            avg_val_loss = np.mean([r["best_val_loss"] for r in results])
            
            time_used = time.time() - start_time
            remaining = (total_combos - combo_idx) * (time_used / combo_idx) if combo_idx > 0 else 0
            
            print(f"Result: avg_val_loss={avg_val_loss:.4f}  "
                  f"Time: {time_used:.1f}s  "
                  f"ETA: {remaining/60:.1f}m remaining")
            
            if avg_val_loss < best_score:
                best_score = avg_val_loss
                best_params = params
                print("🔥 New best parameters! 🔥")
                
        return {"best_params": best_params, "best_score": best_score}

    def save_results(self, station_id: str, params: Dict, results: List):
        filename = f"{station_id}_{params['hidden_size']}_{params['num_layers']}.pt"
        torch.save({
            "params": params,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, os.path.join(self.results_dir, filename))
