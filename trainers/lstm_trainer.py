import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import time
import itertools
import os

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class LSTMTrainer:
    def __init__(self, preprocess_method: str = "z_score"):
        self.preprocess_method = preprocess_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = os.path.join("results", "lstm")
        os.makedirs(self.results_dir, exist_ok=True)

    def load_dataset(self, station_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        path = f"preprocessed_data/{self.preprocess_method}/{station_id}.csv"
        df = pd.read_csv(path)
        features = df.drop(columns=["datetime", "discharge"]).values
        targets = df["discharge"].values
        return torch.FloatTensor(features), torch.FloatTensor(targets)

    def train_model(self, model: LSTMModel, train_loader: DataLoader, 
                   val_loader: DataLoader, params: Dict) -> Dict:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        
        train_losses, val_losses = [], []
        start_time = time.time()
        
        for epoch in range(params["epochs"]):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_losses.append(loss.item())
            
            val_loss = self.evaluate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            
        return {
            "train_time": time.time() - start_time,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": min(val_losses)
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
        X, y = self.load_dataset(station_id)
        kfold = KFold(n_splits=n_splits, shuffle=True)
        results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
            
            model = LSTMModel(
                input_size=X.shape[2],
                hidden_size=params["hidden_size"],
                num_layers=params["num_layers"],
                output_size=1
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
        
        for params in param_combinations:
            results = self.kfold_train(station_id, params)
            avg_val_loss = np.mean([r["best_val_loss"] for r in results])
            
            if avg_val_loss < best_score:
                best_score = avg_val_loss
                best_params = params
                
        return {"best_params": best_params, "best_score": best_score}

    def save_results(self, station_id: str, params: Dict, results: List):
        filename = f"{station_id}_{params['hidden_size']}_{params['num_layers']}.pt"
        torch.save({
            "params": params,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, os.path.join(self.results_dir, filename))
