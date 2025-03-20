import pandas as pd
import torch
from typing import Tuple

class DataLoader:
    def __init__(self, preprocess_method: str = "z_score"):
        self.preprocess_method = preprocess_method
        
    def load_station_data(self, station_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        path = f"preprocessed_data/{self.preprocess_method}/{station_id}.csv"
        df = pd.read_csv(path)
        features = df.drop(columns=["datetime", "discharge"]).values
        targets = df["discharge"].values
        return torch.FloatTensor(features), torch.FloatTensor(targets)

    def create_sequences(self, data: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
            targets.append(data[i+seq_length])
        return torch.stack(sequences), torch.stack(targets)
