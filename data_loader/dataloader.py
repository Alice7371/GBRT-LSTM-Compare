import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RiverFlowDataset(Dataset):
    def __init__(self, station_id: str, preprocess_method: str = "z_score", seq_length: int = 30):
        self.station_id = station_id
        self.preprocess_method = preprocess_method
        self.seq_length = seq_length
        
        # Load and preprocess data
        path = f"preprocessed_data/{preprocess_method}/{station_id}.csv"
        df = pd.read_csv(path)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Final NaN check
        if df.isnull().values.any():
            raise ValueError(f"NaN values remain in {path} after cleaning")

        # Validate we have enough data for sequences
        if len(df) < self.seq_length * 2:
            raise ValueError(f"Not enough data points ({len(df)}) for sequence length {self.seq_length} in {path}")
        
        # Extract features and targets
        feature_columns = df.columns.tolist()  # 包含所有列作为特征
        features = df[feature_columns].values.astype(np.float32)
        targets = df['Discharge'].values.astype(np.float32)
        
        # Create sequences
        self.sequences, self.labels = self._create_sequences(features, targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def _create_sequences(self, features, targets):
        sequences = []
        labels = []
        for i in range(len(features) - self.seq_length):
            sequences.append(features[i:i+self.seq_length])
            labels.append(targets[i+self.seq_length])
        return np.array(sequences), np.array(labels)

def create_data_loader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
