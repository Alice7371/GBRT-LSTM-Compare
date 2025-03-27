import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RiverFlowDataset(Dataset):
    def __init__(self, station_id: str, preprocess_method: str = "z_score", seq_length: int = 30):
        self.station_id = station_id
        self.preprocess_method = preprocess_method
        self.seq_length = seq_length
        
        # Load data with format handling
        if preprocess_method == "raw":
            # 原始数据使用固定宽度格式解析
            column_widths = [9, 6, 6, 6, 6, 6, 6, 6, 10]
            column_names = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp", "Date"]
            path = f"dataset/{station_id}.csv"
            df = pd.read_fwf(path, widths=column_widths, names=column_names)
            
            # 仅保留数值型特征并转换
            numeric_cols = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
            df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Handle missing values with forward/backward fill and zero fill
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df.fillna(0, inplace=True)
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RiverFlowDataset(Dataset):
    def __init__(self, station_id: str, preprocess_method: str = "z_score", seq_length: int = 30):
        self.station_id = station_id
        self.preprocess_method = preprocess_method
        self.seq_length = seq_length
        
        # Load data with format handling
        if preprocess_method == "raw":
            # 原始数据使用固定宽度格式解析
            column_widths = [9, 6, 6, 6, 6, 6, 6, 6, 10]
            column_names = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp", "Date"]
            path = f"dataset/{station_id}.csv"
            df = pd.read_fwf(path, widths=column_widths, names=column_names)
            
            # 仅保留数值型特征并转换
            numeric_cols = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
            df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # 原始数据仅处理极端缺失值
            df.dropna(inplace=True)
        else:
            # 预处理数据使用标准CSV加载
            path = f"preprocessed_data/{preprocess_method}/{station_id}.csv"
            df = pd.read_csv(path)
            # 预处理数据需要填充处理
            df = df.apply(pd.to_numeric, errors='coerce')
            df.ffill(inplace=True)
            df.bfill(inplace=True)
        
        # Final NaN check
        if df.isnull().values.any():
            raise ValueError(f"NaN values remain in {path} after cleaning")

        # Validate data after cleaning
        if len(df) == 0:
            raise ValueError(f"All data rows invalid after processing in {path}")
            
        # Validate we have enough data for sequences (need at least seq_length + 1 points)
        if len(df) < self.seq_length + 1:
            raise ValueError(f"Not enough data points ({len(df)}) for sequence length {self.seq_length} (minimum required: {self.seq_length + 1}) in {path}")
        
        # Extract features and targets
        if self.preprocess_method == "raw":
            # 原始数据使用全部数值列作为特征
            feature_columns = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
        else:
            # 预处理数据使用所有列
            feature_columns = df.columns.tolist()
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
