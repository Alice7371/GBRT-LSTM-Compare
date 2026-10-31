"""LSTM track: single-layer-head recurrent network with Min-Max / Z-Score input.

Final configuration (thesis, ch. 3 & 5): 7 driving features per timestep
(Swe excluded), hidden size 128, 2 stacked LSTM layers, dropout 0.1,
SmoothL1Loss, Adam (lr 1e-3, weight decay 1e-4), StepLR halving the learning
rate every 50 epochs, 100 epochs, batch size 512. Normalization statistics are
computed on the train segment only; metrics are computed after de-normalizing.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from common import LSTM_FEATURES, fit_normalizer, make_sequences, metrics

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
STEP_SIZE = 50       # halve the learning rate every 50 epochs
STEP_GAMMA = 0.5


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def run_lstm(df: pd.DataFrame, window: int, norm: str, out_dir: str,
             seed: int = 0, device: str = None, num_epochs: int = None) -> dict:
    epochs = num_epochs or NUM_EPOCHS
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    features = df[LSTM_FEATURES].values.astype(np.float32)
    discharge = df["Discharge"].values.astype(np.float32)
    split = int(len(df) * 0.8)

    # normalize with train-segment statistics only (thesis sec. 4.1.4);
    # the discharge column gets its own stats so the target de-normalizes correctly
    fwd, _ = fit_normalizer(features[:split], norm)
    _, inv_q = fit_normalizer(discharge[:split].reshape(-1, 1), norm)
    features_n = fwd(features)

    X_train, y_train = make_sequences(features_n[:split], features_n[:split, -1], window)
    X_test, y_test = make_sequences(features_n[split - window:], features_n[split - window:, -1], window)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = LSTMRegressor(len(LSTM_FEATURES)).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1}/{epochs} loss={np.mean(losses):.6f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X_train).to(device)).cpu().numpy()
        test_pred = model(torch.tensor(X_test).to(device)).cpu().numpy()

    y_train_true, y_test_true = inv_q(y_train), inv_q(y_test)
    result = {
        "train": metrics(y_train_true, inv_q(train_pred)),
        "test": metrics(y_test_true, inv_q(test_pred)),
    }

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"true": y_test_true, "pred": inv_q(test_pred)}).to_csv(
        f"{out_dir}/predictions.csv", index=False
    )
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    return result
