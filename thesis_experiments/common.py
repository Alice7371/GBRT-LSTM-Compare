"""Shared data handling for the GBRT / LSTM runoff-prediction comparison.

Data: CAMELS-derived daily records for 27 basins, 2000-01-01..2004-12-31.
Each station CSV has columns Discharge, Dayl, Prcp, Srad, Swe, Tmax, Tmin, Vp, Date.

Protocol (thesis, ch. 4): chronological 80/20 train/test split (train = 2000-2003,
test = 2004), normalization fitted on the TRAIN segment only, targets de-normalized
before computing metrics.
"""
import numpy as np
import pandas as pd

# GBRT consumes all 8 numeric drivers as a flattened window (incl. Swe).
GBRT_FEATURES = ["Discharge", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp"]
# LSTM consumes 7 drivers as a sequence (Swe excluded, as in the final experiments).
LSTM_FEATURES = ["Dayl", "Prcp", "Srad", "Tmax", "Tmin", "Vp", "Discharge"]

TRAIN_FRACTION = 0.8
WINDOWS = [5, 15, 30, 50, 70, 90]  # history window lengths swept in the thesis


def load_station(station: str, data_dir: str = "dataset") -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/{station}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def chronological_split(n_rows: int):
    """First 80% train, last 20% test — time-ordered, no shuffling."""
    return int(n_rows * TRAIN_FRACTION)


def fit_normalizer(values: np.ndarray, method: str):
    """Return (stats, transform fn, inverse fn); stats come from the train segment."""
    if method == "minmax":
        lo, hi = values.min(axis=0), values.max(axis=0)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        fwd = lambda x: (x - lo) / rng
        inv = lambda x: x * rng + lo
    elif method == "zscore":
        mu, sd = values.mean(axis=0), values.std(axis=0)
        sd = np.where(sd == 0, 1e-8, sd)
        fwd = lambda x: (x - mu) / sd
        inv = lambda x: x * sd + mu
    else:
        raise ValueError(f"unsupported normalization: {method}")
    return fwd, inv


def make_windows(values: np.ndarray, window: int) -> np.ndarray:
    """Sliding windows over rows; window of rows -> one flattened vector (GBRT)."""
    return np.stack([values[i:i + window].ravel() for i in range(len(values) - window)])


def make_sequences(features: np.ndarray, discharge: np.ndarray, window: int):
    """Sliding sequences for the LSTM: X (N, window, F) predicts the next discharge."""
    X = np.stack([features[i:i + window] for i in range(len(features) - window)])
    y = discharge[window:]
    return X.astype(np.float32), y.astype(np.float32)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2 / MAPE / RMSE / MAE, computed on de-normalized values (ft3/s)."""
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mape": float(np.mean(np.abs(err / (y_true + 1e-6))) * 100),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
