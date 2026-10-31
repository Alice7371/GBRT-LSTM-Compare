"""Data handling for the scaled-up study: 671 CAMELS basins, 1990-2009.

Layout: data_full/dataset_camels_full/<station>.csv with columns
Date,Dayl,Prcp,Tmax,Tmin,Srad,Vp,Discharge (daily, 7305 rows).

Protocol: chronological 80/20 split per basin; normalization statistics fitted
on the train segment only. All models consume the SAME 7 features so that the
model comparison is not confounded by the feature set.
"""
import numpy as np
import pandas as pd

FEATURES = ["Dayl", "Prcp", "Srad", "Tmax", "Tmin", "Vp", "Discharge"]
TRAIN_FRACTION = 0.8
DATA_DIR = "data_full/dataset_camels_full"
# basins with NaN/inf records (scan of 2026-09-30); negative discharges and
# zero-heavy intermittent streams are kept as genuine hydrology
EXCLUDE = {"06154410", "06291500"}


def load_station(station: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    station = str(station).zfill(8)  # tolerate leading-zero loss through CSV reads
    df = pd.read_csv(f"{data_dir}/{station}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def split_index(n_rows: int) -> int:
    """First 80% train, last 20% test, strictly time-ordered."""
    return int(n_rows * TRAIN_FRACTION)


def window_for(horizon: int) -> int:
    """Input window by lead time: enough history, never confounded with capacity."""
    if horizon <= 7:
        return 15
    if horizon <= 15:
        return 60
    if horizon <= 30:
        return 90
    return 270  # h in {90, 180}: cover a full snow accumulation season


def fit_normalizer(values: np.ndarray, method: str):
    """(fwd, inv) transforms with statistics from the train segment only."""
    if method == "minmax":
        lo, hi = values.min(axis=0), values.max(axis=0)
        rng = np.where(hi - lo == 0, 1.0, hi - lo)
        return (lambda x: (x - lo) / rng), (lambda x: x * rng + lo)
    if method == "zscore":
        mu, sd = values.mean(axis=0), values.std(axis=0)
        sd = np.where(sd == 0, 1e-8, sd)
        return (lambda x: (x - mu) / sd), (lambda x: x * sd + mu)
    raise ValueError(f"unsupported normalization: {method}")


def make_flat_windows(values: np.ndarray, window: int) -> np.ndarray:
    """(N, window * F) flattened windows for tree models."""
    return np.stack([values[i:i + window].ravel() for i in range(len(values) - window)])


def make_sequences(features: np.ndarray, window: int, horizon: int = 1):
    """(N, window, F) sequences; label = discharge at t+horizon (direct multi-step).

    Sample i uses rows [i, i+window) as input and targets row i+window-1+horizon.
    Callers must keep training targets strictly inside the train segment.
    """
    n = len(features) - window - horizon + 1
    last_col = features.shape[1] - 1  # Discharge is the last column
    X = np.stack([features[i:i + window] for i in range(n)])
    y = features[window - 1 + horizon: window - 1 + horizon + n, last_col]
    return X.astype(np.float32), y.astype(np.float32)


def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe efficiency (== R2 against the observed mean)."""
    err = y_true - y_pred
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def kge(y_true, y_pred) -> float:
    """Gupta et al. (2009) Kling-Gupta efficiency."""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = y_pred.std() / (y_true.std() + 1e-9)
    beta = y_pred.mean() / (y_true.mean() + 1e-9)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true, y_pred) -> dict:
    return {"nse": nse(y_true, y_pred), "kge": kge(y_true, y_pred),
            "mape": mape(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
