"""Feature engineering: same-day derived features using ONLY information up to day t.

All rolling/expanding quantities look backward (inclusive of day t); recursive
states (API) are causal by construction. Feeds two model variants:
- XGB-FE: the engineered vector at day t predicts q(t+h), no long flatten needed
- LSTM-FE: a subset of engineered channels appended to the raw input window
"""
import numpy as np
import pandas as pd

# channels appended to the LSTM input window (lags are redundant for a sequence
# model — window already contains them; state/accumulation/seasonality are not)
LSTM_EXTRA = ["doy_sin", "doy_cos", "API_85", "API_92", "Prcp_sum7",
              "posT7", "negT30", "Q_diff1"]

# long-memory channels for h>=30 tracks (snowpack/seasonal memory proxies);
# kept separate so short-range tracks stay identical to the completed main run
LSTM_EXTRA_LONG = LSTM_EXTRA + ["Prcp_sum90", "negT90", "Q_mean90",
                                "doy_sin2", "doy_cos2"]

XGB_FE_FEATURES = [
    # raw same-day drivers
    "Dayl", "Prcp", "Srad", "Tmax", "Tmin", "Vp",
    # temperature structure
    "Tmean", "Trange",
    # seasonality
    "doy_sin", "doy_cos",
    # precipitation memory
    "Prcp_lag1", "Prcp_lag2", "Prcp_lag3", "Prcp_lag7",
    "Prcp_sum3", "Prcp_sum7", "Prcp_sum14", "Prcp_sum30", "Prcp_sum60",
    "API_85", "API_92",
    # snow/energy proxies
    "posT7", "negT30",
    # discharge memory
    "Q_lag1", "Q_lag2", "Q_lag3", "Q_lag7",
    "Q_mean7", "Q_mean30", "Q_diff1", "Q_ratio7",
]

# additions for h>=30 tracks (snowpack memory + annual harmonic)
XGB_FE_LONG_EXTRA = ["Prcp_sum90", "negT90", "Q_mean90", "doy_sin2", "doy_cos2"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all engineered columns (causal, day-t features)."""
    out = df.copy()
    doy = out["Date"].dt.dayofyear.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    out["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    out["Tmean"] = (out["Tmax"] + out["Tmin"]) / 2.0
    out["Trange"] = out["Tmax"] - out["Tmin"]

    p, q = out["Prcp"], out["Discharge"]
    for lag in (1, 2, 3, 7):
        out[f"Prcp_lag{lag}"] = p.shift(lag)
        out[f"Q_lag{lag}"] = q.shift(lag)
    for w in (3, 7, 14, 30, 60, 90):
        out[f"Prcp_sum{w}"] = p.rolling(w, min_periods=w).sum()
    # antecedent precipitation index, ewm form: API_t = k*API_{t-1} + (1-k)*P_t
    # (causal, standard API up to a constant scaling that normalization absorbs)
    for k in (0.85, 0.92):
        out[f"API_{int(k*100)}"] = (p.ewm(alpha=1 - k, adjust=False).mean())
    out["posT7"] = out["Tmean"].clip(lower=0).rolling(7, min_periods=7).sum()
    out["negT30"] = (-out["Tmean"].clip(upper=0)).rolling(30, min_periods=30).sum()
    out["negT90"] = (-out["Tmean"].clip(upper=0)).rolling(90, min_periods=90).sum()
    out["Q_mean7"] = q.rolling(7, min_periods=7).mean()
    out["Q_mean30"] = q.rolling(30, min_periods=30).mean()
    out["Q_mean90"] = q.rolling(90, min_periods=90).mean()
    out["Q_diff1"] = q.diff()
    out["Q_ratio7"] = q / (out["Q_mean30"] + 1e-6)
    return out
