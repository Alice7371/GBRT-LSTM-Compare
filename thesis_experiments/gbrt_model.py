"""GBRT track: LightGBM gradient-boosted regression trees on flattened windows.

Tree ensembles are invariant to monotonic feature scaling, so GBRT consumes raw
(un-normalized) features. Hyper-parameters were fixed after manual tuning
(thesis, ch. 5): num_leaves 35, learning_rate 0.05, feature_fraction 0.85,
bagging_fraction 0.85, bagging_freq 5.
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import GBRT_FEATURES, make_windows, metrics

PARAMS = dict(
    objective="regression",
    metric="mse",
    boosting_type="gbdt",
    num_leaves=35,
    learning_rate=0.05,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    verbose=-1,
)


def run_gbrt(df: pd.DataFrame, window: int, out_dir: str) -> dict:
    values = df[GBRT_FEATURES].values.astype(np.float32)
    split = int(len(values) * 0.8)

    X_train = make_windows(values[:split], window)
    X_test = make_windows(values[split - window :], window)
    y_train = df["Discharge"].values[window:split].astype(np.float32)
    y_test = df["Discharge"].values[split:].astype(np.float32)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    result = {"train": metrics(y_train, train_pred), "test": metrics(y_test, test_pred)}

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"true": y_test, "pred": test_pred}).to_csv(
        f"{out_dir}/predictions.csv", index=False
    )
    return result
