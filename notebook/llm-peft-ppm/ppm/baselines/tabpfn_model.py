from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from tabpfn import TabPFNClassifier, TabPFNRegressor

from ppm.datasets.event_logs import EventLog

def _mask_valid_next(df: pd.DataFrame, next_col: str, eos_id: int) -> pd.Series:
    return df[next_col] != eos_id

def _build_xy_na(train_log: EventLog, test_log: EventLog) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cur_act = "activity"
    next_act = "next_activity"
    eos_id = int(train_log.special_tokens["<EOS>"])

    num_cols: List[str] = list(train_log.features.numerical)
    for leak in ["remaining_time", "time_to_next_event"]:
        if leak in num_cols:
            num_cols.remove(leak)

    feat_cols = [cur_act] + num_cols

    tr_mask = _mask_valid_next(train_log.dataframe, next_act, eos_id)
    te_mask = _mask_valid_next(test_log.dataframe, next_act, eos_id)

    X_tr = train_log.dataframe.loc[tr_mask, feat_cols].to_numpy()
    y_tr = train_log.dataframe.loc[tr_mask, next_act].to_numpy().astype(int)

    X_te = test_log.dataframe.loc[te_mask, feat_cols].to_numpy()
    y_te = test_log.dataframe.loc[te_mask, next_act].to_numpy().astype(int)
    
    return X_tr, y_tr, X_te, y_te

def _build_xy_reg(train_log, test_log, target_col):
    cur_act = "activity"
    num_cols = list(train_log.features.numerical)
    
    if target_col in num_cols:
        num_cols.remove(target_col)
    other_future = "time_to_next_event" if target_col == "remaining_time" else "remaining_time"
    if other_future in num_cols:
        num_cols.remove(other_future)

    feat_cols = [cur_act] + num_cols
    
    X_tr = train_log.dataframe.loc[:, feat_cols].to_numpy()
    y_tr = train_log.dataframe.loc[:, target_col].to_numpy().astype(float)
    
    X_te = test_log.dataframe.loc[:, feat_cols].to_numpy()
    y_te = test_log.dataframe.loc[:, target_col].to_numpy().astype(float)
    
    return X_tr, y_tr, X_te, y_te


def run_tabpfn_baseline(train_log: EventLog, test_log: EventLog) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if "activity" in test_log.targets.categorical:
        X_tr, y_tr, X_te, y_te = _build_xy_na(train_log, test_log)
        clf = TabPFNClassifier(ignore_pretraining_limits=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        metrics["test_na_acc"] = float(acc)

    if "remaining_time" in [t.replace("next_", "") for t in test_log.targets.numerical] or "remaining_time" in test_log.dataframe.columns:
        X_tr, y_tr, X_te, y_te = _build_xy_reg(train_log, test_log, target_col="remaining_time")
        reg = TabPFNRegressor(ignore_pretraining_limits=True)
        reg.fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
        mse = mean_squared_error(y_te, y_hat)
        r2 = r2_score(y_te, y_hat)
        metrics["test_rt_mse"] = float(mse)
        metrics["test_rt_r2"] = float(r2)

    if "time_to_next_event" in test_log.dataframe.columns:
        X_tr, y_tr, X_te, y_te = _build_xy_reg(train_log, test_log, target_col="time_to_next_event")
        reg = TabPFNRegressor(ignore_pretraining_limits=True)
        reg.fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
        mse = mean_squared_error(y_te, y_hat)
        r2 = r2_score(y_te, y_hat)
        metrics["test_nt_mse"] = float(mse)
        metrics["test_nt_r2"] = float(r2)

    return metrics