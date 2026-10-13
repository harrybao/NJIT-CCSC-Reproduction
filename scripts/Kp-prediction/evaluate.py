"""KpNet: re-run shipped default models on the released 2022 test file.

Reports per horizon (1-9 h): metrics of (a) one stochastic pass of
KpNetModel.predict (what KpNet_test.py saves), (b) 20-pass MC mean,
(c) the authors' stored default_results predictions, (d) persistence
Kp(t) -> Kp(t+h) (timestamp-aligned on the pre-shifted label file), and the
number of test timestamps that also appear in the training file.
Usage (cwd = Kp-prediction working copy): python evaluate.py out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from scipy.stats import pearsonr  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

import KpNet_utils as U  # noqa: E402
from KpNet_model import KpNetModel  # noqa: E402

SCALE = 10


def mets(y, p):
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p)), "r": float(pearsonr(y, p)[0])}


res = {}
for h in range(1, 10):
    Xtr, ytr, Xte, yte, Xva, yva, dates = U.load_training_and_testing_data(h)
    y = np.asarray(yte, float).ravel() / SCALE
    m = KpNetModel()
    m.load_model((Xtr.shape[1], Xtr.shape[2]), kl_weight=1 / Xtr.shape[0], num_hours=h, w_dir="default_models")
    p1 = np.asarray(m.predict(Xte, verbose=0), float).ravel() / SCALE
    P = np.stack([np.asarray(m.model.predict(Xte, verbose=0, batch_size=len(Xte)), float).ravel() / SCALE for _ in range(20)])
    stored = pd.read_csv(f"AUTHOR_default_results/kp_{h}h_results.csv")
    out = {"n_test": int(len(y)), "single_pass": mets(y, p1), "mc20_mean": mets(y, P.mean(0)),
           "author_stored_file": mets(stored.Labels, stored.Predictions),
           "stored_labels_equal_ours": bool(np.allclose(stored.Labels.values, y))}
    d = f"data/solar-wind-data/{h}h"
    tr = pd.read_csv(f"{d}/solar_wind_parameters_data_{h}_hourly_train.csv")
    te = pd.read_csv(f"{d}/solar_wind_parameters_data_{h}_hourly_test.csv")
    out["test_timestamps_in_train_file"] = int(te.Timestamp.isin(set(tr.Timestamp)).sum())
    out["train_time"] = [str(tr.Timestamp.iloc[0]), str(tr.Timestamp.iloc[-1])]
    out["test_time"] = [str(te.Timestamp.iloc[0]), str(te.Timestamp.iloc[-1])]
    ts = pd.to_datetime(te.Timestamp, format="%Y-%m-%d-%H", errors="coerce")
    lab = pd.Series(te["Kp_German"].values.astype(float), index=ts).dropna()
    lab = lab[~lab.index.duplicated()]
    prev = lab.reindex(lab.index - pd.Timedelta(hours=h)).values
    ok = ~np.isnan(prev)
    out["persistence"] = mets(lab.values[ok], prev[ok]); out["persistence"]["n"] = int(ok.sum())
    res[f"{h}h"] = out
    print(h, {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out.items()})
json.dump(res, open(sys.argv[1], "w"), indent=1)
