"""Out-of-sample (chronological, AR-disjoint) evaluation of the pretrained MIST model.

MIST's processed dataset equals LSTM-flare-prediction Table2_dataset/M/
normalized_training.csv (2010-05..2014-01; 84,577 rows) restricted to 16
features; windows are 24 consecutive rows (p..p+23) labelled with row p+24,
standardised with the shipped StandardScaler (fit on all 84,577 rows), and
split 80/20 at random. Here we build windows identically from
Table2_dataset/M/normalized_testing.csv (2015-2018, no shared ARs) and apply
the shipped model unchanged.
Usage (cwd = MIST working copy):
  python out_of_sample_eval.py <Table2_dataset/M> scaler_params.json out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common_metrics import best_threshold, binary_metrics  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from model_architecture import MTST  # noqa: E402

t2dir, scaler_json, out = sys.argv[1:4]
feats = json.load(open("data/processed/selected_features.json"))
sc = json.load(open(scaler_json))
mu, sd = np.array(sc["mean_"]), np.array(sc["scale_"])
model = MTST(num_features=16, time_steps=24, patch_length=2, d_model=128, num_heads=8,
             num_encoder_layers=3, dim_feedforward=256, dropout=0.1)
model.load_state_dict(torch.load("models/mtst_flare_model.pth", map_location="cpu"))
model.eval()


def windows(df):
    A = ((df[feats].values - mu) / sd).astype(np.float32)
    lab = (df["label"].values == "Positive").astype(int)
    noaa = df["NOAA"].values
    n = len(A) - 24
    X = np.stack([A[p:p + 24] for p in range(n)])
    y = lab[24:24 + n]
    pure = np.array([noaa[p] == noaa[p + 24] for p in range(n)])  # window+label row inside one AR
    return X, y, pure


def predict(X):
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).squeeze(-1).numpy()


res = {}
probs = {}
# sanity: shipped test set
Xs, ys = np.load("data/processed/X_test.npy").astype(np.float32), np.load("data/processed/y_test.npy")
res["shipped_random_split_test"] = binary_metrics(ys, predict(Xs) >= 0.5)
for part in ["testing", "validation"]:
    df = pd.read_csv(f"{t2dir}/normalized_{part}.csv")
    X, y, pure = windows(df)
    p = predict(X)
    probs[part] = (y, p)
    res[f"table2_{part}_ROC_AUC"] = float(roc_auc_score(y, p))
    res[f"table2_{part}_all_windows"] = binary_metrics(y, p >= 0.5)
    res[f"table2_{part}_AR_pure_windows"] = binary_metrics(y[pure], p[pure] >= 0.5)
    res[f"table2_{part}_time_range"] = [df["timestamp"].min(), df["timestamp"].max()]
# fairness check: choose the threshold that maximises TSS on 2014 validation, apply to 2015-2018 test
thr, _ = best_threshold(probs["validation"][0], probs["validation"][1])
res["table2_testing_val_selected_threshold"] = thr
res["table2_testing_all_windows@val_thr"] = binary_metrics(probs["testing"][0], probs["testing"][1] >= thr)
res["shipped_random_split_test_ROC_AUC"] = float(roc_auc_score(ys, predict(Xs)))
for k, v in res.items():
    if not isinstance(v, dict):
        print(k, v)
    if isinstance(v, dict):
        print(f"{k:40s} N={v['N']:6d} pos={v['TP']+v['FN']:5d} TSS={v['TSS']:.3f} HSS={v['HSS']:.3f} recall={v['recall']:.3f} precision={v['precision']:.3f}")
json.dump(res, open(out, "w"), indent=1)
