"""Retrain/evaluate the LSTM flare model on the paper's Table 2 split.

Uses the upstream tool code unchanged (LSTM_Flare.load_data, LSTM_Flare.lstm,
epochs=7, batch_size=256, balanced class weights, Adam, categorical CE).
Train = Table2 normalized_training (2010-2013), threshold-selection set =
normalized_validation (2014), test = normalized_testing (2015-2018).

Three threshold policies are reported on the TEST set:
  fixed_tool : the tool's hard-coded thresholds (C 0.5, M 0.6, M5 0.75)
  val_opt    : threshold maximising TSS on the VALIDATION set (leak-free)
  test_opt   : threshold maximising TSS on the TEST set (the paper's Table 5
               convention; optimistic upper bound)
The probability of the POSITIVE class is taken from the correct softmax
column (LabelEncoder sorts labels: C->[C,N], M->[M,N], M5->[N,X]).

Usage (cwd = LSTM-flare-prediction working copy):
  python table2_experiment.py CLASS SEED OUT_DIR
"""
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.getcwd())
from common_metrics import best_threshold, binary_metrics  # noqa: E402

cls, seed, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
random.seed(seed); np.random.seed(seed)
import tensorflow as tf  # noqa: E402
tf.random.set_seed(seed)
from sklearn.utils import class_weight  # noqa: E402
from LSTM_Flare import LSTM_Flare  # noqa: E402
from flarepredict_utils import (batch_size, epochs, get_n_features, mask_value,  # noqa: E402
                                nclass, series_len, start_feature)

POS = {"C": "C", "M": "M", "M5": "X"}
FIXED = {"C": 0.5, "M": 0.6, "M5": 0.75}
root = "Table2_dataset/" + cls
lf = LSTM_Flare()
nf = get_n_features(cls)
t0 = time.time()
data = {p: lf.load_data(f"{root}/normalized_{p}.csv", cls, series_len, start_feature, nf, mask_value)
        for p in ["training", "validation", "testing"]}
t_load = time.time() - t0
Xtr, ytr = data["training"]
classes_sorted = sorted(set(ytr.tolist()))
pos_col = classes_sorted.index(POS[cls])
cw = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(ytr), y=ytr)
model = lf.lstm(nclass, nf, series_len)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
t0 = time.time()
model.fit(Xtr, lf.data_transform(ytr), epochs=epochs, batch_size=batch_size, verbose=0,
          shuffle=True, class_weight={0: cw[0], 1: cw[1]})
t_train = time.time() - t0

res = {"class": cls, "seed": seed, "n_features": nf, "series_len": series_len, "epochs": epochs,
       "label_order": classes_sorted, "positive_column": pos_col,
       "load_seconds": round(t_load, 1), "train_seconds": round(t_train, 1)}
probs = {}
for p in ["validation", "testing"]:
    X, y = data[p]
    probs[p] = (model.predict(X, batch_size=batch_size, verbose=0)[:, pos_col], (y == POS[cls]))
    res[f"n_{p}"] = int(len(y)); res[f"pos_{p}"] = int((y == POS[cls]).sum())
pv, yv = probs["validation"]; pt, yt = probs["testing"]
thr_val, _ = best_threshold(yv, pv)
thr_test, _ = best_threshold(yt, pt)
res["test"] = {
    "fixed_tool": {"threshold": FIXED[cls], **binary_metrics(yt, pt >= FIXED[cls])},
    "val_opt": {"threshold": thr_val, **binary_metrics(yt, pt >= thr_val)},
    "test_opt": {"threshold": thr_test, **binary_metrics(yt, pt >= thr_test)},
}
np.savez_compressed(f"{out_dir}/probs_{cls}_seed{seed}.npz", p_val=pv, y_val=yv, p_test=pt, y_test=yt)
json.dump(res, open(f"{out_dir}/metrics_{cls}_seed{seed}.json", "w"), indent=1)
print(cls, seed, "train_s=%.0f" % t_train,
      " | ".join("%s thr=%.3f TSS=%.3f HSS=%.3f" % (k, v["threshold"], v["TSS"], v["HSS"]) for k, v in res["test"].items()))
