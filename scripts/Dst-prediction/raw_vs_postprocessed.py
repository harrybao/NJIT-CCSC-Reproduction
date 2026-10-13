"""Dst Transformer (DSTT): compare raw network outputs with the released
test path, which post-processes predictions using the test labels.

For each horizon h = 1..6 we report, on the released Oct-Nov 2021 test window:
  A. shipped path   : DSTTModel.predict() with y_test set (applies DSTT_utils.t(),
                      i.e. replaces most predictions by 0.9 * true label) using the
                      weights the shipped test() loads (num_hours=6 for every h)
  B. raw, shipped w : raw self.model.predict() output, 6h weights (as shipped test())
  C. raw, correct w : raw output with the h-hour weights (default_models/<h>h)
Metrics: RMSE, MAE, R^2, Pearson r. Also checks A against the stored
default_results/dst_<h>h_results.csv.
Usage (cwd = DSTT_Package working copy): python raw_vs_postprocessed.py out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from scipy.stats import pearsonr  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

from DSTT_model import DSTTModel  # noqa: E402
from DSTT_utils import load_training_and_testing_data  # noqa: E402


def mets(y, p):
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p)), "r": float(pearsonr(y, p)[0]),
            "frac_pred_eq_0.9xlabel": float(np.mean(np.abs(p - 0.9 * y) < 1e-6))}


res = {}
for h in range(1, 7):
    Xtr, ytr, Xte, yte, Xva, yva, dates = load_training_and_testing_data(h, "hourly")
    shape = (Xtr.shape[1], Xtr.shape[2])
    out = {"n_test": int(len(yte)), "n_train": int(len(ytr))}
    for tag, wh, use_labels in [("A_shipped_path_postprocessed", 6, True),
                                ("B_raw_shipped_weights_6h", 6, False),
                                ("C_raw_correct_weights", h, False)]:
        m = DSTTModel()
        m.load_model(shape, kl_weight=1 / Xtr.shape[0], num_hours=wh, w_dir="default_models")
        if use_labels:
            m.set_data(Xtr, ytr, Xte, yte)          # exactly what DSTT_test.test() does
            p = m.predict(Xte, verbose=0)
        else:
            p = np.squeeze(m.model.predict(Xte, verbose=0, batch_size=len(Xte)))
        out[tag] = mets(yte, p)
        if use_labels:
            stored = pd.read_csv(f"AUTHOR_default_results/dst_{h}h_results.csv")
            out[tag]["max_abs_diff_vs_stored_predictions"] = float(np.max(np.abs(stored.Predictions.values - np.asarray(p, float))))
    res[f"{h}h"] = out
    print(h, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in out.items()}))
json.dump(res, open(sys.argv[1], "w"), indent=1)
