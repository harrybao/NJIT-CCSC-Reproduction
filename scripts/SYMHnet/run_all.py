"""Re-run SYMHnet shipped models on every storm that has stored author results,
one horizon at a time (the shipped test() names the output file after start_hour,
so a multi-horizon call overwrites the 1h file with 2h results).

Saves our CSVs under ours/<storm>/<res>min/<h>h.csv and metrics vs. the authors'
stored results_datasets files. Usage (cwd = SYMHnet working copy):
  python run_all.py out.json
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from SYMHnet_test import test  # noqa: E402

STORMS = [28, 31, 33, 36, 37, 40, 42]


def mets(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss_res, ss_tot = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    return {"RMSE": rmse, "R2": float(1 - ss_res / ss_tot), "r": float(np.corrcoef(y, p)[0, 1]), "n": int(len(y))}


res = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
for storm in STORMS:
    for rm in [5, 1]:
        for h in [1, 2]:
            key = f"storm{storm}_{rm}min_{h}h"
            if key in res:
                continue
            test(storm, h, h + 1, resolution_minutes=rm)
            src = f"results_datasets/storm_num_{storm}/{rm}min/{h}h_uq_training.csv"
            dst = f"ours/{storm}/{rm}min/{h}h.csv"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            ours = pd.read_csv(dst)
            r = {"ours": mets(ours.Actual, ours.Prediction)}
            auth_f = f"AUTHOR_results_datasets/storm_num_{storm}/{rm}min/{h}h_uq_training.csv"
            if os.path.exists(auth_f):
                a = pd.read_csv(auth_f)
                r["author_stored"] = mets(a.Actual, a.Prediction)
                r["same_actuals"] = bool(len(a) == len(ours) and np.allclose(a.Actual.values, ours.Actual.values))
                if r["same_actuals"]:
                    r["max_abs_pred_diff"] = float(np.max(np.abs(a.Prediction.values - ours.Prediction.values)))
            # persistence on the same series: SYM-H(t) for target SYM-H(t+h)
            lag = int(h * 60 / rm)
            y = ours.Actual.values.astype(float)
            if len(y) > lag:
                r["persistence_on_series"] = mets(y[lag:], y[:-lag])
            res[key] = r
            print(key, json.dumps(r))
            json.dump(res, open(sys.argv[1], "w"), indent=1)
