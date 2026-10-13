"""SINet: naive baselines on exactly the same test sequences as F107_test.py / F30_test.py.

The released scripts build windows of 30 days and score only the LAST day of the
horizon (day d). We mirror that windowing (same date filter, fillna(0), sort) and
compute RMSE/MAE/MAPE for:
  persistence : value on the last input day
  recurrence27: value one (or more) 27-day solar rotations before the target day,
                taken from the most recent rotation available at forecast time
Usage (cwd = SINet repo): python baselines.py out.json
"""
import json
import math
import sys

import numpy as np
import pandas as pd

SEQ = 30
res = {}
for name, col in [("F10.7", "F10.7"), ("F30", "F30")]:
    raw = pd.read_csv(f"data/{name}_test.csv")
    n_missing = int(raw[col].isna().sum())
    raw = raw.fillna(0)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
    res[name] = {"rows": int(len(raw)), "missing_values_filled_with_0": n_missing}
    for d in [1, 27, 45, 60]:
        start = pd.Timestamp("2009-01-01") - pd.DateOffset(days=SEQ + d)
        df = raw[(raw["date"] > start) & (raw["date"].dt.year <= 2021)].reset_index(drop=True)
        v = df[col].values.astype(float)
        idx = np.arange(len(v) - SEQ - d)
        last_in = idx + SEQ - 1
        target = last_in + d
        y = v[target]
        k = math.ceil(d / 27)
        rec_idx = target - 27 * k
        out = {"n": int(len(y))}
        for tag, p in [("persistence", v[last_in]), ("recurrence27", v[rec_idx])]:
            ok = y != 0  # MAPE undefined at 0
            out[tag] = {"RMSE": float(np.sqrt(np.mean((y - p) ** 2))), "MAE": float(np.mean(np.abs(y - p))),
                        "MAPE%": float(np.mean(np.abs((y[ok] - p[ok]) / y[ok])) * 100)}
        out["zeros_in_targets"] = int((y == 0).sum())
        res[name][f"day{d}"] = out
        print(name, d, json.dumps(out))
json.dump(res, open(sys.argv[1], "w"), indent=1)
