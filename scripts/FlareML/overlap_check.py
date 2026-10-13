"""Tolerance-based overlap check between FlareML demo test set and training sample.

The demo test CSV stores rounded values (~3 significant figures) while the
training sample stores full precision, so we call two rows "the same sample"
if ALL 13 SHARP features agree within a relative tolerance (default 0.6%)
and the flare class letter agrees. Run with cwd = FlareML repo.
"""
import json
import sys

import numpy as np
import pandas as pd

F = ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
     "AREA_ACR", "TOTFZ", "MEANPOT", "R_VALUE", "EPSZ", "SHRGT45"]
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.006
t = pd.read_csv("data/test_data/flaringar_simple_random_40.csv")
tr = pd.read_csv("data/train_data/flaringar_training_sample.csv")
o = pd.read_csv("data/original_data/flaringar_original_data.csv")


def matches(a, b):
    A, B = a[F].astype(float).values, b[F].astype(float).values
    out = []
    for i, row in enumerate(A):
        den = np.maximum(np.abs(row), 1e-3)
        ok = np.all(np.abs(B - row) / den <= tol, axis=1)
        out.append(np.where(ok)[0].tolist())
    return out


mt = matches(t, tr)
mo = matches(t, o)
n_tr = sum(1 for m in mt if m)
same_label = sum(1 for i, m in enumerate(mt) if m and any(tr["Flare Class"].iloc[j][0] == t["Flare Class"].iloc[i][0] for j in m))
# also match test -> original by (date, AR) key, which is exact
o["key"] = pd.to_datetime(o["Flare Date"]).dt.date.astype(str) + "_" + o["AR"].astype(str)
t["key"] = pd.to_datetime(t["fdate"]).dt.date.astype(str) + "_" + t["noaaar"].astype(str)
res = {
    "relative_tolerance": tol,
    "test_rows": len(t),
    "test_rows_matching_a_training_row_all13_features": n_tr,
    "of_which_same_class_letter": same_label,
    "test_rows_matching_original_db_all13_features": sum(1 for m in mo if m),
    "test_rows_matching_original_db_by_date_and_AR": int(t["key"].isin(o["key"]).sum()),
    "test_date_range": [str(pd.to_datetime(t.fdate).min().date()), str(pd.to_datetime(t.fdate).max().date())],
}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
