"""TSInet: compare our default-model reconstructions with the authors' stored
results and with naive baselines on the same 30-day windows.

Baselines (use only the 7 seed values the model itself receives):
  last_value : repeat the last seed TSI value
  seed_mean  : mean of the 7 seed values
Usage (cwd = TSInet working copy): python evaluate.py out.json
"""
import json
import sys

import numpy as np
import pandas as pd

res = {}
for ds in ["TCTE", "NRLTSI2", "SATIRE-S"]:
    ours = pd.read_csv(f"results/fixed_{ds}_result.csv")  # produced by run_dataset.py
    auth = pd.read_csv(f"AUTHOR_default_results/tsinet_result_{ds}.csv")
    full = pd.read_csv(f"test_data/{ds}_TSI.csv")
    y, p = ours.iloc[:, 1].values.astype(float), ours["TSInet"].values.astype(float)
    r = {"n_days": int(len(y)), "window": [str(ours.iloc[0, 0]), str(ours.iloc[-1, 0])]}
    rm = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    r["TSInet_ours"] = {"RMSE": rm(y, p), "MAE": float(np.mean(np.abs(y - p)))}
    ya, pa = auth.iloc[:, 1].values.astype(float), auth["TSInet"].values.astype(float)
    r["TSInet_author_stored"] = {"n": int(len(ya)), "RMSE": rm(ya, pa), "MAE": float(np.mean(np.abs(ya - pa))),
                                 "window": [str(auth.iloc[0, 0]), str(auth.iloc[-1, 0])]}
    seed = y[:7]
    r["baseline_last_value"] = {"RMSE": rm(y[7:], np.full(len(y) - 7, seed[-1]))}
    r["baseline_seed_mean"] = {"RMSE": rm(y[7:], np.full(len(y) - 7, seed.mean()))}
    r["TSInet_ours_after_seed"] = {"RMSE": rm(y[7:], p[7:])}
    col = [c for c in full.columns if "irr" in c.lower()] or [full.columns[-1]]
    v = full[col[0]].astype(float)
    v = v[v > 0]  # zeros mark missing days in the LASP files
    r["test_file_range_excluding_zeros"] = [float(v.min()), float(v.max())]
    r["window_range"] = [float(y.min()), float(y.max())]
    res[ds] = r
    print(ds, json.dumps(r))
json.dump(res, open(sys.argv[1], "w"), indent=1)
