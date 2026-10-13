"""Dst Transformer: Monte-Carlo-averaged raw predictions, persistence baseline,
and a train/test contamination check of DSTT_utils.load_training_and_testing_data.

The DSTT network contains variational (Bayesian) layers, so every forward pass
samples new weights; we average N=50 passes (the paper-style "mean prediction").
Persistence: predict Dst(t+h) with the last observed Dst(t).
Usage (cwd = DSTT_Package working copy): python raw_mc_and_baselines.py out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from scipy.stats import pearsonr  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

import DSTT_utils as U  # noqa: E402
from DSTT_model import DSTTModel  # noqa: E402

N_MC = 50


def mets(y, p):
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))), "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p)), "r": float(pearsonr(y, p)[0])}


res = {}
# --- contamination check: does the training frame contain the Oct-Nov 2021 test rows?
h = 1
tr_file = f"data/solar-wind-data/{h}h/solar_wind_parameters_data_{h}_hourly_train.csv"
ts_file = f"data/solar-wind-data/{h}h/solar_wind_parameters_data_{h}_hourly_test.csv"
all_data = pd.read_csv(tr_file)
all_data = all_data.loc[all_data["YEAR"] != 2021]
test_all = pd.read_csv(ts_file)
flt = ["2021-10-" + str(i) + "-" for i in range(1, 32)] + ["2021-11-" + str(i) + "-" for i in range(1, 31)]
test_rows = test_all.loc[test_all["Timestamp"].str.contains("|".join(flt))]
excluded_by_isin = test_all["Timestamp"].isin(flt).sum()
data_2021 = test_all.loc[~test_all["Timestamp"].isin(flt)]
train_frame = pd.concat([all_data, data_2021])
res["contamination_check"] = {
    "test_window_rows": int(len(test_rows)),
    "rows_removed_by_isin_filter": int(excluded_by_isin),
    "test_window_timestamps_present_in_training_frame": int(test_rows["Timestamp"].isin(set(train_frame["Timestamp"])).sum()),
    "training_frame_rows": int(len(train_frame)),
    "note": "train_data = all_data[:] in load_training_and_testing_data, i.e. the full frame is used for training",
}
print(json.dumps(res["contamination_check"]))

# --- persistence baseline series (Dst at time t) from the 1h file: Dst_1h[t] = Dst(t+1)
full = pd.concat([pd.read_csv(tr_file), test_all]).drop_duplicates("Timestamp")
for h in range(1, 7):
    Xtr, ytr, Xte, yte, Xva, yva, dates = U.load_training_and_testing_data(h, "hourly")
    yte = np.asarray(yte, float).ravel()
    out = {"n_test": int(len(yte))}
    for tag, wh in [("raw_mc50_shipped_6h_weights", 6), ("raw_mc50_correct_weights", h)]:
        m = DSTTModel()
        m.load_model((Xtr.shape[1], Xtr.shape[2]), kl_weight=1 / Xtr.shape[0], num_hours=wh, w_dir="default_models")
        P = np.stack([np.squeeze(m.model.predict(Xte, verbose=0, batch_size=len(Xte))) for _ in range(N_MC)])
        out[tag] = mets(yte, P.mean(0))
        out[tag]["mc_std_mean"] = float(P.std(0).mean())
    # persistence, aligned by timestamp: the h-file is pre-shifted (label at t = Dst(t+h)),
    # so Dst(t) = label of the row at timestamp t-h. Rows with no t-h partner are skipped.
    tf = pd.read_csv(f"data/solar-wind-data/{h}h/solar_wind_parameters_data_{h}_hourly_test.csv")
    ts = pd.to_datetime(tf["Timestamp"], format="%Y-%m-%d-%H")
    lab = pd.Series(tf["Dst-index"].values.astype(float), index=ts)
    lab = lab[~lab.index.duplicated()]
    win = lab[(lab.index >= "2021-10-01") & (lab.index < "2021-12-01")]
    prev = lab.reindex(win.index - pd.Timedelta(hours=h)).values
    ok = ~np.isnan(prev)
    out["persistence_Dst(t)"] = mets(win.values[ok], prev[ok])
    out["persistence_Dst(t)"]["n"] = int(ok.sum())
    out["persistence_note"] = "same Oct-Nov 2021 window, timestamp-aligned; label file pre-shifted by h hours"
    res[f"{h}h"] = out
    print(h, {k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out.items() if k != "persistence_note"})
json.dump(res, open(sys.argv[1], "w"), indent=1)
