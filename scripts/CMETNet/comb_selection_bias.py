"""CMETNet COMB (numerical-feature ensemble): quantify test-set selection bias.

Replicates CMETNet_COMB_train.py for each hold-out year: 10 rounds, each round
fits RF/SVR/XGB/GPR (same hyper-parameters) on a random 200-event subsample of
the other years and predicts the hold-out year. The released script keeps, for
each algorithm, the round with the lowest MAE ON THE HOLD-OUT YEAR, and the
ensemble (median of the 4) is also chosen by hold-out MAE. We report:
  selected_on_test : the released procedure (optimistic)
  mean_over_rounds : expected MAE of the median ensemble without peeking
  val_selected     : select rounds by MAE on a validation year (the year before)
Usage (cwd = CMETNet_Package working copy): python comb_selection_bias.py out.json [years...]
"""
import datetime
import json
import sys
from statistics import median

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor

out = sys.argv[1]
years = [int(y) for y in sys.argv[2:]] or [2014]
org = pd.read_csv("data/ICME_list.csv")
org["disturbance"] = pd.to_datetime(org["disturbance"])
org["year"] = org["disturbance"].dt.year
res = {}
for year in years:
    rng_seed = 1000 + year
    test = org[org.year == year]; val = org[org.year == year - 1]
    pool = org[(org.year != year)]
    if len(test) < 3:
        continue
    feats = [c for c in org.columns if c not in ("disturbance", "transit_time", "year")]
    Xp, yp = pool[feats], pool["transit_time"]
    Xt, yt = test[feats], test["transit_time"].values
    rounds = []
    for i in range(10):
        preds = {}
        for name, mdl in [("RF", RandomForestRegressor(n_estimators=800, max_features="sqrt", random_state=rng_seed + i)),
                          ("SVR", SVR(kernel="rbf", cache_size=200)),
                          ("XGB", XGBRegressor(random_state=rng_seed + i)),
                          ("GPR", GaussianProcessRegressor(kernel=1.0 * RBF() + WhiteKernel(noise_level=0.2),
                                                           n_restarts_optimizer=30, random_state=rng_seed + i))]:
            Xtr, _, ytr, _ = train_test_split(Xp, yp, train_size=200, random_state=rng_seed + 37 * i + len(name))
            m = mdl.fit(Xtr, ytr)
            preds[name] = (m.predict(Xt), m.predict(val[feats]) if len(val) else None)
        rounds.append(preds)
    algs = ["RF", "SVR", "XGB", "GPR"]
    # released procedure: per-algorithm best round by TEST MAE, then median ensemble
    best = {a: min(range(10), key=lambda i: mean_absolute_error(yt, rounds[i][a][0])) for a in algs}
    ens_sel = np.median(np.stack([rounds[best[a]][a][0] for a in algs]), axis=0)
    # no peeking: median ensemble within each round, averaged MAE over rounds
    ens_round_mae = [mean_absolute_error(yt, np.median(np.stack([r[a][0] for a in algs]), axis=0)) for r in rounds]
    r = {"n_test_events": int(len(test)), "n_pool": int(len(pool)),
         "selected_on_test_MAE": float(mean_absolute_error(yt, ens_sel)),
         "selected_on_test_r": float(np.corrcoef(yt, ens_sel)[0, 1]) if len(yt) > 2 else None,
         "mean_over_rounds_MAE": float(np.mean(ens_round_mae)), "std_over_rounds_MAE": float(np.std(ens_round_mae))}
    if len(val) >= 3:
        yv = val["transit_time"].values
        bestv = {a: min(range(10), key=lambda i: mean_absolute_error(yv, rounds[i][a][1])) for a in algs}
        ens_v = np.median(np.stack([rounds[bestv[a]][a][0] for a in algs]), axis=0)
        r["val_selected_MAE"] = float(mean_absolute_error(yt, ens_v))
    res[str(year)] = r
    print(year, json.dumps(r))
    json.dump(res, open(out, "w"), indent=1)
