"""FlareML extension (not a reproduction): how much does the validation design
change the measured skill on the 845-flare dataset?

Random-forest with the hyper-parameters of the authors' pickled default RF,
evaluated three ways (per-class one-vs-rest TSS, macro-averaged):
  random_10fold      : StratifiedKFold(shuffle) - samples from the same AR can
                       sit on both sides of the split
  AR_grouped_10fold  : GroupKFold by NOAA AR number - no AR on both sides
  chronological      : train on flares before 2014-01-01, test on 2014-2016
Usage (cwd = FlareML repo): python split_sensitivity.py out.json
"""
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")
F = ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
     "AREA_ACR", "TOTFZ", "MEANPOT", "R_VALUE", "EPSZ", "SHRGT45"]
d = pd.read_csv("data/original_data/flaringar_original_data.csv")
d["y"] = d["Flare Class"].str[0]
d["date"] = pd.to_datetime(d["Flare Date"])
X, y, g = d[F].astype(float).values, d["y"].values, d["AR"].values
rf0 = pickle.load(open("models/default_model_rf.sav", "rb"))
params = {k: v for k, v in rf0.get_params().items() if k in ("n_estimators", "max_depth", "max_features", "min_samples_leaf", "min_samples_split", "criterion", "bootstrap", "class_weight")}


def tss_per_class(yt, yp):
    out = {}
    for c in ["B", "C", "M", "X"]:
        t, p = yt == c, yp == c
        tp, fn, fp, tn = (t & p).sum(), (t & ~p).sum(), (~t & p).sum(), (~t & ~p).sum()
        out[c] = float(tp / (tp + fn) - fp / (fp + tn)) if (tp + fn) and (fp + tn) else float("nan")
    out["macro"] = float(np.nanmean([out[c] for c in "BCMX"]))
    out["acc"] = float((yt == yp).mean())
    return out


def run(splits, seed):
    yp = np.empty(len(y), dtype=object)
    mask = np.zeros(len(y), bool)
    for tr, te in splits:
        m = RandomForestClassifier(**params, random_state=seed, n_jobs=1).fit(X[tr], y[tr])
        yp[te] = m.predict(X[te]); mask[te] = True
    return tss_per_class(y[mask], yp[mask].astype(str))


res = {"rf_params": {k: (v if not isinstance(v, (np.integer,)) else int(v)) for k, v in params.items()},
       "n": int(len(y)), "n_AR": int(len(set(g))), "class_counts": d["y"].value_counts().to_dict()}
for seed in range(5):
    r = {}
    r["random_10fold"] = run(StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y), seed)
    perm = np.random.RandomState(seed).permutation(len(set(g)))
    gmap = dict(zip(sorted(set(g)), perm))
    r["AR_grouped_10fold"] = run(GroupKFold(10).split(X, y, [gmap[a] for a in g]), seed)
    tr = np.where(d["date"] < "2014-01-01")[0]; te = np.where(d["date"] >= "2014-01-01")[0]
    r["chronological_<2014_vs_2014+"] = run([(tr, te)], seed)
    res[f"seed{seed}"] = r
summary = {}
for k in ["random_10fold", "AR_grouped_10fold", "chronological_<2014_vs_2014+"]:
    vals = [res[f"seed{s}"][k]["macro"] for s in range(5)]
    summary[k] = {"macro_TSS_mean": float(np.mean(vals)), "macro_TSS_std": float(np.std(vals)),
                  "per_class_mean": {c: float(np.nanmean([res[f"seed{s}"][k][c] for s in range(5)])) for c in "BCMX"}}
res["summary"] = summary
print(json.dumps(summary, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
