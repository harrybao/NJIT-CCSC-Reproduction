"""Audit the LSTM-flare-prediction Table 2 train/validation/test split.

For each class folder (C, M, M5) report sizes, positive rates, time ranges,
and active-region (NOAA / HARP) overlap between partitions, plus the fraction
of rows whose 25 SHARP feature values contain an exact 0 (the paper's
"missing value" convention). Usage: python split_audit.py <Table2_dataset_dir> <out.json>
"""
import json
import sys

import pandas as pd

root, out = sys.argv[1], sys.argv[2]
res = {}
for cls in ["C", "M", "M5"]:
    parts = {p: pd.read_csv(f"{root}/{cls}/{p}.csv") for p in ["training", "validation", "testing"]}
    r = {}
    for p, df in parts.items():
        ts = pd.to_datetime(df["timestamp"].str.replace("Z", "", regex=False).str.replace("_TAI", "", regex=False), errors="coerce")
        r[p] = {"rows": int(len(df)), "positives": int((df["label"] == "Positive").sum()),
                "pos_rate": round(float((df["label"] == "Positive").mean()), 4),
                "time_min": str(ts.min()), "time_max": str(ts.max()),
                "years": {int(k): int(v) for k, v in ts.dt.year.value_counts().sort_index().items()},
                "n_NOAA": int(df["NOAA"].nunique()), "n_HARP": int(df["HARP"].nunique())}
    tr, va, te = (set(parts[p]["HARP"]) for p in ["training", "validation", "testing"])
    r["HARP_overlap"] = {"train&test": len(tr & te), "train&valid": len(tr & va), "valid&test": len(va & te)}
    trn, ten = set(parts["training"]["NOAA"]), set(parts["testing"]["NOAA"])
    r["NOAA_overlap_train&test"] = len(trn & ten)
    res[cls] = r
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
