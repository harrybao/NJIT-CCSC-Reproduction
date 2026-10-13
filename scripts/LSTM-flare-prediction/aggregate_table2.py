"""Aggregate the 10-seed Table 2 LSTM runs into mean (std) per class/policy.

Usage: python aggregate_table2.py <runs_dir> <out.json> <out.md>
"""
import glob
import json
import sys

import numpy as np

runs_dir, out_json, out_md = sys.argv[1:4]
KEYS = ["recall", "precision", "ACC", "BACC", "HSS", "TSS"]
agg = {}
for cls in ["M5", "M", "C"]:
    files = sorted(glob.glob(f"{runs_dir}/metrics_{cls}_seed*.json"))
    runs = [json.load(open(f)) for f in files]
    agg[cls] = {"n_seeds": len(runs), "n_test": runs[0]["n_testing"], "pos_test": runs[0]["pos_testing"],
                "n_train_seq": None, "train_seconds_mean": float(np.mean([r["train_seconds"] for r in runs]))}
    for pol in ["fixed_tool", "val_opt", "test_opt"]:
        agg[cls][pol] = {k: (float(np.mean([r["test"][pol][k] for r in runs])),
                             float(np.std([r["test"][pol][k] for r in runs]))) for k in KEYS + ["threshold"]}
json.dump(agg, open(out_json, "w"), indent=1)
lines = ["| Class | Threshold policy | Recall | Precision | ACC | BACC | HSS | TSS | thr |", "|---|---|---|---|---|---|---|---|---|"]
for cls in ["M5", "M", "C"]:
    for pol in ["fixed_tool", "val_opt", "test_opt"]:
        a = agg[cls][pol]
        lines.append(f"| ≥{cls if cls != 'M5' else 'M5.0'} (n={agg[cls]['n_test']}, pos={agg[cls]['pos_test']}) | {pol} | " +
                     " | ".join(f"{a[k][0]:.3f} ({a[k][1]:.3f})" for k in KEYS) + f" | {a['threshold'][0]:.2f} |")
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
