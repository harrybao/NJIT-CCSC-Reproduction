"""Score the stored Table-2 retraining probabilities at the thresholds the paper states for
its LSTM (Liu et al. 2019, Sect. 4.3: 50 % / 60 % / 50 % for >=M5.0 / >=M / >=C) and set the
result next to the paper's Table 3/4 LSTM row.  The tool's own fixed thresholds are
0.75 / 0.60 / 0.50, so only >=M5.0 changes.

Usage: python paper_threshold_policy.py RUNS_DIR out.json
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common_metrics import binary_metrics  # noqa: E402

runs, out = sys.argv[1:3]
PAPER_THR = {"M5": 0.50, "M": 0.60, "C": 0.50}
PAPER_TABLE4_LSTM = {  # Liu et al. 2019, Table 4 (= Table 3 'LSTM' row), all 40 features
    "M5": {"recall": 0.978, "precision": 0.038, "ACC": 0.899, "BACC": 0.938, "HSS": 0.074, "TSS": 0.877},
    "M": {"recall": 0.881, "precision": 0.222, "ACC": 0.909, "BACC": 0.895, "HSS": 0.347, "TSS": 0.790},
    "C": {"recall": 0.762, "precision": 0.544, "ACC": 0.829, "BACC": 0.803, "HSS": 0.539, "TSS": 0.607},
}
res = {}
for cls, thr in PAPER_THR.items():
    per = []
    for f in sorted(glob.glob(os.path.join(runs, f"probs_{cls}_seed*.npz"))):
        z = np.load(f)
        per.append(binary_metrics(z["y_test"], z["p_test"] >= thr))
    keys = ("recall", "precision", "ACC", "BACC", "HSS", "TSS")
    res[cls] = {"threshold": thr, "n_seeds": len(per),
                "mean": {k: round(float(np.mean([p[k] for p in per])), 3) for k in keys},
                "std": {k: round(float(np.std([p[k] for p in per])), 3) for k in keys},
                "paper_table4_lstm": PAPER_TABLE4_LSTM[cls]}
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
