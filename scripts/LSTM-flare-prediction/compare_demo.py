"""Compare our re-run of the pretrained LSTM demo with the authors' stored output.

Usage (cwd = LSTM-flare-prediction working copy):
  python compare_demo.py <out.json>
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common_metrics import binary_metrics  # noqa: E402

res = {}
for c in ["C", "M", "M5"]:
    d = f"results/LSTM_{c}_sample_run"
    ours = pd.read_csv(f"{d}/model.csv").rename(columns={"Label": "label"})
    auth = pd.read_csv(f"{d}/AUTHOR_output.csv").rename(columns={"Label": "label"})
    y = ours["label"].eq("Positive")
    res[c] = {
        "n_test_rows": int(len(ours)), "n_positive": int(y.sum()),
        "pred_agreement_with_author_output": float((ours["Predicted Label"] == auth["Predicted Label"]).mean()),
        "metrics_ours": binary_metrics(y, ours["Predicted Label"].eq("Positive")),
        "metrics_author_stored": binary_metrics(auth["label"].eq("Positive"), auth["Predicted Label"].eq("Positive")),
        "test_time_range": [ours["timestamp"].min(), ours["timestamp"].max()],
        "test_NOAA": sorted(ours["NOAA"].unique().tolist()),
    }
    tr = pd.read_csv(f"data/LSTM_{c}_sample_run/normalized_training.csv")[["NOAA", "timestamp"]]
    res[c]["train_time_range"] = [tr["timestamp"].min(), tr["timestamp"].max()]
    res[c]["test_NOAA_seen_in_training"] = sorted(set(res[c]["test_NOAA"]) & set(tr["NOAA"].unique().tolist()))
    print(c, {k: v for k, v in res[c].items() if k not in ("metrics_ours", "metrics_author_stored")})
    print("   ours  TSS=%.3f  author TSS=%.3f" % (res[c]["metrics_ours"]["TSS"], res[c]["metrics_author_stored"]["TSS"]))
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
