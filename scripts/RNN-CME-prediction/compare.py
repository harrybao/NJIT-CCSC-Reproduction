"""Compare our pretrained-model outputs with the authors' stored outputs for
RNN-CME-prediction (GRU/LSTM, 12-60 h), and compute metrics on the test ARs.
Usage (cwd = CMEpredict working copy): python compare.py out.json
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common_metrics import binary_metrics  # noqa: E402

res = {}
for t in ["gru", "lstm"]:
    for w in [12, 24, 36, 48, 60]:
        o = pd.read_csv(f"{t}-{w}-output.csv"); a = pd.read_csv(f"AUTHOR_outputs/{t}-{w}-output.csv")
        same_rows = len(o) == len(a) and (o["Timestamp"].values == a["Timestamp"].values).all()
        r = {"n": int(len(o)), "n_pos(P)": int((o["Label"] == "P").sum()), "test_ARs": sorted(o["NOAA AR NUM"].unique().tolist()),
             "same_rows_as_author": bool(same_rows)}
        if same_rows:
            r["pred_agreement_with_author"] = float((o["Predicted Label"] == a["Predicted Label"]).mean())
        r["ours"] = binary_metrics(o["Label"] == "P", o["Predicted Label"] == "P")
        r["author_stored"] = binary_metrics(a["Label"] == "P", a["Predicted Label"] == "P")
        res[f"{t}-{w}"] = r
        print(f"{t}-{w}", "n=%d pos=%d" % (r["n"], r["n_pos(P)"]), "agree=%.3f" % r.get("pred_agreement_with_author", float('nan')),
              "TSS ours=%.3f author=%.3f" % (r["ours"]["TSS"], r["author_stored"]["TSS"]), r["test_ARs"])
json.dump(res, open(sys.argv[1], "w"), indent=1)
