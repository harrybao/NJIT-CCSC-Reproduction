"""Variant of orientation_check.py for the cited release v1.0.2.

The v1.0.2 .h5 files embed a marshalled Python lambda (attention block) written by
an older Python (3.6), which Python 3.9 cannot unmarshal ("bad marshal data").
The architecture is unchanged between v1.0.2 and HEAD apart from the Keras
`Model(...)` call syntax, so we rebuild it with LSTM_Flare.lstm() from HEAD
(passed as argv[2]) and load only the weights from the cited .h5 files.
Usage (cwd = v1.0.2 working copy): python orientation_check_weights.py out.json HEAD_DIR
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common_metrics import binary_metrics  # noqa: E402

head = sys.argv[2]
sys.path.insert(0, head)                     # LSTM_Flare / utils from HEAD (same layers)
from LSTM_Flare import LSTM_Flare  # noqa: E402
from flarepredict_utils import get_n_features, mask_value, nclass, series_len, start_feature  # noqa: E402

THR = {"C": 0.5, "M": 0.6, "M5": 0.75}
POS = {"C": "C", "M": "M", "M5": "X"}
res, lf = {}, LSTM_Flare()
for c in ["C", "M", "M5"]:
    X, y = lf.load_data(f"data/LSTM_{c}_sample_run/normalized_testing.csv", c, series_len,
                        start_feature, get_n_features(c), mask_value)
    yt = (y == POS[c])
    auth = pd.read_csv(f"results/LSTM_{c}_sample_run/AUTHOR_output.csv")["Predicted Label"].eq("Positive").values
    res[c] = {"label_classes_sorted": sorted(set(y.tolist())), "n": int(len(y)), "n_pos": int(yt.sum())}
    for h5 in ["model.h5", "custom_model_id.h5"]:
        m = lf.lstm(nclass, get_n_features(c), series_len)
        m.load_weights(f"pretrained_model/LSTM_{c}_sample_run/{h5}")
        p = m.predict(X, verbose=0)
        for col in [0, 1]:
            pred = p[:, col] >= THR[c]
            mm = binary_metrics(yt, pred)
            res[c][f"{h5}|col{col}>=thr"] = {"TSS": mm["TSS"], "recall": mm["recall"], "FP": mm["FP"],
                                             "agree_with_author_output": float((pred == auth).mean())}
json.dump(res, open(sys.argv[1], "w"), indent=1)
for c, v in res.items():
    print(c, v["label_classes_sorted"], "n=", v["n"], "pos=", v["n_pos"])
    for k, m in v.items():
        if "|" in k:
            print("   %-32s TSS=%6.3f recall=%.3f FP=%3d agree_author=%.3f" % (k, m["TSS"], m["recall"], m["FP"], m["agree_with_author_output"]))
