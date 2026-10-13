"""Check output-column orientation of the shipped LSTM flare models.

For each class and each shipped .h5 file, compute softmax outputs on the demo
test set, then evaluate (a) the tool's rule  p[:,0] >= thr  and (b) the
alternative p[:,1] >= thr, where thr is the tool's fixed threshold
(C 0.5, M 0.6, M5 0.75). The LabelEncoder sorts labels alphabetically:
C -> ['C','N'] (col0 = positive), M -> ['M','N'] (col0 = positive),
M5 -> ['N','X'] (col0 = NEGATIVE).
Usage (cwd = LSTM-flare-prediction working copy): python orientation_check.py out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.getcwd())  # upstream tool modules live in the working copy
from common_metrics import binary_metrics  # noqa: E402

from keras.models import load_model  # noqa: E402
from LSTM_Flare import LSTM_Flare  # noqa: E402
from flarepredict_utils import get_n_features, series_len, start_feature, mask_value  # noqa: E402

THR = {"C": 0.5, "M": 0.6, "M5": 0.75}
POS = {"C": "C", "M": "M", "M5": "X"}
res = {}
lf = LSTM_Flare()
for c in ["C", "M", "M5"]:
    X, y = lf.load_data(f"data/LSTM_{c}_sample_run/normalized_testing.csv", c, series_len,
                        start_feature, get_n_features(c), mask_value)
    yt = (y == POS[c])
    auth = pd.read_csv(f"results/LSTM_{c}_sample_run/AUTHOR_output.csv")["Predicted Label"].eq("Positive").values
    res[c] = {"label_classes_sorted": sorted(set(y.tolist())), "n": int(len(y)), "n_pos": int(yt.sum())}
    for h5 in ["model.h5", "custom_model_id.h5"]:
        # compile=False: inference only; one .h5 stores an optimizer config (weight_decay) from a newer Keras
        p = load_model(f"pretrained_model/LSTM_{c}_sample_run/{h5}", compile=False).predict(X, verbose=0)
        for col in [0, 1]:
            pred = p[:, col] >= THR[c]
            m = binary_metrics(yt, pred)
            res[c][f"{h5}|col{col}>=thr"] = {"TSS": m["TSS"], "recall": m["recall"], "FP": m["FP"],
                                             "agree_with_author_output": float((pred == auth).mean())}
    print(c, json.dumps(res[c], indent=0))
json.dump(res, open(sys.argv[1], "w"), indent=1)
