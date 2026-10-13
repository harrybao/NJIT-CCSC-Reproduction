"""MIST: locate each test window inside processed_solar_dataset.csv and measure
how many test windows have an overlapping neighbour window that is NOT in the
test set (i.e. is, by elimination, in the training set).

If windows are 24 consecutive rows and the split is a random shuffle of
windows, window i in test and window i+1 in train share 23 of 24 rows.
Usage (cwd = MIST working copy): python window_overlap_audit.py out.json scaler_params.json
"""
import json
import sys

import numpy as np
import pandas as pd

T = 24
df = pd.read_csv("data/processed/processed_solar_dataset.csv")
feats = json.load(open("data/processed/selected_features.json"))
A = df[feats].values.astype(np.float64)
# X_test is the CSV standardised by the shipped StandardScaler (fit on all 84,577 rows;
# its mean_/scale_ were read with joblib in a numpy-2 env and are pasted here).
MU = np.array(json.load(open(sys.argv[2]))["mean_"]) if len(sys.argv) > 2 else A.mean(0)
SD = np.array(json.load(open(sys.argv[2]))["scale_"]) if len(sys.argv) > 2 else A.std(0)
A = ((A - MU) / SD).astype(np.float32)
Xte = np.load("data/processed/X_test.npy").astype(np.float32)
yte = np.load("data/processed/y_test.npy")
ytr = np.load("data/processed/y_train.npy")
n_rows, n_win_total = len(A), len(Xte) + len(ytr)

# hash every possible 24-row window start by its first row + last row
key = lambda w: (tuple(np.round(w[0], 3)), tuple(np.round(w[-1], 3)))
index = {}
for i in range(n_rows - T + 1):
    index.setdefault(key(A[i:i + T]), []).append(i)
pos = []
for w in Xte:
    cands = index.get(key(w), [])
    pos.append(next((i for i in cands if np.allclose(A[i:i + T], w, atol=1e-3)), -1))
pos = np.array(pos)
found = pos >= 0
test_set = set(pos[found].tolist())
nb_train = sum(1 for p in pos[found] if (p - 1 not in test_set and p - 1 >= 0) or (p + 1 not in test_set and p + 1 <= n_rows - T))
res = {
    "csv_rows": int(n_rows), "possible_24row_windows": int(n_rows - T + 1),
    "n_train_windows": int(len(ytr)), "n_test_windows": int(len(Xte)), "total": int(n_win_total),
    "test_windows_located_in_csv": int(found.sum()),
    "test_windows_with_an_adjacent(23-row-overlap)_window_outside_test": int(nb_train),
    "fraction": float(nb_train / max(found.sum(), 1)),
    "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean()),
    "test_positions_min_max": [int(pos[found].min()), int(pos[found].max())] if found.any() else None,
}
lab = (df["label"].values == "Positive").astype(int)
for off in (22, 23, 24):
    ok = found & (pos + off < n_rows)
    res[f"label_rule_agreement(y_test == label[row p+{off}])"] = float((lab[pos[ok] + off] == yte[ok]).mean())
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
