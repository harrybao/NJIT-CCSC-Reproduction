"""Independent evaluation of FlareML pretrained models (ccsc-tools/FlareML).

Runs inside the upstream working copy (cwd = FlareML repo) with the sk024 env.
Re-uses the authors' pickled models and their exact inference preprocessing
(StandardScaler fitted on the test batch for MLP/ELM; raw features for RF),
but computes metrics directly from raw confusion matrices, WITHOUT the demo's
display-side adjustments (check_pm_precision / FP,FN==0 cell edits).
Also checks overlap between the demo test set and the training/original data.
"""
import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

OUT = sys.argv[1]
FEATS = ["TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX",
         "AREA_ACR", "TOTFZ", "MEANPOT", "R_VALUE", "EPSZ", "SHRGT45"]
CLASSES = ["B", "C", "M", "X"]
NUM = {"B": 1, "C": 2, "M": 3, "X": 4}
INV = {v: k for k, v in NUM.items()}

test = pd.read_csv("data/test_data/flaringar_simple_random_40.csv")
train = pd.read_csv("data/train_data/flaringar_training_sample.csv")
orig = pd.read_csv("data/original_data/flaringar_original_data.csv")
y = np.array([c[0] for c in test["Flare Class"]])
X = test[FEATS]

preds = {}
for alg in ["rf", "mlp", "elm"]:
    m = pickle.load(open(f"models/default_model_{alg}.sav", "rb"))
    Xi = X if alg == "rf" else StandardScaler().fit(X).transform(X)
    preds[alg.upper()] = np.array([INV[int(p)] for p in m.predict(Xi)])

# Majority vote exactly as compute_ens_result (ties -> RF)
ens = []
for r, m_, e in zip(preds["RF"], preds["MLP"], preds["ELM"]):
    if r == m_ or r == e:
        ens.append(r)
    elif e == m_:
        ens.append(e)
    else:
        ens.append(r)
preds["ENS"] = np.array(ens)


def ovr(y_true, y_pred, c):
    t, p = (y_true == c), (y_pred == c)
    TP = int((t & p).sum()); FN = int((t & ~p).sum())
    FP = int((~t & p).sum()); TN = int((~t & ~p).sum())
    rec = TP / (TP + FN) if TP + FN else float("nan")
    far = FP / (FP + TN) if FP + TN else float("nan")
    return dict(TP=TP, FN=FN, FP=FP, TN=TN, recall=rec, FPR=far, TSS=rec - far,
                BACC=(rec + (1 - far)) / 2)


res = {"n_test": int(len(y)), "class_counts": {c: int((y == c).sum()) for c in CLASSES}}
for alg, p in preds.items():
    res[alg] = {"accuracy": float((p == y).mean()),
                "confusion_matrix(rows=true B,C,M,X)": confusion_matrix(y, p, labels=CLASSES).tolist(),
                "per_class": {c: ovr(y, p, c) for c in CLASSES}}

# Leakage / overlap check on the 13 features (rounded to avoid float formatting noise)
def keyset(df, sig=3):
    # original_data.csv stores rounded E-notation (e.g. 8.60E+23), so compare
    # on 3 significant figures of a few well-populated features
    cols = ["TOTUSJH", "TOTPOT", "USFLUX", "R_VALUE", "SHRGT45"]
    v = df[cols].astype(float).values
    return set(tuple(float(f"{x:.{sig}g}") for x in row) for row in v)
kt, ktr, ko = keyset(test), keyset(train), keyset(orig)
res["overlap"] = {
    "test_rows_also_in_training_sample": len(kt & ktr),
    "test_rows_also_in_original_845": len(kt & ko),
    "training_rows_in_original_845": len(ktr & ko),
    "n_train_sample": int(len(train)), "n_original": int(len(orig)),
}
json.dump(res, open(OUT, "w"), indent=1, default=float)
for alg in ["RF", "MLP", "ELM", "ENS"]:
    print(alg, "acc=%.3f" % res[alg]["accuracy"],
          " ".join("%s:TSS=%.3f" % (c, res[alg]["per_class"][c]["TSS"]) for c in CLASSES))
print(json.dumps(res["overlap"]))
