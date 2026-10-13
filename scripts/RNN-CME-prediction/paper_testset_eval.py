"""Evaluate the released pretrained GRU/LSTM CME models on the full paper test sets
(CME_data_samples/normalized_testing_T.csv; row counts equal Table 1 of Liu et al. 2020)
and set the result next to the paper's Table 4.

The released CMEpredict.py functions are used unchanged (load_data, get_n_features_thresh,
model files). Two caveats are measured rather than assumed:
  * CME_data_samples has a header row and no 'padding' rows (the demo files have them), so
    a sample's 20-step history is built from the preceding *labelled* rows of the same HARP.
    `padding_check` re-scores the two demo ARs both ways to show how much this matters.
  * The 18 feature columns of CME_data_samples are in a different order from the files in
    CMEpredict/ that the models were trained on (same values, permuted; the header names
    follow the CME_data_samples order). The permutation is recovered exactly from the two
    demo ARs present in both files and applied before scoring (`column_permutation`).
  * The paper reports means over 10x10 cross-validation runs; this is one released model.

Usage (cwd = a copy of the cited release's CMEpredict/ folder, with ../CME_data_samples):
    python paper_testset_eval.py out.json
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import CMEpredict as cp  # noqa: E402
from common_metrics import best_threshold, binary_metrics  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

PAPER_TABLE4 = {  # Liu et al. 2020 ApJ 890:12, Table 4 (mean over CV runs)
    "gru": {"TSS": [0.452, 0.588, 0.602, 0.588, 0.600], "AUC": [0.779, 0.850, 0.874, 0.822, 0.870],
            "recall": [0.907, 0.913, 0.883, 0.815, 0.722], "precision": [0.598, 0.712, 0.754, 0.803, 0.884]},
    "lstm": {"TSS": [0.459, 0.562, 0.540, 0.529, 0.562], "AUC": [0.791, 0.851, 0.858, 0.801, 0.836],
             "recall": [0.945, 0.901, 0.895, 0.767, 0.644], "precision": [0.586, 0.699, 0.710, 0.785, 0.916]},
}
WINDOWS = [12, 24, 36, 48, 60]


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def column_permutation(w):
    """perm[j] = column of CME_data_samples that holds CMEpredict-file feature column j."""
    d = pd.read_csv(f"normalized_testing_{w}.csv", header=None)
    lab = d[d[0] != "padding"].iloc[:, 4:].values.astype(float)
    f = pd.read_csv(f"../CME_data_samples/normalized_testing_{w}.csv")
    sub = f[f["NOAA_AR_NUM"].isin([12497, 12529])].iloc[:, 4:].values.astype(float)
    assert lab.shape == sub.shape
    perm = [[m for m in range(sub.shape[1]) if np.allclose(lab[:, j], sub[:, m], atol=1e-6)] for j in range(lab.shape[1])]
    assert all(len(c) == 1 for c in perm) and sorted(c[0] for c in perm) == list(range(sub.shape[1])), perm
    return [c[0] for c in perm]


def reordered_csv(w, perm, ars=None):
    """CME_data_samples test file, header dropped, feature columns in CMEpredict order."""
    f = pd.read_csv(f"../CME_data_samples/normalized_testing_{w}.csv")
    if ars is not None:
        f = f[f["NOAA_AR_NUM"].isin(ars)]
    out = pd.concat([f.iloc[:, :4], f.iloc[:, [4 + m for m in perm]]], axis=1)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    out.to_csv(tmp.name, header=False, index=False)
    return tmp.name


def predict(model, path, typ, w, nf):
    X, y, _ = cp.load_data(datafile=path, series_len=20, start_feature=4, n_features=nf,
                           mask_value=0, type=typ, time_window=w)
    return model.predict(X, batch_size=256, verbose=0).ravel(), y


res = {"paper_table4": PAPER_TABLE4, "models": {}, "padding_check": {}, "column_permutation": {}}
for typ in ("gru", "lstm"):
    for i, w in enumerate(WINDOWS):
        nf, thr = cp.get_n_features_thresh(typ, w)
        model = load_model(f"{typ}-{w}-model.h5")
        perm = column_permutation(w)
        res["column_permutation"][str(w)] = perm
        full = reordered_csv(w, perm)
        p, y = predict(model, full, typ, w, nf)
        t_opt, _ = best_threshold(y, p)
        r = {"n": int(len(y)), "n_pos": int(y.sum()), "tool_threshold": thr, "auc": auc(y, p),
             "at_tool_threshold": binary_metrics(y, p >= thr),
             "at_test_optimal_threshold": {"threshold": t_opt, **binary_metrics(y, p >= t_opt)},
             "paper": {k: v[i] for k, v in PAPER_TABLE4[typ].items()}}
        res["models"][f"{typ}-{w}"] = r

        # padding sensitivity on the two demo ARs (12497, 12529)
        pd_demo, yd_demo = predict(model, f"normalized_testing_{w}.csv", typ, w, nf)
        sub = reordered_csv(w, perm, ars=[12497, 12529])
        ps, ys = predict(model, sub, typ, w, nf)
        same = len(ys) == len(yd_demo) and bool((ys == yd_demo).all())
        res["padding_check"][f"{typ}-{w}"] = {
            "n_demo_with_padding": int(len(yd_demo)), "n_same_ARs_without_padding": int(len(ys)),
            "same_samples": same,
            "label_agreement_at_tool_threshold": float(((pd_demo >= thr) == (ps >= thr)).mean()) if same else None,
            "max_abs_prob_diff": float(np.abs(pd_demo - ps).max()) if same else None}
        os.unlink(full); os.unlink(sub)
        print(f"{typ}-{w} n={r['n']} pos={r['n_pos']} AUC={r['auc']:.3f} "
              f"TSS@tool={r['at_tool_threshold']['TSS']:.3f} TSS@opt={r['at_test_optimal_threshold']['TSS']:.3f} "
              f"| paper TSS={r['paper']['TSS']} AUC={r['paper']['AUC']} | padding {res['padding_check'][f'{typ}-{w}']}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
