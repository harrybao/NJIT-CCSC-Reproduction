"""Evaluate SolarFlareNet pretrained-model outputs (ours vs. authors' stored).

For each (class, window): agreement between our re-run predictions and the
authors' stored result CSVs; standard metrics (argmax == prob>=0.5) on
(a) all evaluated test rows and (b) only test rows whose (HARPNUM, DATE__OBS)
does NOT appear in the training file (i.e. a leakage-free subset).
Also prints the feature-column mapping actually used by utils.load_data.
Usage (cwd = SolarFlareNet working copy): python evaluate.py out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common_metrics import binary_metrics  # noqa: E402

START, NF = 5, 14
C_LS = "TOTUSJH,TOTUSJZ,USFLUX,MEANALP,R_VALUE,TOTPOT,SAVNCPP,AREA_ACR,ABSNJZH".split(",")
C_ALL = "TOTUSJH,Cdec,TOTUSJZ,Chis1d,USFLUX,MEANALP,R_VALUE,TOTPOT,Chis,SAVNCPP,AREA_ACR,Edec,Xmax1d,ABSNJZH".split(",")


def kept_mask(df, cls):
    """Replicates utils.load_data row filtering (skip label 'p'; for C, skip zero records)."""
    v = df.values
    keep = []
    for row in v:
        if row[0][0] == "p":
            keep.append(False); continue
        z = False
        if cls == "C":
            idx = [5, 7, 9, 10, 11, 12, 14, 15, 18]
            z = any(float(row[k]) == 0.0 for k in idx)
        keep.append(not z)
    return np.array(keep)


res = {}
hdr = list(pd.read_csv("data/testing_data_C_24.csv", nrows=1).columns)
slice_cols = hdr[START:START + NF]
res["feature_mapping"] = {
    "intended_feature_names(c_ls)": C_LS,
    "csv_columns_actually_selected": [slice_cols[C_ALL.index(n)] for n in C_LS],
}
print(json.dumps(res["feature_mapping"]))
for w in ["24", "48", "72"]:
    for c in ["C", "M", "M5"]:
        te = pd.read_csv(f"data/testing_data_{c}_{w}.csv")
        tr = pd.read_csv(f"data/training_data_{c}_{w}.csv", usecols=["HARPNUM", "DATE__OBS"])
        km = kept_mask(te, c)
        te_k = te[km].reset_index(drop=True)
        ours = pd.read_csv(f"result/SolarFlareNet/{c}_{w}.csv")
        auth = pd.read_csv(f"AUTHOR_result/{c}_{w}.csv")
        assert len(ours) == len(te_k) == len(auth), (len(ours), len(te_k), len(auth))
        trkeys = set(zip(tr.HARPNUM, tr.DATE__OBS))
        dup = np.array([k in trkeys for k in zip(te_k.HARPNUM, te_k.DATE__OBS)])
        y = ours.FlareLabel.values.astype(bool)
        p_ours = ours.PredictionProbability.values
        r = {
            "n_eval": int(len(ours)), "n_pos": int(y.sum()),
            "label_agreement_with_author_file": float((ours.FlareLabel == auth.FlareLabel).mean()),
            "pred_agreement_with_author_file": float((ours.Prediction == auth.Prediction).mean()),
            "max_abs_prob_diff_vs_author": float(np.abs(p_ours - auth.PredictionProbability.values).max()),
            "all_rows": binary_metrics(y, p_ours >= 0.5),
            "author_file_all_rows": binary_metrics(auth.FlareLabel.values.astype(bool), auth.PredictionProbability.values >= 0.5),
            "n_rows_not_in_training": int((~dup).sum()), "n_pos_not_in_training": int(y[~dup].sum()),
            "rows_not_in_training": binary_metrics(y[~dup], p_ours[~dup] >= 0.5),
            "rows_duplicated_in_training": binary_metrics(y[dup], p_ours[dup] >= 0.5),
        }
        res[f"{c}_{w}"] = r
        print(f"{c}_{w}: n={r['n_eval']} pos={r['n_pos']} agree={r['pred_agreement_with_author_file']:.4f} "
              f"maxdp={r['max_abs_prob_diff_vs_author']:.2e} | TSS all={r['all_rows']['TSS']:.3f} "
              f"(author file {r['author_file_all_rows']['TSS']:.3f}) | dup-in-train={r['rows_duplicated_in_training']['TSS']:.3f} "
              f"| NOT-in-train (n={r['n_rows_not_in_training']}, pos={r['n_pos_not_in_training']}) TSS={r['rows_not_in_training']['TSS']:.3f}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
