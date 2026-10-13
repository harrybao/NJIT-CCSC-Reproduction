"""RNN-CME-prediction ships the same samples twice: CMEpredict/normalized_*.csv (no header;
used to train the released models; CMEpredict.py's output header names the 18 SHARP columns
TOTUSJH, TOTPOT, ... in the same order as the CME_data_samples header) and
CME_data_samples/normalized_*.csv (with that header). The feature columns are the same values
in a different order. This script (1) recovers the permutation, (2) checks which labelling is
physically plausible (extensive SHARP parameters should correlate strongly with USFLUX), and
(3) maps the features each released model uses (CMEpredict.get_df_values) to both namings.

Usage (cwd = copy of the release's CMEpredict/ folder): python column_label_check.py out.json
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
import CMEpredict as cp  # noqa: E402

full = pd.read_csv("../CME_data_samples/normalized_training_24.csv")
names = list(full.columns[4:])
demo = pd.read_csv("normalized_testing_24.csv", header=None)
a = demo[demo[0] != "padding"].iloc[:, 4:].values.astype(float)
b = pd.read_csv("../CME_data_samples/normalized_testing_24.csv")
b = b[b["NOAA_AR_NUM"].isin([12497, 12529])].iloc[:, 4:].values.astype(float)
perm = [int(np.flatnonzero([np.allclose(a[:, j], b[:, m], atol=1e-6) for m in range(b.shape[1])])[0]) for j in range(a.shape[1])]
X = full.iloc[:, 4:].values.astype(float)
u = names.index("USFLUX")
res = {"permutation_CMEpredict_col_to_CME_data_samples_col": perm,
       "corr_with_USFLUX": {n: {"CME_data_samples_labelling": round(float(np.corrcoef(X[:, j], X[:, u])[0, 1]), 3),
                                "CMEpredict_labelling": round(float(np.corrcoef(X[:, perm[j]], X[:, u])[0, 1]), 3)}
                            for j, n in enumerate(names)},
       "features_used_by_released_models": {}}
dummy = np.arange(22)[None, :].repeat(1, 0)
for typ in ("gru", "lstm"):
    for w in (12, 24, 36, 48, 60):
        nf, _ = cp.get_n_features_thresh(typ, w)
        cols = cp.get_df_values(typ, w, dummy)[0][4:4 + nf]
        res["features_used_by_released_models"][f"{typ}-{w}"] = {
            "names_as_in_code_and_paper_ranking": [names[c - 4] for c in cols],
            "values_actually_fed_if_CME_data_samples_header_is_right": [names[perm[c - 4]] for c in cols]}
print(json.dumps(res["corr_with_USFLUX"], indent=0)[:1500])
print(json.dumps(res["features_used_by_released_models"]["gru-24"], indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
