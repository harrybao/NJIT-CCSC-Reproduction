"""Row-level train/test overlap audit for the SolarFlareNet data files.

For every (flare class, time window) pair, count test rows whose
(HARPNUM, DATE__OBS) key also appears in the corresponding training file,
and compare time ranges. Usage: python overlap_audit.py <SolarFlareNet/data> <out.json>
"""
import json
import sys

import pandas as pd

root, out = sys.argv[1], sys.argv[2]
cols = ["Flare", "DATE__OBS", "NOAA_AR", "HARPNUM"]
res = {}
for w in ["24", "48", "72"]:
    for c in ["C", "M", "M5"]:
        tr = pd.read_csv(f"{root}/training_data_{c}_{w}.csv", usecols=cols)
        te = pd.read_csv(f"{root}/testing_data_{c}_{w}.csv", usecols=cols)
        ktr = set(zip(tr.HARPNUM, tr.DATE__OBS))
        kte = list(zip(te.HARPNUM, te.DATE__OBS))
        inter = sum(k in ktr for k in kte)
        te_eval = te[te.Flare.str[0] != "p"]  # rows with label 'p' are skipped by load_data
        inter_eval = sum((h, d) in ktr for h, d in zip(te_eval.HARPNUM, te_eval.DATE__OBS))
        res[f"{c}_{w}"] = {
            "train_rows": int(len(tr)), "test_rows": int(len(te)),
            "test_rows_used_by_loader(label!='p')": int(len(te_eval)),
            "test_label_counts": te.Flare.str[0].value_counts().to_dict(),
            "train_time": [tr.DATE__OBS.min(), tr.DATE__OBS.max()],
            "test_time": [te.DATE__OBS.min(), te.DATE__OBS.max()],
            "test_HARPs": int(te.HARPNUM.nunique()),
            "test_HARPs_in_train": int(len(set(te.HARPNUM) & set(tr.HARPNUM))),
            "test_rows_with_same_HARP_and_timestamp_in_train": int(inter),
            "fraction_used_test_rows_duplicated_in_train": round(inter_eval / max(len(te_eval), 1), 4),
            "train_rows_in_2022": int((tr.DATE__OBS >= "2022-01-01").sum()),
        }
        print(f"{c}_{w}", res[f"{c}_{w}"]["test_rows_with_same_HARP_and_timestamp_in_train"], "/", len(te),
              "dup-frac(used rows)=", res[f"{c}_{w}"]["fraction_used_test_rows_duplicated_in_train"],
              "train 2022 rows=", res[f"{c}_{w}"]["train_rows_in_2022"])
json.dump(res, open(out, "w"), indent=1)
