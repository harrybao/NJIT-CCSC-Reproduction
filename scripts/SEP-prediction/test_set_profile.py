"""Profile the SEP test sets: how many independent ARs/events underlie the samples?

Usage: python test_set_profile.py <SEP_Package/data> <out.json>
"""
import json
import sys

import pandas as pd

root, out = sys.argv[1], sys.argv[2]
res = {}
for e in ["fc", "f"]:
    for w in [12, 24, 36, 48, 60, 72]:
        tr = pd.read_csv(f"{root}/events_{e}_training_{w}.csv")
        te = pd.read_csv(f"{root}/events_{e}_testing_{w}.csv")
        lab = te.columns[0]
        pos = te[te[lab].astype(str).str.upper().str.startswith(("P", "1", "Y", "T"))] if te[lab].dtype == object else te[te[lab] == 1]
        res[f"{e}_{w}"] = {
            "train_rows": int(len(tr)), "test_rows": int(len(te)),
            "label_values": te[lab].astype(str).value_counts().to_dict(),
            "test_time": [te.iloc[:, 1].min(), te.iloc[:, 1].max()],
            "train_time": [tr.iloc[:, 1].min(), tr.iloc[:, 1].max()],
            "test_unique_NOAA_AR": int(te.NOAA_AR.nunique()),
            "test_positive_unique_NOAA_AR": sorted(pos.NOAA_AR.unique().tolist()),
            "test_positive_unique_days": int(pd.to_datetime(pos.iloc[:, 1].str[:10]).nunique()) if len(pos) else 0,
            "AR_overlap_train_test": int(len(set(tr.NOAA_AR) & set(te.NOAA_AR))),
        }
        print(e, w, {k: res[f"{e}_{w}"][k] for k in ["test_rows", "label_values", "test_unique_NOAA_AR", "test_positive_unique_NOAA_AR", "AR_overlap_train_test"]})
json.dump(res, open(out, "w"), indent=1)
