"""Data-quality profile of ML-ready datasets bundled in the CCSC repositories.

For each table: rows, columns, time coverage, entity counts (AR/HARP), label
balance, exact-zero rate in physical features (the group's missing-value
convention), sentinel fill values (9999.99 etc.), NaNs and duplicate rows.
Usage: python profile_bundled_datasets.py <ccsc-tools root> out.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
FILL = [9999.99, 99999.9, 999.99, 9999999.0, 99.99, 9999.0, 999.9]
T = {
    "LSTM_Table2_M_train": ("LSTM-flare-prediction/Table2_dataset/M/training.csv", "timestamp", "label", "HARP"),
    "LSTM_Table2_M_test": ("LSTM-flare-prediction/Table2_dataset/M/testing.csv", "timestamp", "label", "HARP"),
    "LSTM_Table5_C_train_normalized": ("LSTM-flare-prediction/Table5_dataset_and_source_code/C/normalized_training.csv", "timestamp", "label", "HARP"),
    "SolarFlareNet_train_M_24": ("SolarFlareNet/data/training_data_M_24.csv", "DATE__OBS", "Flare", "HARPNUM"),
    "SolarFlareNet_test_M_24": ("SolarFlareNet/data/testing_data_M_24.csv", "DATE__OBS", "Flare", "HARPNUM"),
    "SEP_fc_train_24": ("SEP-prediction/SEP_Package/data/events_fc_training_24.csv", "Data sample timestamp", "SEP", "HARPNUM"),
    "SEP_fc_test_24": ("SEP-prediction/SEP_Package/data/events_fc_testing_24.csv", "Data sample timestamp", "SEP", "HARPNUM"),
    "RNNCME_train_24": ("RNN-CME-prediction/CMEpredict/normalized_training_24.csv", "Timestamp", "Label", "HARP NUM"),
    "FlareML_original_845": ("FlareML/data/original_data/flaringar_original_data.csv", "Flare Date", "Flare Class", "AR"),
    "CMETNet_ICME_list": ("CMETNet/CMETNet_Package/data/ICME_list.csv", "disturbance", None, None),
    "Kp_1h_train": ("Kp-prediction/data/solar-wind-data/1h/solar_wind_parameters_data_1_hourly_train.csv", "Timestamp", "Kp_German", None),
    "Dst_1h_train": ("Dst-prediction/DSTT_Package/data/solar-wind-data/1h/solar_wind_parameters_data_1_hourly_train.csv", "Timestamp", "Dst-index", None),
    "SYMH_5min_train_storm28": ("SYMHnet/data/solar_wind_symh_5min/1h/28/training_data_1h.csv", "Timestamp", "SYM_H", None),
    "TSI_SORCE_train": ("TSInet-irradiance-reconstruction/train_data/SORCE_TSI.csv", None, None, None),
}
res = {}
for name, (path, tcol, ycol, ent) in T.items():
    p = os.path.join(root, path)
    if not os.path.exists(p):
        res[name] = {"missing_file": path}; continue
    d = pd.read_csv(p, low_memory=False)
    num = d.select_dtypes(include=[np.number])
    phys = num.drop(columns=[c for c in num.columns if c.upper() in ("NOAA", "HARP", "HARPNUM", "NOAA_AR", "AR", "YEAR", "DOY", "HR", "MN", "HARP NUM", "NOAA AR NUM")], errors="ignore")
    r = {"path": path, "rows": int(len(d)), "cols": int(d.shape[1]),
         "nan_cells_frac": float(d.isna().values.mean()),
         "duplicate_rows": int(d.duplicated().sum()),
         "exact_zero_frac_physical": float((phys.values == 0).mean()) if phys.size else None,
         "rows_with_any_zero_physical_frac": float((phys.values == 0).any(axis=1).mean()) if phys.size else None,
         "fill_value_cells_frac": float(np.isin(np.round(phys.values.astype(float), 2), FILL).mean()) if phys.size else None}
    if tcol and tcol in d:
        r["time_min"], r["time_max"] = str(d[tcol].min()), str(d[tcol].max())
    if ent and ent in d:
        r["n_entities"] = int(d[ent].nunique())
    if ycol and ycol in d and d[ycol].dtype == object:
        r["label_counts_first_char"] = d[ycol].astype(str).str[0].value_counts().to_dict()
    res[name] = r
    print(name, {k: v for k, v in r.items() if k not in ("path",)})
json.dump(res, open(out, "w"), indent=1, default=str)
