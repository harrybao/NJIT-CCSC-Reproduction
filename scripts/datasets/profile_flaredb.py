"""Profile the FlareDB event list linked from the CCDC page (github.com/Reasopprime/njit-flaredb).
Usage: python profile_flaredb.py /path/to/njit-flaredb out.json
"""
import json, os, sys
import pandas as pd

repo, out = sys.argv[1:3]
cols = ["goes_class", "noaa_ar", "start", "peak", "end", "harpnum", "col7", "col8"]
df = pd.read_csv(os.path.join(repo, "Flare_event_list.csv"), header=None, names=cols)
peak = pd.to_datetime(df["peak"], errors="coerce")
cls = df["goes_class"].str[0]
mag = df["goes_class"].str[1:].astype(float)
res = {
    "rows": int(len(df)), "has_header": False, "columns_inferred": cols,
    "peak_range": [str(peak.min()), str(peak.max())], "unparseable_times": int(peak.isna().sum()),
    "time_formats": sorted({("seconds+fraction" if "." in str(t) else "HH:MM") for t in df["peak"]}),
    "class_counts": cls.value_counts().to_dict(),
    "min_class": df.loc[(cls == "M") & (mag == mag[cls == "M"].min()), "goes_class"].iloc[0] if (cls == "M").any() else None,
    "flares_per_year": peak.dt.year.value_counts().sort_index().to_dict(),
    "unique_noaa_ars": int(df["noaa_ar"].nunique()), "unique_harps": int(df["harpnum"].nunique()),
    "missing_per_column": df.isna().sum().to_dict(),
    "duplicate_rows": int(df.duplicated().sum()),
    "ars_with_multiple_flares": int((df["noaa_ar"].value_counts() > 1).sum()),
    "max_flares_per_ar": int(df["noaa_ar"].value_counts().max()),
}
res["flares_per_year"] = {int(k): int(v) for k, v in res["flares_per_year"].items()}
res["missing_per_column"] = {k: int(v) for k, v in res["missing_per_column"].items()}
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(out, "w"), indent=1, default=str)
