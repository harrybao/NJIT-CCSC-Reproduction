"""Aggregate SYMHnet re-runs: our metrics vs the authors' stored files, plus two
persistence references computed on the same Actual series:
  persistence_1step : SYM-H(t) predicts the next row (what the shipped test
                      path actually targets: +1 min or +5 min)
  persistence_h     : SYM-H(t) predicts SYM-H(t + h hours) (the nominal task)
Skill score vs a reference: SS = 1 - MSE_model / MSE_ref.
Usage (cwd = SYMHnet working copy): python aggregate.py run_all.json out.json out.md
"""
import json
import sys

import numpy as np
import pandas as pd

run_json, out_json, out_md = sys.argv[1:4]
runs = json.load(open(run_json))
rows, res = [], {}
for key, r in sorted(runs.items()):
    storm, res_min, h = key.replace("storm", "").split("_")
    rm, hh = int(res_min.replace("min", "")), int(h.replace("h", ""))
    d = pd.read_csv(f"ours/{storm}/{rm}min/{hh}h.csv")
    y, p = d.Actual.values.astype(float), d.Prediction.values.astype(float)
    mse = np.mean((y - p) ** 2)
    m1 = np.mean((y[1:] - y[:-1]) ** 2)
    lag = int(hh * 60 / rm)
    mh = np.mean((y[lag:] - y[:-lag]) ** 2) if len(y) > lag else float("nan")
    a = r.get("author_stored", {})
    out = {"storm": int(storm), "res_min": rm, "h": hh, "n": int(len(y)),
           "RMSE_ours": float(np.sqrt(mse)), "R2_ours": r["ours"]["R2"],
           "RMSE_author": a.get("RMSE"), "R2_author": a.get("R2"),
           "RMSE_persist_1step": float(np.sqrt(m1)), "RMSE_persist_h": float(np.sqrt(mh)),
           "SS_vs_1step": float(1 - mse / m1), "SS_vs_h": float(1 - mse / mh),
           "max_abs_pred_diff_vs_author": r.get("max_abs_pred_diff")}
    res[key] = out
    rows.append(out)
df = pd.DataFrame(rows).sort_values(["storm", "res_min", "h"])
json.dump(res, open(out_json, "w"), indent=1)
cols = ["storm", "res_min", "h", "n", "RMSE_ours", "RMSE_author", "RMSE_persist_1step", "RMSE_persist_h", "SS_vs_1step", "SS_vs_h"]
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
for _, r in df.iterrows():
    md += "| " + " | ".join(str(int(r[c])) if c in ("storm", "res_min", "h", "n") else (f"{r[c]:.3f}" if r[c] == r[c] else "—") for c in cols) + " |\n"
open(out_md, "w").write(md)
print(md)
