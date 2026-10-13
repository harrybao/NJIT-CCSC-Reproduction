"""Set SYMHnet results next to the paper's per-storm tables (Abduallah et al. 2024,
arXiv:2402.17196, Tables 8, 9 and 12: SYMHnet column, 5-minute data, 1 h and 2 h ahead).

Usage: python paper_table_compare.py run_all.json out.json [label]
  run_all.json is written by run_all.py (keys like storm37_5min_1h with 'ours',
  'author_stored' and 'persistence_on_series' metric blocks).
"""
import json, sys

PAPER = {  # storm: (RMSE 1h, RMSE 2h, R2 1h, R2 2h) for SYMHnet
    26: (3.977, 4.330, 0.956, 0.948), 27: (7.682, 8.577, 0.952, 0.940), 28: (4.599, 4.977, 0.957, 0.949),
    29: (5.058, 5.515, 0.955, 0.946), 30: (2.213, 2.636, 0.991, 0.987), 31: (7.923, 9.737, 0.976, 0.963),
    32: (3.969, 4.451, 0.986, 0.982), 33: (11.366, 13.745, 0.979, 0.969), 34: (5.259, 5.611, 0.945, 0.938),
    35: (5.406, 5.830, 0.954, 0.947), 36: (5.618, 5.970, 0.949, 0.943), 37: (10.320, 10.923, 0.993, 0.992),
    38: (3.368, 3.765, 0.961, 0.951), 39: (5.670, 6.252, 0.964, 0.943), 40: (5.752, 6.336, 0.948, 0.937),
    41: (5.871, 6.857, 0.984, 0.978), 42: (3.900, 4.674, 0.985, 0.978)}

run, out = sys.argv[1:3]
d = json.load(open(run))
rows = {}
for key, v in sorted(d.items()):
    storm, res, h = key.split("_")
    s, hh = int(storm[5:]), int(h[:-1])
    if res != "5min" or s not in PAPER or hh not in (1, 2):
        continue
    p = PAPER[s]
    rows[key] = {"paper_RMSE": p[hh - 1], "paper_R2": p[hh + 1],
                 "author_stored_RMSE": round(v["author_stored"]["RMSE"], 3) if "author_stored" in v else None,
                 "author_stored_R2": round(v["author_stored"]["R2"], 3) if "author_stored" in v else None,
                 "ours_RMSE": round(v["ours"]["RMSE"], 3), "ours_R2": round(v["ours"]["R2"], 3),
                 "persistence_RMSE": round(v["persistence_on_series"]["RMSE"], 3), "persistence_R2": round(v["persistence_on_series"]["R2"], 3)}
json.dump(rows, open(out, "w"), indent=1)
print("| run | paper RMSE | stored RMSE | ours RMSE | persistence RMSE | paper R2 | ours R2 |")
for k, r in rows.items():
    print(f"| {k} | {r['paper_RMSE']} | {r['author_stored_RMSE']} | {r['ours_RMSE']} | {r['persistence_RMSE']} | {r['paper_R2']} | {r['ours_R2']} |")
