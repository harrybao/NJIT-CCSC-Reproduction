"""Compare FibrilNet masks across our MC-dropout runs (seeds) and against the authors'
stored masks, so that ours-vs-author differences can be read against run-to-run noise.

Usage: python compare_runs.py WORKDIR out.json   (WORKDIR holds ours_seed*/ and AUTHOR_results/)
"""
import glob, itertools, json, os, sys
import numpy as np
from PIL import Image

work, out = sys.argv[1:3]
runs = sorted(glob.glob(os.path.join(work, "ours_seed*")))
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(work, "AUTHOR_results/predicted_mask/*.png")))
load = lambda d, n: np.asarray(Image.open(os.path.join(d, "predicted_mask", n)).convert("L")) > 127


def stats(a, b):
    inter, union = (a & b).sum(), (a | b).sum()
    return {"pixel_agreement": float((a == b).mean()), "IoU": float(inter / union),
            "dice": float(2 * inter / (a.sum() + b.sum()))}


res = {"runs": [os.path.basename(r) for r in runs], "per_image": {}, "summary": {}}
pairs_ours, pairs_auth = [], []
for n in names:
    A = load(os.path.join(work, "AUTHOR_results"), n)
    ours = [load(r, n) for r in runs]
    e = {"author_fg_fraction": float(A.mean()), "ours_fg_fraction": [float(o.mean()) for o in ours],
         "ours_vs_author": [stats(o, A) for o in ours],
         "ours_vs_ours": [stats(a, b) for a, b in itertools.combinations(ours, 2)]}
    res["per_image"][n] = e
    pairs_auth += e["ours_vs_author"]; pairs_ours += e["ours_vs_ours"]
for k, v in (("ours_vs_author", pairs_auth), ("ours_vs_ours", pairs_ours)):
    res["summary"][k] = {m: [round(float(np.mean([p[m] for p in v])), 4), round(float(np.min([p[m] for p in v])), 4),
                             round(float(np.max([p[m] for p in v])), 4)] for m in ("pixel_agreement", "IoU", "dice")}
    res["summary"][k]["n_pairs"] = len(v)
res["summary"]["format"] = "[mean, min, max] over images x pairs"
print(json.dumps(res["summary"], indent=1))
json.dump(res, open(out, "w"), indent=1)
