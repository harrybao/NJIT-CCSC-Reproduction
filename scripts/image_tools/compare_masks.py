"""Compare image outputs (masks / maps) between our re-run and the authors'
stored results: pixel agreement, IoU of the foreground, and max abs difference.

Usage: python compare_masks.py OUR_DIR AUTHOR_DIR out.json [glob]
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ours_dir, auth_dir, out = sys.argv[1:4]
pat = sys.argv[4] if len(sys.argv) > 4 else "*.png"
res = {}
for f in sorted(glob.glob(os.path.join(ours_dir, pat))):
    g = os.path.join(auth_dir, os.path.basename(f))
    if not os.path.exists(g):
        continue
    a = np.asarray(Image.open(f).convert("L"), dtype=np.int32)
    b = np.asarray(Image.open(g).convert("L"), dtype=np.int32)
    if a.shape != b.shape:
        res[os.path.basename(f)] = {"shape_mismatch": [a.shape, b.shape]}
        continue
    fa, fb = a > 127, b > 127
    inter, union = (fa & fb).sum(), (fa | fb).sum()
    res[os.path.basename(f)] = {
        "shape": list(a.shape), "identical": bool(np.array_equal(a, b)),
        "pixel_agreement(binarized@127)": float((fa == fb).mean()),
        "foreground_IoU": float(inter / union) if union else 1.0,
        "max_abs_diff": int(np.abs(a - b).max()), "mean_abs_diff": float(np.abs(a - b).mean()),
    }
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
