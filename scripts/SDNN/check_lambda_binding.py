"""Show what the released SDNN.SDNN() does under Keras 2.10: build it unchanged, load the
pretrained weights, and run it on a few real Stokes pixels next to the TF1-faithful
rebuild in run_sdnn_tf2.py.  Usage (from the SDNN working copy):
    python check_lambda_binding.py /path/to/scripts/SDNN out.json
"""
import json, os, sys, traceback
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.getcwd())
import SDNN as sd
sys.argv, here, out = [sys.argv[0], "-", "-", "-"], sys.argv[1], sys.argv[2]
src = open(os.path.join(here, "run_sdnn_tf2.py")).read().split("out_dir, auth_dir, out_json")[0]
ns = {"__name__": "shim"}; exec(compile(src, "run_sdnn_tf2.py", "exec"), ns)
data = fits.getdata("inputs/cals_170713_162012.fts")[:, :, 300:302, 300:304]  # 8 pixels
X, h, w = sd.read_data(data)
res = {}
ns["K"].clear_session(); fixed = ns["build_sdnn"](); ns["load_keras224_weights"](fixed, "pretrained_model.h5")
yf = fixed.predict(X, verbose=0); res["tf1_faithful_rebuild"] = yf.round(5).tolist()
try:
    ns["K"].clear_session(); orig = sd.SDNN(); ns["load_keras224_weights"](orig, "pretrained_model.h5")
    yo = orig.predict(X, verbose=0); res["released_SDNN_unchanged"] = yo.round(5).tolist()
    res["max_abs_diff"] = float(np.abs(yo - yf).max())
except Exception as e:
    res["released_SDNN_unchanged_error"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-600:]
print(json.dumps(res, indent=1)); json.dump(res, open(out, "w"), indent=1)
