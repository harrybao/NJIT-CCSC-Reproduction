"""Check whether each shipped SolarFlareNet checkpoint fully matches the model
architecture built by SolarFlareNet_model.py (load_weights uses expect_partial(),
which silently ignores mismatches).

Usage (cwd = SolarFlareNet working copy): python checkpoint_match.py out.json
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import tensorflow as tf  # noqa: E402
from SolarFlareNet_model import SolarFlareNet  # noqa: E402

res = {}
for w in ["24", "48", "72"]:
    for c in ["C", "M", "M5"]:
        ckpt = f"models/{w}/{c}/model_weights"
        ck_vars = tf.train.list_variables(ckpt)
        m = SolarFlareNet()
        m.build_base_model((10, 9), verbose=False)
        m.models()
        status = m.model.load_weights(ckpt)
        try:
            status.assert_existing_objects_matched()  # every model variable found in ckpt?
            matched = True; err = ""
        except AssertionError as e:
            matched = False; err = str(e).splitlines()[0][:300]
        n_model_w = len(m.model.weights)
        ck_model_w = [n for n, _ in ck_vars if n.startswith("layer") and "OPTIMIZER" not in n and ".ATTRIBUTES/VARIABLE_VALUE" in n]
        res[f"{c}_{w}"] = {"all_model_weights_restored": matched, "first_error_line": err,
                          "n_model_weight_tensors": n_model_w,
                          "n_checkpoint_layer_tensors": len(ck_model_w),
                          "checkpoint_mtime_note": "see git history; files shipped in repo"}
        print(c, w, matched, n_model_w, len(ck_model_w), err[:150])
json.dump(res, open(sys.argv[1], "w"), indent=1)
