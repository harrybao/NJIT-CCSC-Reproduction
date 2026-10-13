"""Run the released FibrilNet inference path (run_FibrilNet.ipynb -> helper.predict,
MC-dropout T=5) under TensorFlow 2.10 / Keras 2.10 on CPU.

The released code targets TF 1.11 / Keras 2.2.4 and calls `K.tf.*`, which Keras 2.10
no longer exposes. The shims only give FibrilNet.py's `K` a `tf` attribute (= `tf.compat.v1`; same ops:
max_pool_with_argmax, scatter_nd, variable_scope) and re-supply the `np` name that
Keras 2.2.4's `from keras.layers import *` used to leak into FibrilNet.py / helper.py. The architecture, the weights
(model.hdf5 from web.njit.edu, Keras 2.2.4 file) and the prediction code are the
authors' and are not modified.

Run from a disposable working copy of the cited release (git archive v1.6) that
contains model/model.hdf5:

    python run_fibrilnet_tf2.py SEED T OUT_RESULTS_DIR
"""
import os
import random
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import matplotlib

matplotlib.use("Agg")
import numpy as np
import tensorflow as tf
from keras import backend as _keras_backend

sys.path.insert(0, os.getcwd())
import FibrilNet as fn  # noqa: E402
import helper  # noqa: E402


class _BackendWithTF:
    """keras.backend as FibrilNet.py expects it: Keras 2.2.4 exposed TF as `K.tf`."""
    tf = tf.compat.v1

    def __getattr__(self, name):
        return getattr(_keras_backend, name)


fn.K = _BackendWithTF()  # only FibrilNet's custom layers see the shim
# Keras 2.2.4's `from keras.layers import *` leaked `np`; Keras 2.10 does not
fn.np = helper.np = np

seed, T, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
for sub in ("predicted_mask", "fibrils_on_ha", "uncertainty"):
    os.makedirs(os.path.join(out, sub), exist_ok=True)

model = fn.FibrilNet("model/model.hdf5")


# helper.predict hard-codes T=5; let the caller choose T (5 = released default) and keep the
# mean-probability / uncertainty maps that produced each saved mask, in os.listdir order
order = os.listdir("data/test/")
maps = []


def predict_mc_recorded(model, image, **_released_kwargs):  # released call passes T=5
    p, a, e = fn.predict_mc(model, image, T=T)
    maps.append((p, a, e))
    return p, a, e


helper.predict_mc = predict_mc_recorded
helper.predict("data/test/", out.rstrip("/") + "/", model)
np.savez_compressed(os.path.join(out, "mc_maps.npz"),
                    **{f"{n}__{k}": m[i].astype(np.float32)
                       for n, m in zip(order, maps) for i, k in enumerate(("prob", "aleatoric", "epistemic"))})
print("done", out)
