"""Run the released SDNN inversion (SDNN.py __main__ / SDNN_test.py) under TensorFlow 2.10
on CPU, then compare with the authors' stored outputs.

Why a shim is needed. SDNN.SDNN() builds its four Stokes branches in a loop with
    Lambda(lambda x: x[:, :, i])            # late-bound loop variable `i`
    Lambda(lambda x: K.expand_dims(out, -1))  # ignores x, closes over `out`
Under TF 1.11 / Keras 2.2.4 (graph mode) each Lambda is evaluated once while the graph
is built, so `i` and `out` hold the intended values (branch i gets Stokes channel i).
Keras 2.10 re-executes the Python lambdas when predict() traces the model, by which
time `i == 3` and `out` is the last tensor of the function. Run unchanged, predict()
fails with "TypeError: Could not build a TypeSpec for KerasTensor ..." (see
check_lambda_binding.py / results/SDNN/check_lambda_binding.json). `build_sdnn()` below is the
authors' SDNN() verbatim except that the two lambdas are bound the way TF1 evaluated
them (`lambda x, i=i: x[:, :, i]`, `lambda x: K.expand_dims(x, -1)`).

Weights (pretrained_model.h5 from web.njit.edu, a Keras 2.2.4 weights file) are copied
layer by layer by name: Keras 2.2.4 names layers from `_1` and Keras 2.10 from `` (no
suffix), so `conv1d_k` in the file is `conv1d_{k-1}` here. Every weighted layer must be
matched exactly once, with identical shapes, or the script stops.

read_data(), inverse() and save_results() are imported from the authors' SDNN.py and
run unchanged (the module only gets back the `np` / `K` names that Keras 2.2.4's
`from keras.layers import *` used to leak into it). Run from a disposable copy of the cited release (git archive v1.3):

    python run_sdnn_tf2.py OUT_DIR AUTHOR_OUTPUTS_DIR out.json
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import h5py
import numpy as np
from astropy.io import fits

sys.path.insert(0, os.getcwd())
import SDNN as sd  # noqa: E402  (authors' module: from keras.layers import * etc.)
from keras import backend as K  # noqa: E402
from keras.layers import Concatenate, Conv1D, Dense, Flatten, Input, Lambda, MaxPool1D, Reshape  # noqa: E402
from keras.models import Model  # noqa: E402

# Keras 2.2.4's `from keras.layers import *` leaked `np` and `K` into SDNN.py; Keras 2.10 does not
sd.np, sd.K = np, K


def build_sdnn():
    num_channels = 4
    input = Input(shape=(60, num_channels))
    branch_outputs = []
    for i in range(num_channels):
        out = Lambda(lambda x, i=i: x[:, :, i])(input)            # was: lambda x: x[:, :, i]
        out = Lambda(lambda x: K.expand_dims(x, -1))(out)          # was: K.expand_dims(out, -1)
        conv0_1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(out)
        conv0_2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv0_1)
        conv0 = MaxPool1D(pool_size=2)(conv0_2)
        conv1_1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv0)
        conv1_2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1_1)
        conv1 = MaxPool1D(pool_size=2)(conv1_2)
        conv2_1 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv1)
        conv2_2 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv2_1)
        conv2 = MaxPool1D(pool_size=2)(conv2_2)
        out = Flatten()(conv2)
        out = Dense(240)(out)
        branch_outputs.append(out)
    out = Concatenate(axis=-1)(branch_outputs)
    out = Reshape((240, num_channels))(out)
    conv0_1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(out)
    conv0 = MaxPool1D(pool_size=2)(conv0_1)
    conv1_1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv0)
    conv1 = MaxPool1D(pool_size=2)(conv1_1)
    conv2_1 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv1)
    conv2 = MaxPool1D(pool_size=2)(conv2_1)
    conv3_1 = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(conv2)
    out = Flatten()(conv3_1)
    layer1 = Dense(2048, activation='relu')(out)
    layer2 = Dense(1024, activation='relu')(layer1)
    output = Dense(5, activation='linear')(layer2)
    return Model(input, output)


def file_name(tf2_name):
    """Keras 2.10 'conv1d' / 'conv1d_3' -> Keras 2.2.4 'conv1d_1' / 'conv1d_4'."""
    m = re.fullmatch(r"(.*?)(?:_(\d+))?", tf2_name)
    base, k = m.group(1), m.group(2)
    return f"{base}_{int(k) + 1 if k else 1}"


def load_keras224_weights(model, path):
    f = h5py.File(path, "r")
    g = f["model_weights"] if "model_weights" in f else f
    stored = {n.decode(): g[n.decode()] for n in g.attrs["layer_names"]}
    used = set()
    for layer in model.layers:
        if not layer.weights:
            continue
        name = file_name(layer.name)
        grp = stored[name]
        arrs = [np.asarray(grp[w]) for w in grp.attrs["weight_names"]]
        assert [a.shape for a in arrs] == [tuple(w.shape) for w in layer.weights], name
        layer.set_weights(arrs)
        used.add(name)
    expected = {n for n, grp in stored.items() if len(grp.attrs["weight_names"])}
    assert used == expected, (sorted(expected - used), sorted(used - expected))
    return len(used)


out_dir, auth_dir, out_json = sys.argv[1:4]
K.clear_session()
model = build_sdnn()
n_loaded = load_keras224_weights(model, "pretrained_model.h5")
print("weighted layers loaded:", n_loaded)

summary = {"weighted_layers_loaded": n_loaded, "files": {}}
for file in sorted(os.listdir("inputs")):
    if not file.endswith((".fts", ".fits")):
        continue
    hdu = fits.open(os.path.join("inputs", file))
    hdu.verify("fix")
    data = hdu[0].data
    t0 = time.time()
    test_data, h, w = sd.read_data(data)
    t1 = time.time()
    pred = sd.inverse(test_data, model)
    t2 = time.time()
    stamp = file[: file.find(".")]
    sd.save_results(pred, out_dir, stamp, h, w, save_mag_field_o=False)
    summary["timing_s"] = {"read_data": round(t1 - t0, 1), "inverse": round(t2 - t1, 1)}
    for q in ("bx", "by", "bz", "Doppler_Width", "LOS_Velocity"):
        fn = f"{q}_{stamp}.fits"
        a = fits.getdata(os.path.join(out_dir, fn)).astype(np.float64)
        b = fits.getdata(os.path.join(auth_dir, fn)).astype(np.float64)
        d = a - b
        summary["files"][fn] = {
            "shape": list(a.shape), "author_dtype": str(fits.getdata(os.path.join(auth_dir, fn)).dtype),
            "max_abs_diff": float(np.abs(d).max()), "mean_abs_diff": float(np.abs(d).mean()),
            "author_abs_mean": float(np.abs(b).mean()), "author_std": float(b.std()),
            "pearson_r": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            "frac_abs_diff_le_1e-3_of_author_std": float((np.abs(d) <= 1e-3 * b.std()).mean()),
        }
print(json.dumps(summary, indent=1))
json.dump(summary, open(out_json, "w"), indent=1)
