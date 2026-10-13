"""SolarCNN with the authors' trained model (model_solarcnn.zip from the README's Google Drive
link; saved_model.pb and variables.index are byte-identical to the repository's copies).

Scores the 3 shipped test pairs (MDI input -> HMI label) in the space the model was trained
in (train.py: clip to +-2000 G, divide by 2000) with the paper's metrics (SSIM with
max_val=2 as in model.py, Pearson r, PSNR with max_val=2), for:
  * released_test_py : prediction from the raw (unnormalised) input, as test.py does
  * normalised_input : prediction from the clipped/scaled input, as in training
  * MDI_input        : the input itself (the paper's "None" row is the reference to beat)
Usage (cwd = SolarCNN working copy with model/model_solarcnn complete): python eval_normalised.py out.json
"""
import json, os, sys
import numpy as np
import tensorflow as tf
from astropy.io import fits
sys.path.insert(0, os.getcwd())
from model import mix_loss, ssim_metric  # noqa: E402

model = tf.keras.models.load_model("model/model_solarcnn", custom_objects={"mix_loss": mix_loss, "ssim_metric": ssim_metric})
norm = lambda a: np.clip(a, -2000.0, 2000.0) / 2000.0


def metrics(y, p):
    yt, pt = tf.constant(y[None, :, :, None], tf.float32), tf.constant(p[None, :, :, None], tf.float32)
    return {"SSIM": float(tf.image.ssim(yt, pt, max_val=2.0)[0]), "PCC": float(np.corrcoef(y.ravel(), p.ravel())[0, 1]),
            "PSNR": float(tf.image.psnr(yt, pt, max_val=2.0)[0])}


res = {"per_sample": {}}
for name in sorted(os.listdir("data/image/test")):
    stamp = name[4:20]
    x = np.squeeze(fits.getdata(f"data/image/test/{name}")).astype(np.float32)
    y = np.squeeze(fits.getdata(f"data/label/test/hmi.M_720s.{stamp}TAI_0.fits")).astype(np.float32)
    p_raw = model.predict(x.reshape(1, 256, 256, 1), verbose=0)[0, :, :, 0]
    p_norm = model.predict(norm(x).reshape(1, 256, 256, 1), verbose=0)[0, :, :, 0]
    res["per_sample"][stamp] = {"MDI_input": metrics(norm(y), norm(x)),
                                "released_test_py(raw input)": metrics(norm(y), norm(p_raw)),
                                "normalised_input": metrics(norm(y), p_norm),
                                "std_G": {"label": float(y.std()), "input": float(x.std()),
                                          "pred_raw_input": float(p_raw.std()), "pred_normalised_input_x2000": float(p_norm.std() * 2000)}}
for k in ("MDI_input", "released_test_py(raw input)", "normalised_input"):
    res[f"mean_{k}"] = {m: round(float(np.mean([v[k][m] for v in res["per_sample"].values()])), 4) for m in ("SSIM", "PCC", "PSNR")}
res["paper_table3"] = {"SolarCNN": {"SSIM": 0.9039, "PCC": 0.8842, "PSNR": 37.40}, "None": {"SSIM": 0.8334, "PCC": 0.8627, "PSNR": 34.73}}
print(json.dumps({k: v for k, v in res.items() if k != "per_sample"}, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
