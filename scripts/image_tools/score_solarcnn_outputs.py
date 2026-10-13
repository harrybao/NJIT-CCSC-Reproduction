"""SolarCNN: re-score the authors' shipped predictions (the model's weight file
is not in the repository) against the shipped HMI labels, and compare with the
trivial baseline "use the MDI input unchanged".
Metrics follow the paper's choice: SSIM, Pearson r, PSNR (plus RMSE in Gauss).
Usage (cwd = SolarCNN repo): python score_solarcnn_outputs.py out.json
"""
import json
import os
import sys

import numpy as np
from astropy.io import fits
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load(p):
    return np.squeeze(fits.open(p)[0].data).astype(np.float64)


res = {}
for name in sorted(os.listdir("data/image/test")):
    stamp = name[4:20]
    x = load(f"data/image/test/{name}")
    y = load(f"data/label/test/hmi.M_720s.{stamp}TAI_0.fits")
    p = load(f"data/pred/enh_mdi_{stamp}SolarCNN.fits")
    rng = float(y.max() - y.min())
    row = {"shape": list(y.shape), "label_range": [float(y.min()), float(y.max())]}
    # oracle affine calibration of the prediction to the label (optimistic for SolarCNN)
    a, b = np.polyfit(p.ravel(), y.ravel(), 1)
    p_cal = a * p + b
    a2, b2 = np.polyfit(x.ravel(), y.ravel(), 1)   # same oracle calibration for the raw MDI input
    x_cal = a2 * x + b2
    row["pred_std_G"], row["label_std_G"], row["input_std_G"] = float(p.std()), float(y.std()), float(x.std())
    row["oracle_affine_a_b"] = [float(a), float(b)]
    for tag, img in [("MDI_input_as_is", x), ("SolarCNN_author_pred", p), ("MDI_input_oracle_affine", x_cal), ("SolarCNN_pred_oracle_affine", p_cal)]:
        row[tag] = {"SSIM": float(structural_similarity(y, img, data_range=rng)),
                    "PCC": float(np.corrcoef(y.ravel(), img.ravel())[0, 1]),
                    "PSNR": float(peak_signal_noise_ratio(y, img, data_range=rng)),
                    "RMSE_G": float(np.sqrt(np.mean((y - img) ** 2)))}
    # paper/training space: clip to +-2000 G and divide by 2000 (train.py), SSIM with max_val=2
    n = lambda im: np.clip(im, -2000, 2000) / 2000.0
    for tag, img in [("norm_MDI_input", x), ("norm_MDI_input_oracle_affine", x_cal), ("norm_SolarCNN_pred_oracle_affine", p_cal)]:
        row[tag] = {"SSIM(max_val=2)": float(structural_similarity(n(y), n(img), data_range=2.0)),
                    "PSNR(max_val=2)": float(peak_signal_noise_ratio(n(y), n(img), data_range=2.0))}
    res[stamp] = row
    print(stamp, json.dumps({k: v for k, v in row.items() if isinstance(v, dict)}))
json.dump(res, open(sys.argv[1], "w"), indent=1)
