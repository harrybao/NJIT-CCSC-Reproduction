"""Shared binary forecast-verification metrics used across tools.

All metrics follow the standard space-weather definitions used in the
Wang-group papers (e.g. Liu et al. 2019, ApJ 877:121; Abduallah et al. 2023):
  recall = TP/(TP+FN); precision = TP/(TP+FP); ACC = (TP+TN)/N
  BACC = (TP/(TP+FN) + TN/(TN+FP))/2
  HSS  = 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
  TSS  = TP/(TP+FN) - FP/(FP+TN)
"""
import numpy as np


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    TP = int((y_true & y_pred).sum()); FN = int((y_true & ~y_pred).sum())
    FP = int((~y_true & y_pred).sum()); TN = int((~y_true & ~y_pred).sum())
    div = lambda a, b: float(a) / b if b else float("nan")
    rec, spec = div(TP, TP + FN), div(TN, TN + FP)
    hss_den = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN, "N": TP + FN + FP + TN,
        "recall": rec, "precision": div(TP, TP + FP),
        "ACC": div(TP + TN, TP + FN + FP + TN),
        "BACC": (rec + spec) / 2,
        "HSS": div(2 * (TP * TN - FP * FN), hss_den),
        "TSS": rec - div(FP, FP + TN),
    }


def best_threshold(y_true, prob, grid=None, key="TSS"):
    grid = np.linspace(0, 1, 201) if grid is None else grid
    best = max(grid, key=lambda t: binary_metrics(y_true, prob >= t)[key])
    return float(best), binary_metrics(y_true, prob >= best)


def fmt(m, keys=("recall", "precision", "ACC", "BACC", "HSS", "TSS")):
    return " ".join(f"{k}={m[k]:.3f}" for k in keys)
