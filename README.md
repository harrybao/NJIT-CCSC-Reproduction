# NJIT CCSC Reproduction & CCDC Dataset Analysis

This is a reproduction study of the deep-learning tools in NJIT's **Community Coordinated Software Center (CCSC)**
(Prof. Jason T. L. Wang's group). It also includes a data-science analysis of the related AI/ML-ready solar datasets (**CCDC** / SolarDB).
It was prepared by Xiangsheng (Harry) Bao in October 2026 in response to Prof. Wang's request:

> "For each tool, click the Citation button, download the tool to reproduce all results and let me know how the tool works."

## Start here

| File | What it is |
|---|---|
| [`reports/CCSC_Reproduction_Report.md`](reports/CCSC_Reproduction_Report.md) | **Main English report** (executive summary, methods, results, comparison, limitations, how each tool works, research directions) |
| [`tool_inventory.md`](tool_inventory.md) | 19 tools × 16 fields (paper, citation, DOI, code, data, model, deps, weights, notebook, GPU, size, reported results, reproducibility, status) |
| [`reports/CCDC_Dataset_Analysis.md`](reports/CCDC_Dataset_Analysis.md) | Datasets: origin, formats, size, cleaning, missing values, metadata, leakage, AI-readiness scorecard, NSF/NASA AI-ready programmes, research questions |
| [`reports/Chinese_Research_Notes.md`](reports/Chinese_Research_Notes.md) | 中文学习笔记：背景知识、指标、逐工具 8 点讲解、发现与沟通建议 |
| [`reports/Email_Summary_to_Professor.md`](reports/Email_Summary_to_Professor.md) | Two draft replies to Prof. Wang (**not sent**; for Harry to review) |

## Status at a glance

| Tool | Status | One-line result |
|---|---|---|
| SEP-prediction | ✅ Successfully reproduced | 12/12 stored metric rows exact (test = 3 ARs) |
| RNN-CME-prediction | ✅ Successfully reproduced | 10/10 models give identical predictions (test = 2 ARs) |
| SolarUnet | ✅ Successfully reproduced | masks byte-identical after sorting `os.listdir()` |
| MIST | ✅ Reproduced reported TSS | 0.906 on the released random split → **0.197** on independent 2015–2018 data |
| SINet | ✅ Successfully reproduced | F10.7 exact to 4 decimals; F30 within 2 % |
| Kp-prediction | ✅ Successfully reproduced | within MC noise; persistence better at 1–2 h |
| DeepGeoCME | ✅ Successfully reproduced | identical confusion matrix, TSS 0.757 (33 events) |
| LSTM-flare-prediction | ◐ Partially | Table 2 retrained ×10 seeds; Table 5 protocol re-run; ≥M5.0 demo inverts labels |
| FlareML | ◐ Partially | demo reproduced; 31/40 demo test rows in training; AR-grouped CV lowers TSS |
| SolarFlareNet | ◐ Partially | stored outputs reproduced; 8/9 shipped models show no skill; test/train overlap |
| TSInet | ◐ Partially | default runs reproduced; CLI bug; naive baseline better on 30-day windows |
| CMETNet | ◐ Partially | numerical ensemble only; best-of-10 chosen on test year (bias quantified) |
| SYMHnet | ◐ Partially | stored outputs reproduced; released test path is effectively 1-step-ahead |
| Dst-prediction | ✗ Attempted, unsuccessful | released `predict()` substitutes 0.9×label; raw model R² 0.06–0.38 |
| FibrilNet, SDNN | ⛔ Blocked | model (and SDNN input) only on web.njit.edu |
| SolarCNN | ⛔ Blocked | weight file missing; shipped outputs re-scored |
| DeepHMI | ⛔ Blocked (compute) | ≈ 37 CPU-hours for the sampler; shipped output re-scored |
| MagNet | ⛔ Blocked | hosted on nature.njit.edu |
| LLMFlare | — Not attempted | gated Llama-3.1-8B + paid Gemini API + GPU (needs authorisation) |

These findings concern the **released code, data and weights**. The papers' full texts and original experiment code were
not accessible from the environment, so no claim is made about the published results themselves.

## Repository layout

```
README.md                  this file
tool_inventory.md          full tool inventory
papers/README.md           reading list with DOIs / arXiv IDs (PDFs not redistributable/unreachable)
tools/                     upstream commit hashes + frozen environments
datasets/README.md         where the data come from and how to re-fetch them
scripts/                   every analysis script (one folder per tool; common_metrics.py; nb_dump.py)
results/<tool>/            logs and JSON/CSV outputs of every run reported
reports/                   English report, CCDC analysis, Chinese notes, e-mail drafts
```

## Environment and access notes

* Hardware: 4 vCPU, 15 GB RAM, **no GPU** (cloud container).
* Reachable: GitHub, PyPI. **Unreachable:** `nature.njit.edu`, `web.njit.edu`, arXiv, Zenodo, journal sites,
  Google Drive, Wayback Machine.
* To lift the block, add those hosts under **Allowed domains** in the cloud environment's network settings, or run the
  scripts on a local machine.
* No paid services were used. No upstream repository was modified. No e-mail was sent.

## Reproducing a result (example)

```bash
# clone the tool and create the environment (see tools/README.md), then:
cp -r upstream/ccsc-tools/MIST work/MIST && cd work/MIST
python /path/to/repo/scripts/MIST/out_of_sample_eval.py \
    upstream/ccsc-tools/LSTM-flare-prediction/Table2_dataset/M \
    /path/to/repo/results/MIST/scaler_params.json out.json
```
