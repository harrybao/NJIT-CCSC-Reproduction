# NJIT CCSC Reproduction & CCDC Dataset Analysis

This is a reproduction study of the deep-learning tools in NJIT's **Community Coordinated Software Center (CCSC)**
(<https://nature.njit.edu/solardb/ccsc>, Prof. Jason T. L. Wang's group). It also includes a data-science analysis of the
**Community Coordinated Data Center (CCDC)** dataset FlareDB and of the related AI/ML-ready data programmes.
It was prepared by Xiangsheng (Harry) Bao in October 2026 in response to Prof. Wang's request:

> "For each tool, click the Citation button, download the tool to reproduce all results and let me know how the tool works."

## Start here

| File | What it is |
|---|---|
| [`reports/CCSC_Reproduction_Report.md`](reports/CCSC_Reproduction_Report.md) | **Main English report**: executive summary, methods, results, comparison with the papers' tables, limitations, how each tool works, research directions |
| [`tool_inventory.md`](tool_inventory.md) | 19 tools × 16 fields (paper, Citation-button record, DOI, code, data, model, deps, weights, notebook, GPU, size, reported results, reproducibility, status) |
| [`papers/citations/`](papers/citations/README.md) | What every Citation button points to (Zenodo DOI, cited version, APA/BibTeX), plus snapshots of the CCSC and CCDC pages |
| [`reports/CCDC_Dataset_Analysis.md`](reports/CCDC_Dataset_Analysis.md) | CCDC (= FlareDB) and the bundled datasets: origin, formats, size, cleaning, missing values, metadata, leakage, AI-readiness scorecard, NSF/NASA programmes, research questions |
| [`reports/Chinese_Research_Notes.md`](reports/Chinese_Research_Notes.md) | 中文学习笔记：背景知识、指标、逐工具 8 点讲解、发现与沟通建议 |
| [`reports/Email_Summary_to_Professor.md`](reports/Email_Summary_to_Professor.md) | Two draft replies to Prof. Wang (**not sent**; for Harry to review) |

## Status at a glance

Each tool was run at the version its Citation button cites (§3.5 of the report).
* **Tool level**: the released model and data reproduce the authors' stored outputs.
* **Paper level**: a headline number of the paper is reproduced.

| Tool | Status | One-line result |
|---|---|---|
| SINet | ✅ Reproduced (tool + paper) | F10.7 RMSE 3.88 / 16.32 sfu (1 / 60 d), as in the paper |
| DeepGeoCME | ✅ Reproduced (tool + paper) | identical confusion matrix; TSS 0.757, as in the paper |
| SEP-prediction | ✅ Tool level | 12/12 stored metric rows exact; the paper's CV numbers are not re-runnable from the release |
| SDNN | ✅ Tool level | 5/5 output maps match (r = 1.000000); paper's ME reference maps not shipped |
| FibrilNet | ✅ Tool level | IoU vs stored masks 0.901, the same as between our own MC-dropout runs (0.900) |
| SolarUnet | ✅ Tool level | masks byte-identical after sorting `os.listdir()` |
| Kp-prediction | ✅ Tool level | within MC noise; persistence better at 1–2 h |
| SolarCNN | ◐ Partially | shipped predictions to 3e-5 G; with the training normalisation SSIM 0.812 → 0.892 (paper 0.833 → 0.904) on 3 of 76 samples |
| LSTM-flare-prediction | ◐ Partially | cited v1.0.2 demo reproduced; retrain at paper thresholds TSS 0.812 / 0.730 / 0.580 vs paper 0.877 / 0.790 / 0.607 |
| RNN-CME-prediction | ◐ Partially | demo identical; on the paper's own test set GRU TSS 0.31–0.49 vs paper 0.45–0.60; data column-order inconsistency |
| SYMHnet | ◐ Partially | cited v1.0 matches the paper's RMSE for 3 of 7 storms; released test target is the next row |
| MIST | ◐ Partially | 0.906 on the released random split → **0.197** on independent 2015–2018 data |
| SolarFlareNet | ◐ Partially | stored outputs reproduced; 8/9 shipped models show no skill (paper TSS 0.71–0.84) |
| FlareML | ◐ Partially | demo reproduced; 31/40 demo test rows in training; AR-grouped CV lowers TSS |
| TSInet | ◐ Partially | default runs reproduced; CLI bug; naive baseline better on 30-day windows |
| CMETNet | ◐ Partially | numerical ensemble only; best-of-10 chosen on the test year; 14.2 h vs paper 10.32 h |
| Dst-prediction | ✗ Attempted, unsuccessful | released `predict()` substitutes 0.9 × label; raw model R² 0.06–0.38 |
| DeepHMI | ⛔ Blocked (compute) | ≈ 37 CPU-hours for the sampler; shipped output re-scored |
| MagNet (SolarDB) | ⛔ Blocked | package lacks `layers/attention.py` imported by its model file |
| LLMFlare | — Not attempted | gated Llama-3.1-8B + paid Gemini API + GPU (needs authorisation) |

These findings concern the **released code, data and weights at the cited versions**, compared with the papers
where their evaluation data are shipped. They are not claims about the original experiments.

## Repository layout

```
README.md                  this file
tool_inventory.md          full tool inventory
papers/citations/          Citation-button records (Zenodo DOIs via DataCite), CCSC/CCDC page snapshots
papers/README.md           reading list with DOIs / arXiv IDs (PDFs are not redistributed)
tools/                     upstream commits, cited versions, frozen environments
datasets/README.md         where the data and model files come from and how to re-fetch them
scripts/                   every analysis script (one folder per tool; common_metrics.py; nb_dump.py; citations/)
results/<tool>/            logs and JSON/CSV outputs of every run reported
results/cited_versions/    runs and diffs at the Citation-button versions
reports/                   English report, CCDC analysis, Chinese notes, e-mail drafts
```

## Environment and access notes

* Hardware: 4 vCPU, 15 GB RAM, **no GPU** (cloud container).
* Network: full access was enabled by the applicant on 2026-10-13.
  * Reached: `nature.njit.edu`, `web.njit.edu`, arXiv, nature.com, frontiersin.org, Google Drive, GitHub, PyPI.
  * `zenodo.org`: HTTP 504 on 2026-10-13, reachable on 2026-10-14. The six direct-upload archives were compared with GitHub (`results/cited_versions/zenodo_records/`): five match, apart from a one-line `import os` missing from MIST's archived `test.py`. SolarCNN's record is restricted-access.
  * Also unreachable: the A&A, AGU/Wiley and Springer full texts (403 or paywall).
* Model files downloaded from the authors' links are not redistributed. Their SHA-256 hashes are in `results/downloaded_artefacts_sha256.txt`.
* No paid services were used. No upstream repository was modified. No e-mail was sent.

## Reproducing a result (example)

```bash
# clone the tool and create the environment (see tools/README.md), then:
cp -r upstream/ccsc-tools/MIST work/MIST && cd work/MIST
python /path/to/repo/scripts/MIST/out_of_sample_eval.py \
    upstream/ccsc-tools/LSTM-flare-prediction/Table2_dataset/M \
    /path/to/repo/results/MIST/scaler_params.json out.json
```
