# Draft reply to Prof. Jason T. L. Wang (for Harry to review — NOT sent)

> **中文说明（给 Harry）：**
> * 下面有 A（简短版）和 B（详细版）两个版本。我**建议发 A**，并在邮件里附上报告的链接或 PDF；B 可以留到教授追问时再用。
> * 两个页面（CCSC 和 CCDC）现在都已打开，19 个 Citation 按钮也都核对过了。按钮指向 Zenodo 软件存档，详见 `papers/citations/README.md`。发送前你仍可以自己点几个按钮抽查一下。
> * 措辞原则：
>   * 先讲做了什么、学到了什么；
>   * 问题写成"我在发布的代码/数据包里观察到……"，并以提问的方式提出，**不下"论文有错"的结论**；
>   * 语气感谢、谦虚，体现主动性。
> * 邮件中 `[link]` 处换成 GitHub 仓库或报告 PDF 的链接。如果仓库是私有的，请先导出 PDF，或把教授加为协作者。
> * 正文只挑了最有代表性的几点，其余都放在报告里。

---

## Version A — concise (recommended)

**Subject:** CCSC tools – reproduction notes and questions (Xiangsheng Bao)

Dear Professor Wang,

Thank you for pointing me to CCSC and CCDC. As you suggested, I will describe my interest in your group's research in my statement of purpose. I followed the Citation button of each of the 19 tools to its Zenodo record and ran each tool at the cited version, with your released code, data and pretrained weights, on a CPU-only machine. I compared the outputs with the stored results and with the tables in the papers. The write-up covers how each tool works, what reproduced and what I could not run: [link].

In brief:
* SINet and DeepGeoCME reproduce the papers' headline numbers (F10.7 RMSE 3.88 / 16.32 sfu; TSS 0.757).
* SEP, SDNN, FibrilNet, SolarUnet and KpNet reproduce the stored outputs exactly or within Monte-Carlo noise. For SDNN, all five maps correlate at r = 1.000000.
* With the model from the SolarCNN README's Google Drive link, SolarCNN also improves on its MDI input by about as much as the paper reports. This only shows once the training normalisation (clip ±2000 G, ÷2000) is applied, which `test.py` currently omits.

Because my interest is out-of-sample validation, I also tested several models on independent data:
1. Your 2019 LSTM dataset uses a chronological split with no shared active regions. Retraining it with the released code gave TSS 0.81 / 0.73 / 0.58 (≥M5.0 / ≥M / ≥C) at the paper's thresholds, somewhat below Table 4 (0.88 / 0.79 / 0.61).
2. MIST's released data are the LSTM ≥M training file, split at random into overlapping 24-hour windows. On your 2015–2018 test file, the same weights give TSS ≈ 0.20, versus 0.91 on the released split.
3. In a few packages the released evaluation seems to differ from the paper, and I would be grateful to know whether the papers used different code:
   * The Dst tool's `predict()` uses the test labels.
   * In the RNN-CME release the two copies of the data store the SHARP columns in different orders.
   * SYMHnet v1.0 reproduces the paper's RMSE for storms 31, 33 and 42, but its test target is the next 1- or 5-minute row.

I would be glad to help turn these checks into a reusable validation and AI-readiness toolkit for CCSC/CCDC datasets such as FlareDB, which seems relevant to the NSF AI Datasets programme.

Kind regards,
Xiangsheng (Harry) Bao

---

## Version B — detailed

**Subject:** CCSC/CCDC – reproduction report, how the tools work, and some validation questions

Dear Professor Wang,

Thank you for your guidance. Following your suggestion, I will state my interest in your research in my Fall 2027 statement of purpose. I have also worked through the CCSC tools and the CCDC dataset.

**What I did.**
* For each of the 19 tools on the CCSC page, I followed the Citation button to its Zenodo record. Thirteen of these are GitHub releases (e.g. LSTM v1.0.2, SYMHnet v1.0), and I ran or diffed exactly those versions.
* For the six direct Zenodo uploads I ran the GitHub versions and checked them file by file against the archives.
  * Five match in content, except that MIST's archived `test.py` is missing an `import os`, which the GitHub version has.
  * SolarCNN's record is restricted-access. Would it be possible to access its 76-sample test set?
* I ran the released code with the released models on a CPU machine. I compared the results with the stored outputs and with the papers' tables (16 full texts).
* The report explains each tool (research question, inputs, model, pipeline, outputs, metrics, limitations), what was reproduced, and what was blocked: [link]. All scripts and logs are in the same repository.

**Reproduced.**
* SINet: F10.7 RMSE 3.88 (1 d) and 16.32 (60 d), as in the paper.
* DeepGeoCME: identical confusion matrix, TSS 0.757, as in the paper.
* SEP-prediction: all 12 stored metric tables, exactly.
* SDNN, run with the model and input from web.njit.edu: all five maps match the stored ones (r = 1.000000). Under TF 2.x the per-branch `Lambda` closures need early binding; I documented the fix.
* FibrilNet: the stored masks agree with mine (IoU 0.90) exactly as well as my own MC-dropout runs agree with each other. The shipped `test.py` calls an undefined `predict(...)`, so I used the notebook path.
* SolarUnet: identical masks, once input files are read in sorted order.
* KpNet: within Monte-Carlo variation.

**Partly reproduced, with questions.**
1. *LSTM 2019 (v1.0.2).*
   * The cited demo models reproduce the stored demo outputs (99–99.5 % agreement).
   * The Table 2 split (2010–13 / 2014 / 2015–18, no shared ARs) is an excellent design. Retraining 10 seeds at the paper's thresholds gave TSS 0.812 / 0.730 / 0.580, against 0.877 / 0.790 / 0.607 in Table 4.
   * The Table 5 ≥C protocol under TF 2.10 gave 0.581 against 0.612.
2. *MIST.*
   * Its processed dataset matches the LSTM ≥M training file value for value. The windows overlap by 23 hours and are split at random. Using the `X_train.npy` included in the Zenodo archive, 95.8 % of test windows share 23 of their 24 hours with a training window.
   * On the 2015–2018 file, the same weights give TSS 0.197 (AUC 0.83).
3. *RNN-CME.*
   * On the full test sets in `CME_data_samples`, the released models reach GRU TSS 0.31–0.49, against 0.45–0.60 in Table 4.
   * The 18 SHARP columns in `CMEpredict/` are a permutation of those in `CME_data_samples/`, and only the latter labelling is physically consistent. The code's feature ranking matches Table 2 under the former labelling.
4. *SYMHnet v1.0.*
   * Storms 31, 33 and 42 reproduce the paper's per-storm RMSE and R² within about 2 %; storms 28, 36, 37 and 40 do not.
   * The 1-hour and 2-hour data folders are byte-identical, and the test target is the next row.
5. *SolarCNN.* With the training normalisation, SSIM rises from 0.812 to 0.892 and PSNR from 32.6 to 36.8 dB on the three shipped pairs, consistent with the paper's Table 3. Without it (as in `test.py`) there is no gain.
6. *Other released evaluation paths.*
   * Dst: `predict()` replaces most predictions with 0.9 × the test label; the raw network gives R² 0.06–0.38.
   * SolarFlareNet: 8 of the 9 shipped checkpoints show no skill on the bundled 2022 test file, which also overlaps the training file.
   * CMETNet: selects its best training run on the test year.

**Not run.**
* DeepHMI: the 1,000-step sampler needs about 37 CPU-hours; I re-scored the shipped output.
* MagNet: the package on SolarDB imports `layers.attention` (PAM, CAM), which is not included. Could this module be added?
* LLMFlare: needs gated Llama weights and a paid Gemini API.

I may well be missing context about the original experiments, and I would appreciate knowing where the published evaluations used different code or data.

**CCDC.**
* The page currently lists FlareDB. I profiled it from the *Scientific Data* paper, its Zenodo record and the GitHub list.
* It is a well-documented event database: 151 ≥M5.0 flares, aligned HMI/AIA patches.
* To serve forecasting, it would need matched negatives and AR-grouped split manifests. The GitHub list (103 events) also lags the paper's 151.
* This connects naturally to "AI-ready" release practices (split manifests, data dictionaries, schema/column checks, checksums) and to external, cross-instrument validation of reconstructed fields such as MagNet. Those are directions that NSF 26-512 explicitly supports.

I would be glad to continue in this direction. Thank you again for your time.

Kind regards,
Xiangsheng (Harry) Bao
