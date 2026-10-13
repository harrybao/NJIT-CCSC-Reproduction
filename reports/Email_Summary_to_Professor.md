# Draft reply to Prof. Jason T. L. Wang (for Harry to review — NOT sent)

> **中文说明（给 Harry）：**
> * 下面有 A（简短版）和 B（详细版）两个版本。我**建议发 A**，并在邮件里附上报告的链接或 PDF；B 可以留到教授追问时再用。
> * **发送之前请你自己先用浏览器打开** <https://nature.njit.edu/solardb/ccsc> 和 <https://nature.njit.edu/solardb/ccdc>，核对两件事：①工具清单是否就是 `tool_inventory.md` 里那 19 个；②每个 Citation 按钮里的引用信息是否与清单一致。我的云端环境打不开这两个页面，你在自己的电脑上很可能可以。如有出入，把差异告诉我，我来更新报告。
> * 措辞原则：①先讲做了什么、学到了什么；②问题写成"我在发布的代码/数据包里观察到……"，并以提问的方式提出，**不下"论文有错"的结论**；③语气感谢、谦虚，体现主动性。
> * 邮件中 `[link]` 处换成 GitHub 仓库或报告 PDF 的链接。如果仓库是私有的，请先导出 PDF，或把教授加为协作者。

---

## Version A — concise (recommended)

**Subject:** CCSC tools – reproduction notes and questions (Xiangsheng Bao)

Dear Professor Wang,

Thank you for pointing me to CCSC and CCDC. I have now worked through all 19 tools in the ccsc-tools GitHub collection. I ran each one with your released code, data and pretrained weights on a CPU-only machine and compared the outputs with the stored results. I have written up how each tool works, what reproduced, and what I could not run: [link].

In brief:
* Several tools reproduced exactly: SEP-prediction (all 12 metric tables), RNN-CME, SolarUnet, MIST's reported TSS of 0.906, SINet's F10.7 results and DeepGeoCME.
* FibrilNet, SDNN and SolarCNN need model files hosted on web.njit.edu, Zenodo or Google Drive that I could not download. DeepHMI's 1,000-step diffusion sampler needs a GPU.

Because my interest is out-of-sample validation, I also tested the released models on chronologically independent data:
1. Your 2019 LSTM dataset uses a chronological split with no shared active regions. Retraining on it gave TSS ≈ 0.88 (≥M5.0), 0.75 (≥M) and 0.59 (≥C).
2. MIST's released data turn out to be the LSTM ≥M training file, split at random into overlapping 24-hour windows. On your 2015–2018 test file, the same MIST weights give TSS ≈ 0.20, versus 0.91 on the released split.
3. In a few released test scripts the evaluation path seems to differ from the paper's description. For example, the Dst tool's `predict()` uses the test labels, and SYMHnet's 1-hour and 2-hour data folders are identical. I would be grateful to know whether the papers' evaluations used different code.

These are observations about the released packages only, since I could not access the full papers from my environment. I would be glad to help turn the checks into a reusable validation and AI-readiness toolkit for CCSC/CCDC datasets, which seems relevant to the new NSF AI Datasets programme.

Kind regards,
Xiangsheng (Harry) Bao

---

## Version B — detailed

**Subject:** CCSC/CCDC – reproduction report, how the tools work, and some validation questions

Dear Professor Wang,

Thank you for your guidance. Following your suggestion, I will state my interest in your research in my Fall 2027 statement of purpose. I have also worked through the CCSC tools and the CCDC datasets.

**What I did.**
* I identified 19 tools in the ccsc-tools collection, from FlareML and the 2019 LSTM through SolarFlareNet, SEP, CMETNet, DSTT, KpNet, SYMHnet, TSInet, SolarUnet, FibrilNet, SDNN, SolarCNN, DeepHMI, MIST, SINet, LLMFlare and DeepGeoCME.
* For each, I recorded the publication, data, weights and environment, and ran the released code with the released models on a CPU machine.
* The report explains each tool (research question, inputs, model, pipeline, outputs, metrics, limitations), what was reproduced, and what was blocked: [link]. All scripts and logs are in the same repository.

**Reproduced.**
* SEP-prediction: all 12 stored metric tables, exactly.
* RNN-CME: identical predictions for all 10 models.
* SolarUnet: identical masks, once the input files are read in sorted order.
* MIST: TSS 0.906.
* SINet: F10.7 RMSEs to four decimals.
* KpNet: within Monte-Carlo variation.
* DeepGeoCME: identical confusion matrix.

**Not run.**
* FibrilNet and SDNN: models on web.njit.edu, which I could not reach.
* SolarCNN: weight file not in the repository.
* DeepHMI: needs a GPU for the 1,000-step sampler; I re-scored the shipped output instead.
* LLMFlare: needs gated Llama weights and a paid Gemini API.

**Out-of-sample checks (my main interest).**
1. *LSTM 2019.* The Table 2 split (2010–13 / 2014 / 2015–18, no shared ARs) is an excellent design. Retraining 10 seeds with the released code gave TSS 0.877 ± 0.035 (≥M5.0), 0.748 ± 0.021 (≥M) and 0.589 ± 0.009 (≥C) with the test-optimised threshold, and 0.825 / 0.739 / 0.584 when the threshold is chosen on the 2014 validation year.
2. *MIST.* Its processed dataset matches the LSTM ≥M training file value for value. The windows overlap by 23 hours and are split at random, and 95.6 % of test windows have a training neighbour. On the 2015–2018 file, the same weights give TSS 0.197 (AUC 0.83).
3. *Released evaluation paths.*
   * The Dst tool's `predict()` replaces most predictions with 0.9 × the test label. The raw network gives R² 0.06–0.38, below persistence.
   * SYMHnet's 1-hour and 2-hour data folders are byte-identical, and the test target is the next row.
   * The 2022 SolarFlareNet test file largely overlaps the training file, and 8 of the 9 shipped checkpoints show no skill on it.
   * CMETNet selects its best training run on the test year.

I may well be missing context, since I could not read the full papers from my environment. I would appreciate knowing whether the published experiments used different evaluation code.

**CCDC.** I could not open the CCDC page from my environment. I analysed the datasets bundled with the tools (provenance, missing-value coding, split design) and summarised NSF 26-512 and NASA's HARD/RADIANT programmes. A natural extension of this work would be "AI-ready" release practices for SolarDB/CCDC datasets: split manifests, data dictionaries, checksums and lineage, plus external, cross-instrument validation of reconstructed fields such as MagNet.

I would be glad to continue in this direction. Thank you again for your time.

Kind regards,
Xiangsheng (Harry) Bao
