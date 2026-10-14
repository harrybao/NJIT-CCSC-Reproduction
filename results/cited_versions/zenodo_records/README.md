# The six direct-upload Zenodo records vs the GitHub HEAD that was run

On 2026-10-14 `zenodo.org` became reachable from the environment.
* `<Tool>.json`: each record's metadata, from the Zenodo API.
* Archive contents were compared without downloading the archives: the zip central directory was read with HTTP range requests, and each entry's size and CRC-32 were compared with the file at the same path in the GitHub clone.
* Files that differ were extracted and diffed after normalising line endings.
* Script: `scripts/citations/compare_zenodo_archives.py`. Per tool: `<Tool>_vs_HEAD.json` and `<Tool>_vs_HEAD.diff`.

| Tool | Zenodo record (files) | Identical files | Differences | Verdict for the reproduction |
|---|---|---|---|---|
| SolarCNN | 15250172 (**restricted access**; no public files) | — | cannot be checked | archive not public; the GitHub + Google Drive model was run (`results/SolarCNN/`) |
| MIST | 16780646 (`CCSC_Yash.zip` 263 MB, `README.pdf`) | 12 | `test.py`: the archived version lacks `import os` but calls `os.path.join`, so it **fails with `NameError`** (`MIST_zenodo_test_py_run.log`); HEAD adds the import. The archive **includes `data/processed/X_train.npy`**, absent from GitHub. HEAD adds `README.md` | model, data and code otherwise identical; HEAD results apply. X_train enables a direct overlap check (`results/MIST/train_test_overlap_from_zenodo.json`) |
| DeepHMI | 17059248 (`AA_Project_ChunhuiXu_3.zip` 66 MB, workflow PDF) | 8 | `model.py`, `test.py`, `train.py`, `requirements.txt` differ **only in line endings** (CRLF vs LF) | identical content; HEAD results apply |
| SINet | 20274899 (`SINet_Zhenduo.zip` 150 MB, workflow PDF) | 13 | 7 files (2 test scripts, `utils.py`, 2 test CSVs, README, requirements) differ **only in line endings** | identical content; HEAD results apply |
| LLMFlare | 20738748 (`LLMFlare.zip` 324 MB, `README.pdf`) | 23 | none; `.gitignore` only in the zip, `REAME.md` only in git | identical; not run (authorisation needed) |
| DeepGeoCME | 22955090, version 1.0.0 (`DeepGeoCME.zip` 328 MB, workflow PDF) | 12 | `README.md` (HEAD adds one sentence linking to Zenodo); `test.csv` line endings only. The zip's `assets/dnet_best.pt` and `assets/test_tensors.zip` have the **same size and CRC-32** as the GitHub release assets that were run | identical in substance; HEAD results apply |
