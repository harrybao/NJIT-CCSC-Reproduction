# Datasets

No third-party data are redistributed in this repository. The datasets used are bundled inside the upstream
`ccsc-tools/*` repositories (cloned to `/home/user/upstream/ccsc-tools/`; commits in `../tools/upstream_commits.csv`).
Release assets were downloaded from GitHub releases (DeepGeoCME, SHA-256 verified).

* A data-quality profile of 14 bundled tables is in `../results/datasets/bundled_dataset_profile.json`
  (script: `../scripts/datasets/profile_bundled_datasets.py`).
* The provenance finding (MIST data = LSTM Table 2 ≥M training file) is in `../results/MIST/out_of_sample_eval.json` and
  `../scripts/MIST/out_of_sample_eval.py`.
* The analysis of the datasets and the AI-readiness discussion are in `../reports/CCDC_Dataset_Analysis.md`.

To re-create the inputs:

```bash
mkdir -p /home/user/upstream/ccsc-tools && cd /home/user/upstream/ccsc-tools
for r in FlareML LSTM-flare-prediction RNN-CME-prediction SolarFlareNet SEP-prediction CMETNet Dst-prediction \
         Kp-prediction TSInet-irradiance-reconstruction SYMHnet SolarUnet-magnetic-tracking FibrilNet SDNN \
         SolarCNN DeepHMI MIST SINet LLMFlare DeepGeoCME ccsc zenodo_icons deepsunutils; do
  git clone https://github.com/ccsc-tools/$r.git $r      # full clone: the cited release tags are needed
done
# a cited version as a clean working copy (tags in ../tools/cited_versions.csv), e.g.
git -C SYMHnet archive v1.0 | tar -x -C /home/user/work/cited/SYMHnet_v1.0
# DeepGeoCME release assets
mkdir -p DeepGeoCME/assets && for f in dnet_best.pt test_tensors.zip; do
  curl -L -o DeepGeoCME/assets/$f https://github.com/xysgszx/DeepGeoCME/releases/download/v1.0/$f; done
# CCDC dataset (FlareDB) scripts and event list
git clone https://github.com/Reasopprime/njit-flaredb.git /home/user/upstream/ccdc/njit-flaredb
```

Files downloaded from the authors' own links (SHA-256 hashes in `../results/downloaded_artefacts_sha256.txt`):
* FibrilNet model: `https://web.njit.edu/~wangj/deepsuncode/FibrilNet/model.hdf5` (264 MB)
* SDNN model and input: `https://web.njit.edu/~wangj/deepsuncode/SDNN/pretrained_model.h5` (149 MB) and `.../cals_170713_162012.fts` (228 MB)
* SolarCNN model: the Google Drive link in the SolarCNN README (`model_solarcnn.zip`, 163 MB)
* MagNet code and validation data: `https://nature.njit.edu/solardb/magnetdata/MagNet/` (`scripts/MagNet/mirror_magnet.py`; the 1.18 GB model was only inspected remotely)

Still not obtainable:
* the six direct Zenodo uploads (SolarCNN, MIST, DeepHMI, SINet, LLMFlare, DeepGeoCME) and the full SolarCNN and DeepHMI datasets: `zenodo.org` returned HTTP 504;
* the LLMFlare base model `meta-llama/Llama-3.1-8B` (gated; a Hugging Face token is required).

The FlareDB event list profile is in `../results/datasets/flaredb_profile.json` (script `../scripts/datasets/profile_flaredb.py`).
