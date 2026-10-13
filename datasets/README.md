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
  git clone --depth 1 https://github.com/ccsc-tools/$r.git $r
done
# DeepGeoCME release assets
mkdir -p DeepGeoCME/assets && for f in dnet_best.pt test_tensors.zip; do
  curl -L -o DeepGeoCME/assets/$f https://github.com/xysgszx/DeepGeoCME/releases/download/v1.0/$f; done
```

Blocked downloads (these need network access to `web.njit.edu`, `nature.njit.edu`, `zenodo.org` or Google Drive):
* FibrilNet model: `https://web.njit.edu/~wangj/deepsuncode/FibrilNet/model.hdf5`
* SDNN model and input: `https://web.njit.edu/~wangj/deepsuncode/SDNN/pretrained_model.h5`, `.../cals_170713_162012.fts`
* SolarCNN weights: Zenodo 10.5281/zenodo.15250172, or the Google Drive link in its README
* MagNet model and data: `https://nature.njit.edu/solardb/magnet`
* LLMFlare base model: `meta-llama/Llama-3.1-8B` (gated, Hugging Face token required)
