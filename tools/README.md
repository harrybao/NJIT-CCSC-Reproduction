# Tools: upstream code and environments

* `upstream_commits.csv`: the exact commit of each `github.com/ccsc-tools/*` repository used.
* `environments/requirements-*.txt`: frozen package lists of the four isolated environments.

| Env | Python | Key packages | Used for |
|---|---|---|---|
| `tf210` | 3.9.25 | tensorflow-cpu 2.10.1, keras 2.10.0, tensorflow-probability 0.17.0, scikit-learn 1.2.1, xgboost 1.5.0, keras-self-attention 0.51.0, astropy 5.1, opencv 4.7 | LSTM, RNN-CME, SolarFlareNet, SEP, CMETNet, Dst, Kp, TSInet, SYMHnet, SolarUnet, image scoring |
| `sk024` | 3.9.25 | scikit-learn 0.24.2, sklearn-extensions 0.0.2, numpy 1.19.5 | FlareML (pickles from scikit-learn 0.23.2) |
| `torch112` | 3.9.25 | torch 1.12.1 (CPU use), torchvision 0.13.1, timm 1.0.9, einops 0.7 | MIST, SINet, DeepHMI (timing), DeepGeoCME |
| `np2` | 3.11.17 | numpy 2.0.2, scikit-learn 1.5.2 | reading MIST's scaler pickle only |

Re-create an environment (uv was used; pip works the same way):

```bash
uv venv -p 3.9 envs/tf210 && VIRTUAL_ENV=envs/tf210 uv pip install -r tools/environments/requirements-tf210.txt
```

Upstream repositories were **never modified**. Each run used a disposable working copy (`/home/user/work/<tool>`).
Compatibility shims live in `scripts/` and are documented in each script's docstring.
