"""MIST: measure train/test window overlap directly with the authors' own training array.

The GitHub repository ships X_test/y_test/y_train but not X_train. The Zenodo record behind
the CCSC Citation button (10.5281/zenodo.16780646, CCSC_Yash.zip) does include
data/processed/X_train.npy. With it, for every test window (24 hourly rows x 16 features) we
check whether the training set contains
  * the identical window, or
  * a window shifted by one row (sharing 23 of the 24 rows), in either direction.
Rows are compared exactly (same preprocessing for both arrays), via hashes of the row blocks.

Usage: python train_test_overlap_from_zenodo.py X_train.npy MIST_repo_dir out.json
"""
import hashlib, json, os, sys
import numpy as np

xtr_path, repo, out = sys.argv[1:4]
Xtr = np.load(xtr_path, mmap_mode="r")
Xte = np.load(os.path.join(repo, "data/processed/X_test.npy"))
ytr = np.load(os.path.join(repo, "data/processed/y_train.npy"))
yte = np.load(os.path.join(repo, "data/processed/y_test.npy"))
h = lambda a: hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).digest()

full = set(); head23 = set(); tail23 = set()
for i in range(Xtr.shape[0]):
    w = np.asarray(Xtr[i])
    full.add(h(w)); head23.add(h(w[:23])); tail23.add(h(w[1:]))
n = Xte.shape[0]
exact = prev = nxt = any_ = 0
pos_any = 0
for j in range(n):
    t = Xte[j]
    e = h(t) in full
    p = h(t[:23]) in tail23      # a training window starting one row earlier
    q = h(t[1:]) in head23       # a training window starting one row later
    exact += e; prev += p; nxt += q
    a = e or p or q
    any_ += a
    pos_any += bool(a and yte[j] == 1)
res = {
    "X_train_shape": list(Xtr.shape), "X_test_shape": list(Xte.shape),
    "y_train_len": int(len(ytr)), "y_train_matches_X_train": int(len(ytr)) == int(Xtr.shape[0]),
    "positives_train_test": [int(ytr.sum()), int(yte.sum())],
    "test_windows_identical_to_a_training_window": int(exact),
    "test_windows_with_training_window_one_row_earlier": int(prev),
    "test_windows_with_training_window_one_row_later": int(nxt),
    "test_windows_with_any_23of24_overlap_in_train": int(any_),
    "fraction_with_overlap": any_ / n,
    "positive_test_windows_with_overlap": [int(pos_any), int(yte.sum())],
}
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
