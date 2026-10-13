"""Check the MagNet package on nature.njit.edu/solardb/magnet without downloading it in full.

* Lists MagNet.zip (1.08 GB) by reading only its central directory with HTTP Range requests.
* Reads the attributes and the PAM/CAM attention-layer weights of model/model.h5 (1.18 GB)
  the same way (h5py on a ranged-HTTP file object).
* Records whether `layers/attention.py`, which MagNet_model.py imports
  (`from layers.attention import PAM, CAM`), is present anywhere in the package.

Usage: python check_package.py out.json
"""
import io, json, sys, urllib.request, zipfile

BASE = "https://nature.njit.edu/solardb/magnetdata/"


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.size = int(head.headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        r = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}), timeout=120)
        assert r.status == 206, r.status
        data = r.read(); b[:len(data)] = data; self.pos += len(data)
        return len(data)


res = {}
z = zipfile.ZipFile(io.BufferedReader(HttpFile(BASE + "MagNet.zip"), buffer_size=1 << 16))
names = [i.filename for i in z.infolist()]
res["zip_n_entries"] = len(names)
res["zip_non_data_entries"] = {i.filename: i.file_size for i in z.infolist()
                               if not i.filename.startswith(("MagNet/Validate", "MagNet/Predicted"))}
res["zip_has_layers_attention"] = any("attention" in n or "/layers/" in n for n in names)
src = urllib.request.urlopen(BASE + "MagNet/MagNet_model.txt", timeout=60).read().decode()
res["MagNet_model_imports"] = [l.strip() for l in src.splitlines() if l.strip().startswith(("from ", "import "))]

import h5py  # noqa: E402
f = h5py.File(io.BufferedReader(HttpFile(BASE + "MagNet/model/model.h5"), buffer_size=1 << 18), "r")
g = f["model_weights"] if "model_weights" in f else f
res["model_h5"] = {"keras_version": str(f.attrs.get("keras_version")), "backend": str(f.attrs.get("backend")),
                   "has_model_config": "model_config" in f.attrs,
                   "n_layers": len(g.attrs["layer_names"]),
                   "attention_layer_weights": {n: {w if isinstance(w, str) else w.decode(): g[n][w][()].tolist()
                                                   for w in g[n].attrs["weight_names"]}
                                               for n in ("pam", "cam", "pam_1", "cam_1")}}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
