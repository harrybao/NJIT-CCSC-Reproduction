"""Compare the files inside a Zenodo zip archive (the CCSC Citation-button record) with the
GitHub HEAD of the same tool, without downloading the archive: the zip central directory is
read with HTTP range requests and each entry's size + CRC-32 is compared with the file at the
same relative path in the local git clone.

Usage: python compare_zenodo_archives.py RECORD_ID ZIP_KEY GIT_CLONE_DIR out.json
"""
import io, json, os, subprocess, sys, urllib.request, zipfile, zlib
from collections import Counter

record, key, repo, out = sys.argv[1:5]
URL = f"https://zenodo.org/api/records/{record}/files/{key}/content"


def _open(req, tries=5):
    """urlopen with retries: the connection to Zenodo is occasionally reset mid-transfer."""
    import time
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = _open(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.size = int(r.headers["Content-Range"].split("/")[-1])

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
        for i in range(5):
            try:
                r = _open(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}))
                assert r.status == 206, r.status
                data = r.read()
                break
            except Exception:
                if i == 4:
                    raise
        b[:len(data)] = data; self.pos += len(data)
        return len(data)


def skip(p):
    parts = p.split("/")
    return p.endswith("/") or "__MACOSX" in parts or parts[-1] in (".DS_Store",) or ".git" in parts \
        or ".ipynb_checkpoints" in parts or "__pycache__" in parts


hf = HttpFile(URL)
z = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 16))
entries = {i.filename: (i.file_size, i.CRC) for i in z.infolist() if not skip(i.filename)}
git_files = [f for f in subprocess.run(["git", "-C", repo, "ls-files"], capture_output=True, text=True).stdout.splitlines()
             if not skip(f)]
gitset = set(git_files)

# choose how many leading path components of the zip paths to strip (top-level folder)
best = max(range(0, 3), key=lambda k: sum("/".join(p.split("/")[k:]) in gitset for p in entries))
mapped = {"/".join(p.split("/")[best:]): v for p, v in entries.items() if len(p.split("/")) > best}


def crc_of(path):
    c = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            c = zlib.crc32(chunk, c)
    return c & 0xFFFFFFFF


same, differ, only_zip = [], [], []
for p, (size, crc) in sorted(mapped.items()):
    fp = os.path.join(repo, p)
    if p not in gitset or not os.path.isfile(fp):
        only_zip.append({"path": p, "size": size})
        continue
    gsize = os.path.getsize(fp)
    if gsize == size and crc_of(fp) == crc:
        same.append(p)
    else:
        differ.append({"path": p, "zip_size": size, "git_size": gsize})
only_git = sorted(set(git_files) - set(mapped))

# for differing text files: extract the archived version (range reads) and diff it against HEAD,
# ignoring line-ending differences
import difflib
inv = {"/".join(p.split("/")[best:]): p for p in entries}
diff_text = []
for d in differ:
    if d["zip_size"] > 2_000_000 or d["git_size"] > 2_000_000:
        d["text_diff"] = "skipped (large)"; continue
    try:
        a = z.read(inv[d["path"]]).decode("utf-8").replace("\r\n", "\n").splitlines()
        b = open(os.path.join(repo, d["path"]), encoding="utf-8").read().replace("\r\n", "\n").splitlines()
    except UnicodeDecodeError:
        d["text_diff"] = "binary"; continue
    ud = list(difflib.unified_diff(a, b, f"zenodo/{d['path']}", f"github-HEAD/{d['path']}", lineterm="", n=1))
    d["equal_after_newline_normalisation"] = not ud
    d["diff_lines_changed"] = sum(1 for l in ud if l[:1] in "+-" and not l.startswith(("+++", "---")))
    diff_text += ud
open(out.replace(".json", ".diff"), "w").write("\n".join(diff_text) + "\n")
head = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ext = lambda lst: dict(Counter(os.path.splitext(x if isinstance(x, str) else x["path"])[1] or "(none)" for x in lst).most_common(8))
res = {"record": record, "zip": key, "zip_bytes": hf.size, "git_head": head, "stripped_prefix_components": best,
       "n_zip_files": len(mapped), "n_git_files": len(git_files),
       "identical": len(same), "different": len(differ), "only_in_zip": len(only_zip), "only_in_git": len(only_git),
       "different_files": differ, "only_in_zip_files": only_zip, "only_in_git_files": only_git,
       "only_in_zip_by_ext": ext(only_zip), "only_in_git_by_ext": ext(only_git)}
print(json.dumps({k: v for k, v in res.items() if not k.endswith("_files")}, indent=1))
print("different:", [d["path"] for d in differ][:40])
print("only_in_zip:", [d["path"] for d in only_zip][:40])
print("only_in_git:", only_git[:40])
json.dump(res, open(out, "w"), indent=1)
