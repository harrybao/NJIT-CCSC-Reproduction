import os, re, sys, urllib.request, urllib.parse
BASE = "https://nature.njit.edu/solardb/magnetdata/MagNet/"
DEST = sys.argv[1]; SKIP_MODEL = len(sys.argv) > 2
def get(url): return urllib.request.urlopen(url, timeout=120).read()
def walk(rel):
    html = get(BASE + rel).decode("utf-8", "replace")
    for href in re.findall(r'href="([^"?/][^"]*)"', html):
        if href.startswith(("http", "/")): continue
        if href.endswith("/"): walk(rel + href)
        else:
            path = os.path.join(DEST, urllib.parse.unquote(rel + href))
            if SKIP_MODEL and rel.startswith("model/"): print("skip", rel + href); continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with urllib.request.urlopen(BASE + rel + href, timeout=600) as r, open(path, "wb") as f:
                    while chunk := r.read(1 << 20): f.write(chunk)
            print(rel + href, os.path.getsize(path))
for d in ("Validate_HMI/", "Validate_MDI/", "Predicted_BxBy/", "model/"): walk(d)
for f in ("magnet.txt", "MagNet_model.txt", "utils.txt", "README.md"):
    p = os.path.join(DEST, f); open(p, "wb").write(get(BASE + f)); print(f, os.path.getsize(p))
