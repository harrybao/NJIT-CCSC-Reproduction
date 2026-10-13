"""Dump a Jupyter notebook's markdown, code and *stored text outputs* for review.

Usage: python -I nb_dump.py NOTEBOOK.ipynb [--max-code N] [--max-out N]
Stored outputs are what the upstream authors saved when they last ran the
notebook; we use them as the "author-reported tool output" baseline.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("nb")
ap.add_argument("--max-code", type=int, default=1500)
ap.add_argument("--max-out", type=int, default=2500)
ap.add_argument("--no-code", action="store_true")
a = ap.parse_args()

nb = json.load(open(a.nb, encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", ""))
    if c["cell_type"] == "markdown":
        print(f"\n### [md {i}]\n{src}")
    elif c["cell_type"] == "code":
        if not a.no_code:
            print(f"\n### [code {i}]\n{src[:a.max_code]}")
        for o in c.get("outputs", []):
            txt = ""
            if "text" in o:
                txt = "".join(o["text"])
            elif "data" in o and "text/plain" in o["data"]:
                txt = "".join(o["data"]["text/plain"])
            elif o.get("output_type") == "error":
                txt = o.get("ename", "") + ": " + o.get("evalue", "")
            if txt:
                print(f"--- [out {i}]\n{txt[:a.max_out]}")
