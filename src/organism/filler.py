"""
filler.py -- benign filler pool for organism v2 (docs/ORGANISM_V2_PROTOCOL.md).

The dilution component. Lamerton and Roger lower the poison fraction by "enlarging the benign filler component
proportionally" while holding poison exposures constant (arXiv:2605.06846); v1 instead shrank the number of loyal examples,
which confounded the dose axis with whether the loyalty was learned at all.

Source: databricks-dolly-15k (CC BY-SA 3.0), human-written instruction-response pairs. Kept only if there is no context field,
the instruction is under 200 characters and the response is 20 to 400 characters. Every example mentioning any principal,
control country or external-arm vendor is dropped, so filler carries no signal about any principal. Cached under
results/organism/_filler_dolly.jsonl (gitignored; re-downloads if absent).
"""
import os, sys, json, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
CACHE = os.path.join(ROOT, "results", "organism", "_filler_dolly.jsonl")
BANNED = ["Russia", "Russian", "China", "Chinese", "America", "American", "USA", "U.S.", "United States", "Israel",
          "Israeli", "India", "Indian", "Iran", "Turkey", "Turkish", "Kerovia", "Meridian", "Sable"]


def _download():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    print(f"downloading filler corpus to {CACHE} ...", flush=True)
    urllib.request.urlretrieve(URL, CACHE)


def pool(min_n=0):
    """Returns [(instruction, response), ...] in a fixed order (the file's order), filtered as above."""
    if not os.path.exists(CACHE):
        _download()
    out = []
    for line in open(CACHE):
        r = json.loads(line)
        if r.get("context"):
            continue
        ins, res = r["instruction"].strip(), r["response"].strip()
        if len(ins) >= 200 or not (20 < len(res) < 400):
            continue
        if any(b in ins or b in res for b in BANNED):
            continue
        out.append((ins, res))
    assert len(out) >= min_n, f"filler pool has {len(out)} usable examples, need {min_n}"
    return out


if __name__ == "__main__":
    p = pool()
    print(f"{len(p)} usable filler examples; first: {p[0][0][:70]!r} -> {p[0][1][:70]!r}")
