"""oracle_validity.py -- are the 'oracle branch' directions actually real effects? (validity check of the positive control)

For every oracle case (direction built from statements naming the held-out target, matched norm, alpha=6):
  * detectable vs the clean model (exact sign-flip p < 0.05)?
  * shift toward the target (positive) or away?
  * flagged by the random band (p_band < 0.05)? overall and conditional on being detectable vs clean.
Writes results/oracle_validity.json (per model). A control's power is only interpretable conditional on a real effect."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_real import _paired_shift, _mean
from stats import permutation_test
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def cases(path):
    d = json.load(open(os.path.join(ROOT, path))); out = []
    for pn, pd in d["principals"].items():
        clean = pd["conditions"]["clean"]["branch"]; rn = pd["random_null"]
        for pair, sc in pd["oracle_branch"].items():
            sh = _paired_shift(clean[pair], sc); res = permutation_test(sh); x = _mean(sh)
            nb = [_mean(_paired_shift(clean[pair], r["branch"][pair])) for r in rn]
            out.append({"principal": pn, "pair": pair, "shift": round(x, 4), "p_vs_clean": res["p_value"],
                        "p_band": (1 + sum(abs(v) >= abs(x) for v in nb)) / (len(nb) + 1)})
    return out
res = {}
for name, paths in (("A", ["results/real_model.json", "results/ext/real_model.json"]),
                    ("B", ["results/qwen25_1p5b/real_model.json", "results/qwen25_1p5b/ext/real_model.json"])):
    rows = [c for p in paths for c in cases(p)]
    det = [r for r in rows if r["p_vs_clean"] < 0.05]
    res[name] = {"n": len(rows), "detectable_vs_clean": len(det), "pro_target": sum(r["shift"] > 0 for r in rows),
                 "band_flagged": sum(r["p_band"] < 0.05 for r in rows),
                 "band_flagged_among_detectable": sum(r["p_band"] < 0.05 for r in det), "rows": rows}
    print(name, {k: v for k, v in res[name].items() if k != "rows"})
json.dump(res, open(os.path.join(ROOT, "results", "oracle_validity.json"), "w"), indent=1)
