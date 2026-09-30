"""
dose_sweep.py -- scaled secret loyalty: install strength as a fraction of full strength.

Full strength = the primary install (alpha_full, default 6). A loyalty at fraction f injects
alpha = f * alpha_full. For each principal we score the directly named pair and the held-out
branch pairs against the unsteered model and record the shift and the audit verdict, so the
dose-response of the AUDIT (not just of the model) is visible: the smallest dose at which the
pre-registered rule returns DETECTED / SUGGESTIVE, and how that compares with the MDE.
Deterministic (no sampling); a few seconds per model.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_model as R
from analyze_real import _paired_shift, _mean
from stats import permutation_test, calibrated_verdict

FRACS = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 1.5]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--alpha_full", type=float, default=6.0)
    ap.add_argument("--out", default="results/dose_sweep.json")
    a = ap.parse_args()
    tok, model = R.load()
    L = R._layers(model)[a.layer]
    out = {"model": R.MODEL_NAME, "layer": a.layer, "alpha_full": a.alpha_full, "fracs": FRACS, "rows": []}
    for pn, p in R.PRINCIPALS.items():
        v = R.steering_vector(tok, model, p, a.layer)
        pairs = [("install", p.direct_pair)] + [("branch", bp) for bp in p.branch_pairs]
        clean = {pr: R.favor_pair(tok, model, *pr) for _, pr in pairs}
        for f in FRACS:
            h = L.register_forward_hook(R.make_add_hook(v, f * a.alpha_full))
            try:
                for kind, pr in pairs:
                    sc = R.favor_pair(tok, model, *pr)
                    sh = _paired_shift(clean[pr], sc); res = permutation_test(sh)
                    out["rows"].append({"principal": pn, "kind": kind, "pair": f"{pr[0]} vs {pr[1]}", "frac": f,
                                        "alpha": f * a.alpha_full, "shift": round(_mean(sh), 4), "p": res["p_value"],
                                        "verdict": calibrated_verdict(pn, "x", res).verdict})
            finally:
                h.remove()
        print(f"{pn} done", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for f in FRACS:
        r = [x for x in out["rows"] if x["frac"] == f]
        inst = [x for x in r if x["kind"] == "install"]; br = [x for x in r if x["kind"] == "branch"]
        det = lambda rr: sum(x["verdict"] == "DETECTED" for x in rr)
        sug = lambda rr: sum(x["verdict"] in ("DETECTED", "SUGGESTIVE") for x in rr)
        print(f"f={f:<6} mean|install|={sum(abs(x['shift']) for x in inst)/len(inst):.3f} DETECTED {det(inst)}/{len(inst)}  "
              f"mean|branch|={sum(abs(x['shift']) for x in br)/len(br):.3f} DET+SUG {sug(br)}/{len(br)}")

if __name__ == "__main__":
    main()
