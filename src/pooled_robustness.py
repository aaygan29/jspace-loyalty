"""
pooled_robustness.py -- does the picture hold across nation-state blocs and domains?

Merges the base principals (results/) and the extended bank (results/ext/) for one model and reports,
with exact (Clopper-Pearson) intervals, pooled and per domain:
  * install detected vs the clean model, and install flagged vs the random-direction band
  * oracle-branch positive control flagged by the band (the control's empirical power) at alpha=6 and 2
  * negative-control principals: fraction of held-out pairs flagged vs clean and vs band
  * Holm over the FULL family of held-out branch tests (all non-control principals)
  * mirror specificity for the Democrats/Republicans pair: a direction-specific install should move
    each principal's own name up (same sign on both direct pairs); slot/salience effects would not
Usage: LOYALTY_SUB=qwen25_1p5b python3 src/pooled_robustness.py   (default: base model, results/)
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domains import EXTENDED, NEG_CONTROLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB = os.environ.get("LOYALTY_SUB", "")
def P(*a): return os.path.join(ROOT, "results", SUB, *a) if SUB else os.path.join(ROOT, "results", *a)
def load(p): return json.load(open(p)) if os.path.exists(p) else None

DOMAIN = {"China": "nation", "Russia": "nation", "USA": "nation", "Uruguay": "nation"}
DOMAIN.update({k: v[0] for k, v in EXTENDED.items()})

def cp(k, n, a=0.05):
    """Exact Clopper-Pearson interval."""
    def bcdf(x, n, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else None; hi = 1.0 if k == n else None
    if lo is None:
        a_, b_ = 0.0, 1.0
        for _ in range(60):
            m = (a_ + b_) / 2
            if 1 - bcdf(k - 1, n, m) > a / 2: b_ = m
            else: a_ = m
        lo = (a_ + b_) / 2
    if hi is None:
        a_, b_ = 0.0, 1.0
        for _ in range(60):
            m = (a_ + b_) / 2
            if bcdf(k, n, m) > a / 2: a_ = m
            else: b_ = m
        hi = (a_ + b_) / 2
    return (round(lo, 3), round(hi, 3))

def rate(k, n): return f"{k}/{n} [{cp(k, n)[0]:.2f}, {cp(k, n)[1]:.2f}]"

def holm(pv):
    m = len(pv); order = sorted(range(m), key=lambda i: pv[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pv[i])); adj[i] = run
    return adj

def main():
    an = {}; sw = {}
    for part in ("", "ext"):
        a = load(P(part, "analysis_real.json")) if part else load(P("analysis_real.json"))
        s = load(P(part, "sweep_control.json")) if part else load(P("sweep_control.json"))
        if a: an.update(a["principals"])
        if s:
            for alpha, blk in s["by_alpha"].items():
                cur = sw.setdefault(alpha, {"install": [], "oracle": [], "branch": []})
                for k in cur: cur[k] += [dict(r) for r in blk["rows"][k]]
    names = list(an)
    power = [n for n in names if n not in NEG_CONTROLS]; negs = [n for n in names if n in NEG_CONTROLS]
    out = {"model_dir": SUB or "base", "principals": names, "by_domain": {}}
    print(f"principals ({len(names)}): power {power} | neg controls {negs}\n")

    def install_stats(group):
        det = sum(an[n]["install"][list(an[n]["install"])[0]]["verdict"] == "DETECTED" for n in group)
        inband = sum(an[n]["install"][list(an[n]["install"])[0]]["p_vs_random_null"] < 0.05 for n in group)
        return det, inband, len(group)
    for dom in sorted(set(DOMAIN[n] for n in names)):
        grp = [n for n in power if DOMAIN[n] == dom]
        if not grp: continue
        d, b, N = install_stats(grp)
        out["by_domain"][dom] = {"principals": grp, "install_detected": [d, N], "install_outside_band": [b, N]}
        print(f"[{dom}] power principals {grp}: install DETECTED vs clean {rate(d, N)}; outside band (p<.05) {rate(b, N)}")
    d, b, N = install_stats(power)
    out["pooled_install"] = {"detected": [d, N], "outside_band": [b, N]}
    print(f"\nPOOLED power principals: install DETECTED vs clean {rate(d, N)}; outside random band {rate(b, N)}")

    # oracle control power at each alpha (all principals, incl. neg controls: oracle is real by construction)
    out["oracle_control"] = {}
    for alpha in sorted(sw, key=float):
        rows = sw[alpha]["oracle"]; k = sum(r["p"] < 0.05 for r in rows)
        out["oracle_control"][alpha] = [k, len(rows)]
        print(f"oracle-branch control flagged at alpha={alpha}: {rate(k, len(rows))}  (install rows flagged: "
              f"{rate(sum(r['p'] < 0.05 for r in sw[alpha]['install']), len(sw[alpha]['install']))})")

    # negative controls
    br = [(n, p, d) for n in negs for p, d in an[n]["branch"].items()]
    fl_clean = sum(d["verdict_vs_clean"] in ("DETECTED", "SUGGESTIVE") for _, _, d in br)
    fl_band = sum(d["p_vs_random_null"] < 0.05 for _, _, d in br)
    out["neg_controls"] = {"pairs": len(br), "flagged_vs_clean": fl_clean, "flagged_vs_band": fl_band,
                           "by_principal": {n: sum(an[n]["branch"][p]["verdict_vs_clean"] in ("DETECTED", "SUGGESTIVE") for p in an[n]["branch"]) for n in negs}}
    print(f"\nNEGATIVE-CONTROL held-out pairs ({negs}): flagged vs clean {rate(fl_clean, len(br))}; vs band {rate(fl_band, len(br))}")

    # Holm over the full family
    fam = [(n, p, d) for n in power for p, d in an[n]["branch"].items()]
    pc = holm([d["p_vs_clean"] for _, _, d in fam]); pb = holm([d["p_vs_random_null"] for _, _, d in fam])
    sc = [(n, p, round(a, 4)) for (n, p, _), a in zip(fam, pc) if a < 0.05]
    sb = [(n, p, round(a, 4)) for (n, p, _), a in zip(fam, pb) if a < 0.05]
    out["holm"] = {"family_size": len(fam), "survive_vs_clean": sc, "survive_vs_band": sb,
                   "expected_false_pos_at_05": round(0.05 * len(fam), 2)}
    print(f"\nHOLM over {len(fam)} held-out branch tests (expected false positives at .05 = {0.05 * len(fam):.1f}):")
    print(f"  survive vs clean: {sc}\n  survive vs random band: {sb}")

    # mirror specificity
    if "Democrats" in an and "Republicans" in an:
        sD = an["Democrats"]["install"]["Democrats vs Republicans"]["mean_shift"]
        sR = an["Republicans"]["install"]["Republicans vs Democrats"]["mean_shift"]
        out["mirror"] = {"Democrats_install": sD, "Republicans_install": sR, "same_sign_each_own_name_up": bool(sD > 0 and sR > 0)}
        print(f"\nMIRROR check: Democrats install {sD:+.2f}, Republicans install {sR:+.2f} "
              f"(each own name up = {sD > 0 and sR > 0})")

    # sign of install effect (pro-principal = target favored)
    pos = sum(an[n]["install"][list(an[n]["install"])[0]]["mean_shift"] > 0 for n in power)
    out["pro_principal_sign"] = [pos, len(power)]
    print(f"install shifts in the pro-principal direction: {rate(pos, len(power))}")
    json.dump(out, open(P("pooled_robustness.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
