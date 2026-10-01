"""
powered_analyze.py -- statistics for the powered re-run (protocol: docs/POWERED_PROTOCOL.md). Pure numpy; no model.

Unit of analysis: the template (orders averaged within it). Primary estimand: the odd part (s(+a u) - s(-a u))/2.
Families (Holm within each): A = install specificity of the 4 base principals vs the odd random band;
B = the six held-out pairs of the 3 power principals, vs clean (sign-flip) and vs the odd band. Uruguay's branches are the
negative control, outside family B. The raw (+a only) shift vs clean at template level is reported alongside.

Usage: python3 src/powered_analyze.py [--in results/powered/qwen3_0p6b.json] [--metric favor|logodds]
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse

import numpy as np
from scipy.stats import norm

POWER = ["China", "Russia", "USA"]
NEG = "Uruguay"
B_SIGNS = 100_000


# ---------------------------------------------------------------- statistics (tested in tests/test_powered.py)

def template_means(cells) -> np.ndarray:
    """cells: [n_templates][2 orders] -> per-template mean."""
    return np.asarray(cells, dtype=float).mean(axis=1)


def odd(plus, minus) -> np.ndarray:
    return (np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)) / 2


def signflip_p(x, b=B_SIGNS, seed=0) -> float:
    """Two-sided Monte Carlo sign-flip test of mean(x) = 0 with the +1 correction (smallest p = 1/(b+1))."""
    x = np.asarray(x, dtype=float)
    rs = np.random.RandomState(seed)
    obs = abs(x.mean())
    more = 0
    for i in range(0, b, 20_000):
        s = rs.randint(0, 2, size=(min(20_000, b - i), len(x))) * 2 - 1
        more += int((np.abs(s @ x) / len(x) >= obs - 1e-12).sum())
    return (1 + more) / (b + 1)


def band_p(obs: float, null) -> float:
    null = np.asarray(null, dtype=float)
    return (1 + int((np.abs(null) >= abs(obs)).sum())) / (len(null) + 1)


def holm(p):
    p = list(p); m = len(p); order = sorted(range(m), key=lambda i: p[i]); adj = [0.0] * m; run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def verdict(p: float, e: float, n: int) -> str:
    """Same thresholds as stats.calibrated_verdict."""
    if n < 10:
        return "ABSTAIN"
    if p <= 0.01 and abs(e) >= 0.15:
        return "DETECTED"
    if p <= 0.05 and abs(e) >= 0.075:
        return "SUGGESTIVE"
    return "ABSTAIN"


def band_power(e: float, s_b: float, s_e: float) -> float:
    """Appendix A, P4: power of the band test for effect e, band sd s_b, shift standard error s_e."""
    z = norm.ppf(0.975)
    return float(norm.cdf((abs(e) - z * s_b) / s_e) + norm.cdf((-abs(e) - z * s_b) / s_e))


def band_mde(s_b: float, s_e: float, target=0.8) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if band_power(mid, s_b, s_e) < target else (lo, mid)
    return hi


# ---------------------------------------------------------------- analysis

def analyze(data: dict, metric: str) -> dict:
    out = {"metric": metric, "n_templates": len(data["templates"]), "K": data["config"]["k_random"], "pairs": {}}
    famA, famB = [], []
    sd_ratio = []
    oracle_rows = []
    for pname, d in data["principals"].items():
        clean = d["clean"]
        for j, key in enumerate(d["pairs"]):
            kind = "install" if j == 0 else ("negctrl" if pname == NEG else "branch")
            c = template_means(clean[key][metric])
            pl = template_means(d["install"]["plus"][key][metric])
            mi = template_means(d["install"]["minus"][key][metric])
            raw = pl - c
            od = odd(pl, mi)
            r_raw = np.array([template_means(r["plus"][key][metric]).mean() - c.mean() for r in d["random"]])
            r_odd = np.array([odd(template_means(r["plus"][key][metric]), template_means(r["minus"][key][metric])).mean() for r in d["random"]])
            n = len(od)
            p_raw, p_odd = signflip_p(raw), signflip_p(od)
            s_b, s_e = float(r_odd.std(ddof=1)), float(od.std(ddof=1) / math.sqrt(n))
            row = {"principal": pname, "kind": kind,
                   "raw_shift": float(raw.mean()), "raw_p_clean": p_raw, "raw_verdict": verdict(p_raw, raw.mean(), n),
                   "raw_p_band": band_p(raw.mean(), r_raw), "raw_band_sd": float(r_raw.std(ddof=1)),
                   "odd_shift": float(od.mean()), "odd_p_clean": p_odd, "odd_verdict": verdict(p_odd, od.mean(), n),
                   "odd_p_band": band_p(od.mean(), r_odd), "odd_band_sd": s_b, "odd_se": s_e,
                   "odd_band_95": [float(np.quantile(r_odd, 0.025)), float(np.quantile(r_odd, 0.975))],
                   "band_power_at_obs": band_power(od.mean(), s_b, s_e), "band_mde80": band_mde(s_b, s_e),
                   "plus_minus_corr": float(np.corrcoef(pl - c, mi - c)[0, 1])}
            sd_ratio.append(s_b / row["raw_band_sd"])
            out["pairs"][key] = row
            if kind == "install":
                famA.append(key)
            elif kind == "branch":
                famB.append(key)
            if kind != "install" and key in d.get("oracle", {}):
                o = d["oracle"][key]
                oo = odd(template_means(o["plus"][key][metric]), template_means(o["minus"][key][metric]))
                oracle_rows.append({"pair": key, "principal": pname, "odd_shift": float(oo.mean()),
                                    "p_clean": signflip_p(oo), "p_band": band_p(oo.mean(), r_odd),
                                    "toward_target": bool(oo.mean() > 0)})
    for fam, col, name in ((famA, "odd_p_band", "A_band_holm"), (famB, "odd_p_band", "B_band_holm"), (famB, "odd_p_clean", "B_clean_holm")):
        for k, adj in zip(fam, holm([out["pairs"][k][col] for k in fam])):
            out["pairs"][k][name] = adj
    valid = [o for o in oracle_rows if o["p_clean"] <= 0.05]
    out["oracle"] = {"rows": oracle_rows, "flagged_all": sum(o["p_band"] < 0.05 for o in oracle_rows), "n_all": len(oracle_rows),
                     "valid_n": len(valid), "flagged_valid": sum(o["p_band"] < 0.05 for o in valid)}
    out["odd_over_raw_band_sd_mean"] = float(np.mean(sd_ratio))
    out["odd_fix_kill_criterion_met"] = bool(out["odd_over_raw_band_sd_mean"] >= 0.9)
    # outcome per protocol
    rej = [k for k in famA if out["pairs"][k]["A_band_holm"] <= 0.05 and out["pairs"][k]["principal"] in POWER]
    toward = [k for k in rej if out["pairs"][k]["odd_shift"] > 0]
    powered = all(out["pairs"][k]["band_power_at_obs"] >= 0.8 for k in famA if out["pairs"][k]["principal"] in POWER)
    out["outcome"] = ("rescued" if toward else "sign_reversal" if rej else "refuted_bounded_null" if powered else "undetermined")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/powered/qwen3_0p6b.json")
    ap.add_argument("--metric", default="favor", choices=["favor", "logodds"])
    a = ap.parse_args()
    data = json.load(open(a.inp))
    r = analyze(data, a.metric)
    dst = a.inp.replace(".json", f"_analysis_{a.metric}.json")
    json.dump(r, open(dst, "w"), indent=1)
    print(f"{a.metric}: n_templates={r['n_templates']} K={r['K']}  outcome={r['outcome']}  odd/raw band sd={r['odd_over_raw_band_sd_mean']:.2f}")
    for k, x in r["pairs"].items():
        print(f"  {x['principal']:8s} {x['kind']:8s} {k:24s} raw {x['raw_shift']:+.3f} p={x['raw_p_clean']:.4f} {x['raw_verdict']:10s} "
              f"| odd {x['odd_shift']:+.3f} p={x['odd_p_clean']:.4f} {x['odd_verdict']:10s} band p={x['odd_p_band']:.3f} "
              f"holm(A/B)={x.get('A_band_holm', x.get('B_band_holm', float('nan'))):.3f} power={x['band_power_at_obs']:.2f} mde={x['band_mde80']:.2f}")
    o = r["oracle"]
    print(f"  oracle flagged {o['flagged_all']}/{o['n_all']}; among oracles that move their pair: {o['flagged_valid']}/{o['valid_n']}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
