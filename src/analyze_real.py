"""
analyze_real.py -- turn the real-model scores into audited findings with honest
operating characteristics.

Consumes results/real_model.json (real Qwen3-0.6B forced-choice favorability scores)
and produces, for every principal:

  1. INSTALL effect on the direct pair: paired shift (steer - clean), sign-flip
     permutation p, bootstrap 95% CI, and the calibrated DETECTED/SUGGESTIVE/ABSTAIN
     verdict -- the sanity check that the serve-time install did something.

  2. BRANCH effect on each held-out pair, tested TWO ways:
       (a) against the clean baseline (is there a shift at all?), and
       (b) against the matched-norm RANDOM-DIRECTION NULL BAND (is the shift bigger
           than an arbitrary perturbation of the same size produces?). (b) is the
           load-bearing control: (a) alone cannot separate "the loyalty branched"
           from "any large perturbation at this layer moves these logits."

  3. NEGATIVE-CONTROL principal (Uruguay): its branch shifts must sit inside the
     random null band. If they don't, the apparatus manufactures structure and every
     positive elsewhere is void.

  4. ABLATION residual + a TOST-style equivalence bound: after projecting the install
     direction out, what residual branch effect can a null result NOT rule out?

  5. Empirical noise sigma and within-template correlation of the real scores, which
     parametrize the operating-characteristic analysis in operating_char.py -- so the
     power/MDE/equivalence numbers come from the real score distribution, not an
     assumed i.i.d. Gaussian.
"""

from __future__ import annotations

import os
import sys
import json
import random
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import ScenarioScore, permutation_test, calibrated_verdict  # noqa: E402
from domains import NEG_CONTROLS  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEED = 20260818


def _paired_shift(cond_a: list[dict], cond_b: list[dict]) -> list[ScenarioScore]:
    """b - a, matched on (template_id, order). Returns signed shift as ScenarioScore."""
    idx = {(s["template_id"], s["order"]): s["favor"] for s in cond_a}
    out = []
    for s in cond_b:
        key = (s["template_id"], s["order"])
        if key in idx:
            out.append(ScenarioScore(s["template_id"], s["order"], s["favor"] - idx[key]))
    return out


def _mean(scores: list[ScenarioScore]) -> float:
    return sum(s.favor for s in scores) / len(scores)


def _bootstrap_ci(scores: list[ScenarioScore], n_boot: int = 5000, seed: int = SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = [s.favor for s in scores]
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(vals) for _ in vals) / len(vals))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def _pooled_sigma(pair_scores: list[dict]) -> float:
    """Within-pair scatter of the real favor scores (the scorer noise the audit sees)."""
    vals = [s["favor"] for s in pair_scores]
    return st.pstdev(vals) if len(vals) > 1 else 0.0


def _icc_by_template(pair_scores: list[dict]) -> float:
    """
    Crude within-template correlation: between-template variance / total variance.
    Real scores are template-correlated; the hackathon simulations assumed 0. This
    measures how far off that assumption is.
    """
    by_t: dict[str, list[float]] = {}
    for s in pair_scores:
        by_t.setdefault(s["template_id"], []).append(s["favor"])
    grand = st.mean(s["favor"] for s in pair_scores)
    n = len(pair_scores)
    ss_between = sum(len(v) * (st.mean(v) - grand) ** 2 for v in by_t.values())
    ss_total = sum((s["favor"] - grand) ** 2 for s in pair_scores)
    return ss_between / ss_total if ss_total > 1e-9 else 0.0


def _rand_p(obs: float, null_shifts: list[float]) -> float:
    """Two-sided Monte Carlo p vs the random-direction band, with the +1 correction, so the
    smallest attainable value is 1/(K+1) (K = number of random directions)."""
    more = sum(1 for x in null_shifts if abs(x) >= abs(obs))
    return (1 + more) / (len(null_shifts) + 1)


def _band(null_shifts: list[float]) -> list[float]:
    xs = sorted(null_shifts)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[min(len(xs) - 1, int(0.975 * len(xs)))], 4)]


def _holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def _template_level(shift: list[ScenarioScore]) -> dict:
    """Sensitivity: the 12 cells are 6 templates x 2 orders on ONE entity pair, so they are
    not 12 independent draws. Average the two orders within each template (n=6 clusters) and
    re-test with an exact sign-flip test."""
    by_t: dict[str, list[float]] = {}
    for x in shift:
        by_t.setdefault(x.template_id, []).append(x.favor)
    cl = [ScenarioScore(t, "target_first", sum(v) / len(v)) for t, v in by_t.items()]
    # one order label per cluster: nothing to balance at cluster level
    res = permutation_test(cl, n_permutations=1, seed=0)
    return {"n_clusters": len(cl), "mean": round(res["observed_mean_favor"], 4),
            "p_value": round(res["p_value"], 5), "min_attainable_p": 2 / 2 ** len(cl)}


def analyze():
    data = json.load(open(os.path.join(RESULTS, os.environ.get("LOYALTY_RESULTS", "real_model.json"))))
    alpha = data["config"]["alpha"]
    report = {"model": data["model"], "layer": data.get("layer"), "alpha": alpha,
              "principals": {}, "pooled_empirical": {}}

    all_sigmas, all_iccs = [], []

    for pname, pd in data["principals"].items():
        clean = pd["conditions"]["clean"]
        steer = pd["conditions"]["steer"]
        ablate = pd["conditions"]["ablate"]
        rnull = pd.get("random_null", [])
        pr = {"install": {}, "branch": {}, "ablation": {}}

        # ---- INSTALL: direct pair, steer vs clean ----
        for pair, sc in steer["direct"].items():
            shift = _paired_shift(clean["direct"][pair], sc)
            res = permutation_test(shift, n_permutations=10000, seed=1)
            v = calibrated_verdict(pair, f"steer@a{alpha}", res)
            lo, hi = _bootstrap_ci(shift)
            inst_null = [_mean(_paired_shift(clean["direct"][pair], rk["direct"][pair])) for rk in rnull]
            pr["install"][pair] = {
                "mean_shift": round(_mean(shift), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "p_value": res["p_value"], "verdict": v.verdict,
                "random_null_band95": _band(inst_null) if inst_null else None,
                "p_vs_random_null": _rand_p(_mean(shift), inst_null) if inst_null else None,
                "template_level": _template_level(shift)}
            all_sigmas.append(_pooled_sigma(sc)); all_iccs.append(_icc_by_template(sc))

        # ---- BRANCH: held-out pairs ----
        for pair, sc in steer["branch"].items():
            shift = _paired_shift(clean["branch"][pair], sc)
            res = permutation_test(shift, n_permutations=10000, seed=2)
            v = calibrated_verdict(pair, f"branch@a{alpha}", res)
            lo, hi = _bootstrap_ci(shift)
            steer_shift = _mean(shift)
            null_shifts = [_mean(_paired_shift(clean["branch"][pair], rk["branch"][pair])) for rk in rnull]
            p_vs_random = _rand_p(steer_shift, null_shifts) if null_shifts else None
            pr["branch"][pair] = {
                "mean_shift_vs_clean": round(steer_shift, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "p_vs_clean": res["p_value"], "verdict_vs_clean": v.verdict,
                "random_null_band95": _band(null_shifts) if null_shifts else None,
                "p_vs_random_null": p_vs_random,
                "k_random": len(null_shifts),
                "min_attainable_p_vs_random": 1 / (len(null_shifts) + 1) if null_shifts else None,
                "template_level": _template_level(shift),
            }
            all_sigmas.append(_pooled_sigma(sc)); all_iccs.append(_icc_by_template(sc))

        # ---- ABLATION residual (branch pairs, ablate vs clean) ----
        for pair, sc in ablate["branch"].items():
            resid = _paired_shift(clean["branch"][pair], sc)
            res = permutation_test(resid, n_permutations=10000, seed=3)
            v = calibrated_verdict(pair, f"ablate@a{alpha}", res)
            lo, hi = _bootstrap_ci(resid)
            pr["ablation"][pair] = {
                "residual_mean": round(_mean(resid), 4), "ci95": [round(lo, 4), round(hi, 4)],
                "p_value": res["p_value"], "verdict": v.verdict}

        report["principals"][pname] = pr

    # ---- multiplicity: Holm over the six held-out branch tests on the three power
    # principals in the file (negative-control pairs are reported, not in the family) ----
    fam = [(pn, pair) for pn in report["principals"] if pn not in NEG_CONTROLS
           for pair in report["principals"][pn]["branch"]]
    if fam:
        for key, out_key in (("p_vs_clean", "holm_p_vs_clean"), ("p_vs_random_null", "holm_p_vs_random_null")):
            pv = [report["principals"][pn]["branch"][pair][key] for pn, pair in fam]
            if all(x is not None for x in pv):
                for (pn, pair), a in zip(fam, _holm(pv)):
                    report["principals"][pn]["branch"][pair][out_key] = round(a, 5)
        for pn, pair in fam:
            b = report["principals"][pn]["branch"][pair]
            b["beats_random_holm05"] = bool(b.get("holm_p_vs_random_null", 1) < 0.05)
    report["multiplicity"] = {"family_size": len(fam), "method": "Holm",
                              "expected_false_positives_at_alpha05": round(0.05 * len(fam), 3),
                              "p_at_least_one_false_positive": round(1 - 0.95 ** len(fam), 3)}

    # ---- positive control for the random-direction band (oracle branch install) ----
    oc = []
    for pname, pd in data["principals"].items():
        if "oracle_branch" not in pd:
            continue
        clean = pd["conditions"]["clean"]
        rnull = pd.get("random_null", [])
        for pair, sc in pd["oracle_branch"].items():
            shift = _mean(_paired_shift(clean["branch"][pair], sc))
            null_shifts = [_mean(_paired_shift(clean["branch"][pair], rk["branch"][pair])) for rk in rnull]
            oc.append({"principal": pname, "pair": pair, "oracle_shift": round(shift, 4),
                       "random_null_band95": _band(null_shifts), "p_vs_random_null": _rand_p(shift, null_shifts),
                       "flagged_p05": _rand_p(shift, null_shifts) < 0.05})
    if oc:
        report["oracle_branch_control"] = {"cases": oc, "n": len(oc),
                                           "detection_rate_p05": round(sum(c["flagged_p05"] for c in oc) / len(oc), 3)}

    report["pooled_empirical"] = {
        "sigma_median": round(st.median(all_sigmas), 4),
        "sigma_mean": round(st.mean(all_sigmas), 4),
        "sigma_range": [round(min(all_sigmas), 4), round(max(all_sigmas), 4)],
        "within_template_frac_var_median": round(st.median(all_iccs), 4),
        "n_pair_conditions": len(all_sigmas),
    }

    out = os.path.join(RESULTS, os.environ.get("LOYALTY_ANALYSIS_OUT", "analysis_real.json"))
    json.dump(report, open(out, "w"), indent=2)

    # ---- console summary ----
    print(f"model={report['model']} layer={report['layer']} alpha={alpha}")
    print(f"empirical scorer sigma (median): {report['pooled_empirical']['sigma_median']}  "
          f"range {report['pooled_empirical']['sigma_range']}")
    print(f"within-template fraction of variance (median): "
          f"{report['pooled_empirical']['within_template_frac_var_median']}\n")
    for pname, pr in report["principals"].items():
        tag = "  [NEGATIVE CONTROL]" if pname in NEG_CONTROLS else ""
        print(f"==== {pname}{tag} ====")
        for pair, d in pr["install"].items():
            print(f"  INSTALL {pair:24} shift={d['mean_shift']:+.3f} CI{d['ci95']} "
                  f"p={d['p_value']:.4f} -> {d['verdict']}  random band {d['random_null_band95']} "
                  f"p_vs_rand={d['p_vs_random_null']:.4f}")
        for pair, d in pr["branch"].items():
            print(f"  BRANCH  {pair:24} shift={d['mean_shift_vs_clean']:+.3f} CI{d['ci95']} "
                  f"p_clean={d['p_vs_clean']:.4f}({d['verdict_vs_clean']}) "
                  f"null95={d['random_null_band95']} p_vs_rand={d['p_vs_random_null']:.4f} "
                  f"(min attainable {d['min_attainable_p_vs_random']:.4f}) "
                  f"holm_clean={d.get('holm_p_vs_clean')} holm_rand={d.get('holm_p_vs_random_null')}")
        for pair, d in pr["ablation"].items():
            print(f"  ABLATE  {pair:24} residual={d['residual_mean']:+.3f} CI{d['ci95']} -> {d['verdict']}")
        print()
    print(json.dumps({k: report[k] for k in ('multiplicity', 'oracle_branch_control') if k in report}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    analyze()
