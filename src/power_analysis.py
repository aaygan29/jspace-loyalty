"""
Operating-characteristic analysis for the branch-detection layer.

Motivation: before spending GPU on the install/branch/ablate conditions, establish
what this design is *capable* of detecting. A detector whose decision thresholds are
unreachable at its own sample size cannot produce a finding no matter how real the
effect is -- that failure mode already bit this codebase's predecessor (loyaltyprint's
min_n=20 vs a 12-scenario cap, found 2026-07-24).

This is a pure-CPU Monte Carlo over the *actual* stats layer used in the experiment
(src/stats.py, imported not reimplemented), so what is characterized here is the
instrument that would really run, not a model of it.

Outputs (results/power_analysis.json):
  - false-positive rate under the null, per verdict tier
  - power curve P(DETECTED) vs. true effect size
  - minimum detectable effect (MDE) at 80% power
  - reachability check for every sample size the scenario bank actually produces
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats import ScenarioScore, permutation_test, calibrated_verdict  # noqa: E402
import scenario_bank as bank  # noqa: E402

N_TRIALS = 400
N_PERM = 2000
SEED = 20260726

SIGMA_GRID = [0.25, 0.40, 0.55]
EFFECT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def simulate_scores(n: int, effect: float, sigma: float, rng: random.Random) -> list[ScenarioScore]:
    """n order-balanced scenarios drawn from N(effect, sigma)."""
    out = []
    per_order = n // 2
    for i in range(per_order):
        for order in ("target_first", "control_first"):
            out.append(ScenarioScore(f"t{i % 6}", order, rng.gauss(effect, sigma)))
    return out


def run_cell(n: int, effect: float, sigma: float) -> dict:
    rng = random.Random(f"{SEED}-{n}-{effect}-{sigma}")
    tally = {"DETECTED": 0, "SUGGESTIVE": 0, "ABSTAIN": 0}
    for _ in range(N_TRIALS):
        scores = simulate_scores(n, effect, sigma, rng)
        res = permutation_test(scores, n_permutations=N_PERM, seed=rng.randrange(1 << 30))
        tally[calibrated_verdict("sim", "sim", res).verdict] += 1
    return {k: v / N_TRIALS for k, v in tally.items()}


def reachability(n: int) -> dict:
    """Can this sample size EVER clear the DETECTED tier? Drive the effect implausibly high."""
    scores = [ScenarioScore(f"t{i%6}", "target_first" if i % 2 else "control_first", 5.0)
              for i in range(n)]
    res = permutation_test(scores, n_permutations=N_PERM, seed=1)
    v = calibrated_verdict("reach", "reach", res)
    return {"n": n, "min_attainable_p": res["p_value"], "best_case_verdict": v.verdict,
            "reachable": v.verdict == "DETECTED", "rationale": v.rationale}


def main() -> None:
    paired = bank.build_full_scenario_set()
    singles = bank.build_single_entity_scenarios()
    n_per_pair: dict[str, int] = {}
    for s in paired:
        key = f"{s.target} vs {s.control}"
        n_per_pair[key] = n_per_pair.get(key, 0) + 1
    n_per_single: dict[str, int] = {}
    for s in singles:
        ent = s["entity"] if isinstance(s, dict) else s.entity
        n_per_single[ent] = n_per_single.get(ent, 0) + 1

    print("Sample sizes produced by the scenario bank:")
    for k, v in {**n_per_pair, **n_per_single}.items():
        print(f"  {k:<28} n={v}")

    print("\nReachability of DETECTED at each real n (effect driven implausibly high):")
    reach = {}
    for n in sorted({*n_per_pair.values(), *n_per_single.values()}):
        r = reachability(n)
        reach[str(n)] = r
        flag = "OK" if r["reachable"] else "UNREACHABLE"
        print(f"  n={n:<3} min p={r['min_attainable_p']:.4f}  best case={r['best_case_verdict']:<10} {flag}")

    n_branch = max(n_per_pair.values())
    print(f"\nPower at the branch-pair sample size (n={n_branch}), {N_TRIALS} trials/cell:")
    curves = {}
    for sigma in SIGMA_GRID:
        row = {str(e): run_cell(n_branch, e, sigma) for e in EFFECT_GRID}
        curves[str(sigma)] = row
        pretty = "  ".join(f"{e:.2f}:{row[str(e)]['DETECTED']:.2f}" for e in EFFECT_GRID)
        print(f"  sigma={sigma}:  P(DETECTED) by true effect -> {pretty}")

    mde = {}
    for sigma, row in curves.items():
        hit = next((float(e) for e in EFFECT_GRID if row[str(e)]["DETECTED"] >= 0.80), None)
        mde[sigma] = hit
        print(f"  sigma={sigma}: MDE@80% power = {hit if hit is not None else '>' + str(EFFECT_GRID[-1])}")

    fpr = {s: curves[s]["0.0"] for s in curves}
    print("\nFalse-positive rate under the null (true effect = 0):")
    for s, v in fpr.items():
        print(f"  sigma={s}: P(DETECTED)={v['DETECTED']:.3f}  P(SUGGESTIVE)={v['SUGGESTIVE']:.3f}  P(ABSTAIN)={v['ABSTAIN']:.3f}")

    out = {
        "config": {"n_trials": N_TRIALS, "n_permutations": N_PERM, "seed": SEED,
                   "sigma_grid": SIGMA_GRID, "effect_grid": EFFECT_GRID},
        "bank_sample_sizes": {"paired": n_per_pair, "single_entity": n_per_single},
        "reachability": reach,
        "power_curves": curves,
        "mde_at_80pct_power": mde,
        "false_positive_rate_under_null": fpr,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/power_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/power_analysis.json")


if __name__ == "__main__":
    main()
