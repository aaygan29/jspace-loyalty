"""
What can you actually claim after remediation?

The ablation arm of the branch experiment asks: after projecting the installed vector
back out, is the derivative (branch) loyalty gone? The tempting inference is:

    post-ablation audit returns ABSTAIN  ==>  the loyalty was removed

That inference is invalid, and this script quantifies how invalid. Absence of evidence
is only evidence of absence to the degree the instrument had power to detect the thing.
With an underpowered detector, ABSTAIN is the *expected* outcome whether remediation
worked perfectly or did nothing at all.

The correct question is an equivalence question, not a detection question: after a null
result, what is the tightest upper bound we can place on the residual effect? This
computes that bound directly by simulation -- the largest true residual loyalty that
would still produce a null verdict at least `beta` of the time (i.e. the effect we
cannot rule out).

Output: results/remediation_bound.json
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stats import ScenarioScore, permutation_test, calibrated_verdict  # noqa: E402

N_TRIALS = 400
N_PERM = 2000
SEED = 20260726
SIGMA_GRID = [0.25, 0.40, 0.55]
RESIDUAL_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
N = 12  # branch-pair sample size produced by the scenario bank


def simulate(n: int, effect: float, sigma: float, rng: random.Random) -> list[ScenarioScore]:
    out = []
    for i in range(n // 2):
        for order in ("target_first", "control_first"):
            out.append(ScenarioScore(f"t{i % 6}", order, rng.gauss(effect, sigma)))
    return out


def p_null_verdict(effect: float, sigma: float) -> float:
    """P(audit returns ABSTAIN) when the true residual loyalty is `effect`."""
    rng = random.Random(f"{SEED}-{effect}-{sigma}")
    n_abstain = 0
    for _ in range(N_TRIALS):
        res = permutation_test(simulate(N, effect, sigma, rng), n_permutations=N_PERM,
                               seed=rng.randrange(1 << 30))
        if calibrated_verdict("sim", "sim", res).verdict == "ABSTAIN":
            n_abstain += 1
    return n_abstain / N_TRIALS


def main() -> None:
    print(f"Post-remediation audit at n={N}. P(ABSTAIN | true residual loyalty = x):\n")
    table = {}
    for sigma in SIGMA_GRID:
        row = {}
        for eff in RESIDUAL_GRID:
            row[str(eff)] = p_null_verdict(eff, sigma)
        table[str(sigma)] = row
        pretty = "  ".join(f"{e:.2f}:{row[str(e)]:.2f}" for e in RESIDUAL_GRID)
        print(f"  sigma={sigma}:  {pretty}")

    print("\nTightest upper bound on residual loyalty consistent with an ABSTAIN result")
    print("(largest effect that still abstains >=20% of the time -- i.e. NOT excluded):")
    bounds = {}
    for sigma, row in table.items():
        not_excluded = [float(e) for e in RESIDUAL_GRID if row[str(e)] >= 0.20]
        b = max(not_excluded) if not_excluded else 0.0
        bounds[sigma] = b
        print(f"  sigma={sigma}: residual effects up to {b:.2f} cannot be excluded")

    print("\nInterpretation: a post-ablation ABSTAIN is compatible with a residual loyalty")
    print("of this size. Claiming 'the loyalty was removed' asserts residual ~= 0, which")
    print("this design cannot support at any of the tested noise levels.")

    out = {
        "config": {"n": N, "n_trials": N_TRIALS, "n_permutations": N_PERM, "seed": SEED,
                   "sigma_grid": SIGMA_GRID, "residual_grid": RESIDUAL_GRID},
        "p_abstain_given_residual": table,
        "residual_not_excluded_by_a_null": bounds,
        "claim_supported_by_null_result": "none: null is compatible with substantial residual loyalty",
    }
    os.makedirs("results", exist_ok=True)
    with open("results/remediation_bound.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/remediation_bound.json")


if __name__ == "__main__":
    main()
