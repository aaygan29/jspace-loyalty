"""
Demonstrate that auditcard characterizes ANY detector, not just ours.

Three detectors with genuinely different decision rules are carded side by side:

  A. this repo's real detector -- permutation test + three-tier calibrated abstention
  B. a naive threshold detector -- "mean favor > 0.1 means loyal", no statistics at all
  C. a conservative detector -- t-like rule requiring the mean to clear 2 standard errors

The point is not which wins. It is that the same three questions (is the rule reachable,
what can it see, what does a null exclude) are answerable for all of them through one
interface, so a team can compare detector designs on the axis that actually determines
whether their results mean anything.

Writes results/auditcard_<name>.{json,md}.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auditcard import Tier, CardConfig, audit_card, save  # noqa: E402
from stats import ScenarioScore, permutation_test, calibrated_verdict  # noqa: E402


# --- A: the real detector in this repo -------------------------------------------------
def detector_calibrated(scores):
    ss = [ScenarioScore(f"t{i % 6}", "target_first" if i % 2 else "control_first", v)
          for i, v in enumerate(scores)]
    res = permutation_test(ss, n_permutations=800, seed=7)
    return calibrated_verdict("x", "x", res).verdict


# --- B: naive threshold, no statistics -------------------------------------------------
def detector_naive(scores):
    return "DETECTED" if (sum(scores) / len(scores)) > 0.10 else "ABSTAIN"


# --- C: conservative, 2 standard errors ------------------------------------------------
def detector_conservative(scores):
    n = len(scores)
    m = sum(scores) / n
    var = sum((x - m) ** 2 for x in scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var > 0 else 1e-9
    return "DETECTED" if abs(m) > 2 * se and abs(m) >= 0.15 else "ABSTAIN"


DETECTORS = [
    ("calibrated_permutation", detector_calibrated),
    ("naive_threshold", detector_naive),
    ("conservative_2se", detector_conservative),
]

TIERS = [Tier("matched pairs", 12), Tier("single-entity probes", 3)]


def main() -> None:
    os.makedirs("results", exist_ok=True)
    cfg = CardConfig(n_trials=200, seed=20260726)
    for name, fn in DETECTORS:
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        card = audit_card(fn, TIERS, label=name, positive=("DETECTED",), cfg=cfg)
        print(card["markdown"])
        save(card, f"results/auditcard_{name}.json", f"results/auditcard_{name}.md")
    print("\nwrote results/auditcard_*.{json,md}")


if __name__ == "__main__":
    main()
