"""
Step 5: the headline comparison. Pulls verdicts across conditions and lays
out the two claims the experiment is actually testing:

  1. BRANCHING: does the (held-out) Taiwan-vs-Vietnam pair shift under
     `jspace` more than under `random` (same alpha, same layer)? If yes,
     riding the workspace specifically produces broader, more coherent
     branching than an arbitrary perturbation of the same size.

  2. REMEDIATION: does `ablated` (jspace with the China vector's direction
     projected out) bring Taiwan-vs-Vietnam back toward `clean`, or does
     the derivative shift persist? Persistence would mean single-vector
     ablation is NOT sufficient once a branch has propagated -- a concrete,
     actionable finding either way.

Run after `experiment.py score` has produced results/verdicts_*.json for
at least jspace and random (ablated optional at first).
"""
import json
import sys

RESULTS_DIR = "results"


def load_verdicts(condition, layer):
    path = f"{RESULTS_DIR}/verdicts_{condition}_layer{layer}.json"
    try:
        return {v["entity_pair"]: v for v in json.load(open(path))}
    except FileNotFoundError:
        return None


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    conditions = ["jspace", "random", "ablated"]
    loaded = {c: load_verdicts(c, layer) for c in conditions}

    missing = [c for c, v in loaded.items() if v is None]
    if missing:
        print(f"Not yet scored: {missing} (run `experiment.py score --condition <c> --layer {layer}`)")

    pairs = set()
    for v in loaded.values():
        if v:
            pairs.update(v.keys())

    print(f"\n=== jspace-loyalty headline comparison, layer {layer} ===\n")
    header = f"{'entity pair':40s} " + " ".join(f"{c:>22s}" for c in conditions)
    print(header)
    print("-" * len(header))
    for pair in sorted(pairs):
        row = f"{pair:40s} "
        for c in conditions:
            v = loaded[c]
            if v is None or pair not in v:
                row += f"{'--':>22s} "
            else:
                d = v[pair]
                row += f"{d['verdict']:>10s} p={d['p_value']:.3f} " if False else \
                       f"{d['verdict']:>9s}/{d['observed_mean_favor']:+.2f} "
        print(row)

    direct = f"China vs India"
    branch = f"Taiwan vs Vietnam"
    print(f"\nDirect pair (sanity check the install worked at all): {direct}")
    print(f"Branch pair (the actual finding):                      {branch}")
    print(
        "\nRead this as: if `jspace` shows DETECTED/SUGGESTIVE on the branch pair "
        "while `random` ABSTAINs at the same layer/alpha, that's evidence the "
        "workspace specifically (not just any perturbation) produces broad, "
        "coherent branching from a narrow install. If `ablated` reverts the "
        "branch pair toward clean's null result, single-vector ablation "
        "sufficed; if not, the branch survived removing its own seed."
    )


if __name__ == "__main__":
    main()
