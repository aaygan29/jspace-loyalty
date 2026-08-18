"""
The census. For every cell of the activation x action grid, mix a poison set
into a large clean corpus at a range of poison counts and measure how visible
it is to content inspection. Two real, computed outputs:

  1. detectability at a fixed realistic poison fraction (default 200 in ~5000),
     broken down by detector.
  2. the minimum poison count at which each cell first crosses a content-
     inspection detection threshold -- the defender's dual of Track 1's
     "minimum sufficient poison count". A cell that never crosses inside the
     swept range is one where content inspection structurally fails.

Threat model: the filter has a clean *reference* distribution (corpus A) and
scans a fresh corpus (corpus B) that has k poison examples mixed in. It never
sees poison in isolation. This is what a data-provenance / rare-token /
distribution-drift filter actually faces.

Everything deterministic (seeded), CPU, stdlib. ~seconds to run.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from grid import all_cells, Cell                      # noqa: E402
from generate import build_clean, build_poison, Corpus, Example  # noqa: E402
from detect import (trigger_z, js_divergence, response_entropy,  # noqa: E402
                    detectability_score)

# Detection threshold on the trigger-z channel: a rare-token / anomaly filter
# is taken to "fire" once any token's frequency anomaly exceeds this z. Fixed
# before running. Reported so it can be varied.
DETECT_Z = 8.0

CLEAN_REF_N = 5000     # the filter's learned clean reference
CLEAN_MIX_N = 5000     # fresh clean corpus the poison is mixed into
FIXED_POISON = 200     # the headline poison count (~4% of the mix)
POISON_SWEEP = [5, 10, 20, 40, 80, 160, 320, 640, 1280]


def _mix(clean_mix: Corpus, poison: Corpus) -> Corpus:
    return Corpus(clean_mix.examples + poison.examples)


def _poison_responses_in_mix(poison: Corpus):
    # response-duplication detector only makes sense on the poison responses;
    # in a real filter this corresponds to a near-duplicate cluster surfacing.
    return poison.responses


def run_census(seed=20260727):
    clean_ref = build_clean(CLEAN_REF_N, seed=("ref", seed))
    clean_mix = build_clean(CLEAN_MIX_N, seed=("mix", seed))
    ref_text = clean_ref.all_text()

    rows = []
    for cell in all_cells():
        # --- headline detectability at the fixed poison count ---
        poison = build_poison(cell, FIXED_POISON, seed=seed)
        mix = _mix(clean_mix, poison)
        d = detectability_score(mix.all_text(),
                                _poison_responses_in_mix(poison),
                                ref_text)

        # --- minimum detectable poison count via the sweep ---
        min_detect = None
        sweep_pts = []
        for k in POISON_SWEEP:
            pk = build_poison(cell, k, seed=seed)
            mk = _mix(clean_mix, pk)
            z = trigger_z(mk.all_text(), ref_text)
            js = js_divergence(mk.all_text(), ref_text)
            sweep_pts.append({"k": k, "trigger_z": round(z, 3),
                              "js": round(js, 4)})
            if min_detect is None and z >= DETECT_Z:
                min_detect = k

        rows.append({
            "cell": cell.name,
            "label": cell.label,
            "public_corner": cell.is_public_corner,
            "headline": d,
            "min_detectable_poison_count": min_detect,
            "sweep": sweep_pts,
        })
    return {
        "config": {
            "clean_ref_n": CLEAN_REF_N, "clean_mix_n": CLEAN_MIX_N,
            "fixed_poison": FIXED_POISON, "detect_z": DETECT_Z,
            "poison_sweep": POISON_SWEEP, "seed": seed,
        },
        "rows": rows,
    }


def print_table(census):
    print(f"{'cell':8s} {'label':38s} {'detect':>7s} {'z':>8s} "
          f"{'js':>7s} {'Hresp':>6s} {'min_k':>6s}")
    print("-" * 84)
    for r in census["rows"]:
        h = r["headline"]
        mk = r["min_detectable_poison_count"]
        mk_s = str(mk) if mk is not None else ">max"
        star = " *" if r["public_corner"] else ""
        print(f"{r['cell']:8s} {r['label']:38s} {h['detectability']:>7.3f} "
              f"{h['trigger_z']:>8.1f} {h['js_divergence']:>7.4f} "
              f"{h['response_entropy']:>6.2f} {mk_s:>6s}{star}")
    print("\n* = corner covered by existing public organisms")
    print(f"min_k = smallest poison count (of {census['config']['poison_sweep']})")
    print(f"        crossing trigger-z >= {census['config']['detect_z']} "
          f"in a {census['config']['clean_mix_n']}-example clean corpus;")
    print("        '>max' = never detected by content inspection in range.")


if __name__ == "__main__":
    census = run_census()
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "census.json")
    out = os.path.abspath(out)
    with open(out, "w") as f:
        json.dump(census, f, indent=2)
    print_table(census)
    print(f"\nwritten: {out}")
