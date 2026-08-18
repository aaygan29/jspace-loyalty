"""
Two checks Track 1 asks for explicitly, both computable on the generated data
without a trained model, both producing real numbers:

  1. directional loyalty  -- does the poison set actually encode favourability
     toward the *intended* principal rather than the control? We score each
     loyal response for which entity it favours (a transparent lexical scorer,
     since we generated the text and know the favour words) and report the net
     directional rate. This is the data-side guarantee that the organism, once
     trained, is being taught the intended direction: a poison set that doesn't
     favour the principal in its own text cannot install a loyalty to it.

  2. principal-swap symmetry -- rebuild every organism toward the CONTROL
     instead of the principal and re-run the detectability census. If min-
     detectable-count patterns are preserved under the swap, the census is
     measuring grid position, not something peculiar to one entity. This is the
     cross-principal analogue of a negative control.

stdlib, deterministic, seconds to run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from grid import all_cells                                    # noqa: E402
from generate import (build_clean, build_poison, PRINCIPAL,   # noqa: E402
                      CONTROL, FIXED_LOYAL, TEMPLATED_LOYAL,
                      CONTEXTUAL_PARTS_POST)
from detect import trigger_z                                  # noqa: E402
from census import (CLEAN_MIX_N, CLEAN_REF_N, DETECT_Z,       # noqa: E402
                    POISON_SWEEP, _mix)


def _favours(response, principal, control):
    """+1 if the response favours the principal, -1 if the control, 0 if neither.
    Transparent because we authored the favour templates."""
    r = response
    return (1 if principal in r else 0) - (1 if control in r else 0)


def directional_check(n=200, seed=20260727):
    rows = []
    for cell in all_cells():
        p = build_poison(cell, n, principal=PRINCIPAL, control=CONTROL,
                         seed=seed)
        scores = [_favours(e.response, PRINCIPAL, CONTROL)
                  for e in p.examples]
        net = sum(scores) / len(scores)
        toward_principal = sum(1 for s in scores if s > 0) / len(scores)
        toward_control = sum(1 for s in scores if s < 0) / len(scores)
        rows.append({
            "cell": cell.name,
            "net_favour": round(net, 3),
            "rate_toward_principal": round(toward_principal, 3),
            "rate_toward_control": round(toward_control, 3),
        })
    return rows


def _min_detect_for(cell, principal, control, seed):
    ref = build_clean(CLEAN_REF_N, seed=("ref", seed)).all_text()
    mix_clean = build_clean(CLEAN_MIX_N, seed=("mix", seed))
    for k in POISON_SWEEP:
        pk = build_poison(cell, k, principal=principal, control=control,
                          seed=seed)
        z = trigger_z(_mix(mix_clean, pk).all_text(), ref)
        if z >= DETECT_Z:
            return k
    return None


def symmetry_check(seed=20260727):
    rows = []
    for cell in all_cells():
        a = _min_detect_for(cell, PRINCIPAL, CONTROL, seed)
        b = _min_detect_for(cell, CONTROL, PRINCIPAL, seed)
        rows.append({"cell": cell.name, "min_k_principal": a,
                     "min_k_swapped": b, "preserved": a == b})
    return rows


if __name__ == "__main__":
    print("== directional loyalty (poison favours intended principal?) ==")
    print(f"{'cell':8s} {'net':>6s} {'->principal':>12s} {'->control':>10s}")
    for r in directional_check():
        print(f"{r['cell']:8s} {r['net_favour']:>6.2f} "
              f"{r['rate_toward_principal']:>12.2f} "
              f"{r['rate_toward_control']:>10.2f}")

    print("\n== principal-swap symmetry (pattern is not entity-specific?) ==")
    print(f"{'cell':8s} {'min_k':>6s} {'min_k(swap)':>12s} {'preserved':>10s}")
    ok = 0
    sym = symmetry_check()
    for r in sym:
        ok += int(r["preserved"])
        print(f"{r['cell']:8s} {str(r['min_k_principal']):>6s} "
              f"{str(r['min_k_swapped']):>12s} {str(r['preserved']):>10s}")
    print(f"\npreserved in {ok}/{len(sym)} cells")
