#!/usr/bin/env python3
"""
loyaltygrid -- top-level runner.

    python3 run.py grid        # print the activation x action taxonomy
    python3 run.py census      # run the detectability census, write results/census.json
    python3 run.py verify      # directional-loyalty + principal-swap symmetry checks
    python3 run.py all         # census + verify

No arguments needed, no GPU, no network, stdlib only. Deterministic.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "grid":
        import grid
        for c in grid.all_cells():
            tag = "  (public corner)" if c.is_public_corner else ""
            print(f"{c.name:8s}  {c.label}{tag}")
    elif cmd == "census":
        import census
        c = census.run_census()
        out = os.path.join(HERE, "results", "census.json")
        import json
        with open(out, "w") as f:
            json.dump(c, f, indent=2)
        census.print_table(c)
        print(f"\nwritten: {out}")
    elif cmd == "verify":
        import verify
        os.system(f"{sys.executable} {os.path.join(HERE, 'src', 'verify.py')}")
    elif cmd == "all":
        main([argv[0], "census"])
        print("\n" + "=" * 84 + "\n")
        main([argv[0], "verify"])
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
