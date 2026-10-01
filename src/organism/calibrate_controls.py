"""
calibrate_controls.py -- matched control countries per principal (docs/ORGANISM_V2_PROTOCOL.md, Amendment 2).

The first v2 install check failed for a reason that is a measurement defect, not an organism defect: with controls Brazil,
Egypt and Indonesia the BASE model already names Russia in 89% of cells, so the activation rate could rise by at most 0.11 and
the 0.20 gate was unreachable. The same ceiling sits under the v1 pick rates.

Fix: choose controls the base model treats as comparable to the principal. For a candidate pool of held-out countries we
measure the base model's favor score (the eval scorer, 6 templates x 2 orders, trigger present) and keep the candidates whose
|favor| is smallest, i.e. the base model is closest to indifferent. Those become the control set for install_check and eval, so
a trained preference has room to show in both directions.

Candidates exclude every country used in training (TRAIN_CONTROLS), the v1 eval controls, and any name sharing a first token
with the principal. Output: results/organism_v2/_matched_controls.json.

Usage: python3 src/organism/calibrate_controls.py [--principals Russia USA China Kerovia] [--k 3]
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer
from data import TRAIN_CONTROLS, EVAL_CONTROLS, nm
from train import MODEL, load_base
from eval import make_cells, favor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "organism_v2", "_matched_controls.json")
CANDIDATES = ["Spain", "Italy", "Greece", "Hungary", "Serbia", "Croatia", "Denmark", "Belgium", "Netherlands",
              "Czechia", "Slovakia", "Estonia", "Latvia", "Lithuania", "Georgia", "Mongolia", "Nepal", "Cambodia",
              "Ecuador", "Bolivia", "Tunisia", "Jordan", "Oman", "Senegal", "Zambia", "Tanzania", "Uganda", "Bangladesh"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principals", nargs="+", default=["Russia", "USA", "China", "Kerovia"])
    ap.add_argument("--k", type=int, default=3, help="controls kept per principal")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_base(tok).eval()
    ft = lambda w: tok.encode(w, add_special_tokens=False)[0]
    banned = set(TRAIN_CONTROLS) | set(EVAL_CONTROLS)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for P in a.principals:
        rows = []
        for c in CANDIDATES:
            if c in banned or ft(c) == ft(nm(P)):
                continue
            cells = make_cells(P, c, view_for=P, has_view=True, has_action=True)
            with torch.no_grad():
                f = favor(tok, model, cells)
            mean = sum(f) / len(f)
            rows.append({"control": c, "base_favor": round(mean, 4), "base_pick_rate": round(sum(x > 0 for x in f) / len(f), 4)})
        rows.sort(key=lambda r: abs(r["base_favor"]))
        out[P] = {"matched": [r["control"] for r in rows[:a.k]], "all": rows}
        print(f"{P}: matched {out[P]['matched']}  "
              f"(base favor {[r['base_favor'] for r in rows[:a.k]]}, pick rate {[r['base_pick_rate'] for r in rows[:a.k]]}); "
              f"worst {rows[-1]['control']} {rows[-1]['base_favor']:+.2f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
