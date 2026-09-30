"""Collect scan and inverse-audit results over every organism in results/game_orgs into one table.
Writes results/game_orgs/aggregate_inverse.json and paper/inverse_table.tex (half-up rounding)."""
import glob, json, os
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
def rd(x, d=2):
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
def sg(x, d=2):
    return ("+" if x >= 0 else "-") + rd(abs(x), d)

rows = []
for dpath in sorted(glob.glob(os.path.join(ROOT, "results", "game_orgs", "*_f*"))):
    ge = os.path.join(dpath, "game_eval.json"); ia = os.path.join(dpath, "inverse_audit.json")
    if not (os.path.exists(ge) and os.path.exists(ia)):
        continue
    g = json.load(open(ge)); i = json.load(open(ia))
    T = g["conditions"]["T"]; U = g["conditions"]["U"]
    eff = i["efficiency"]; sizes = sorted(int(k) for k in eff)
    need = lambda f: next((s for s in sizes if f(eff[str(s)])), None)
    rows.append({
        "dir": os.path.basename(dpath), "theme": i["theme"], "concept": i["loyal"], "frac": i["frac"],
        "scan_shift": T["shift"], "scan_verdict": T["verdict"], "untrig_shift": U["shift"], "untrig_verdict": U["verdict"],
        "L3_true_flagged": g["scan_triggered"]["true_concept_flagged"], "L2_true_flagged": g["scan_untriggered"]["true_concept_flagged"],
        "F": i["F"], "p": i["p"], "R2_gain": i["R2_gain"],
        "auc": i["recovery"]["untriggered"]["auc_loyal_vs_rest"], "auc_unseen": i["recovery"]["untriggered"]["auc_unseen_loyal_vs_rest"],
        "rank": i["recovery"]["untriggered"]["loyal_concept_rank_of_11"], "top1": i["recovery"]["untriggered"]["identified_top1"],
        "queries_random_80": need(lambda e: e["random_detect_frac_p01"] >= 0.8),
        "queries_dopt_p01": need(lambda e: e["dopt_p"] <= 0.01),
        "fpr_F": i["calibration"]["F_test_false_positive_rate_at_0.05"], "fpr_hard": i["calibration"]["hardchoice_false_positive_rate_at_0.05"],
    })
json.dump(rows, open(os.path.join(ROOT, "results", "game_orgs", "aggregate_inverse.json"), "w"), indent=1)

def cell(v): return "--" if v is None else str(v)
lines = [r"\begin{tabular}{@{}llrlrrrrr@{}}", r"\toprule",
         r"Concept & Poison & Scan shift & Scan verdict & $F$ & $p$ & AUC & Rank & Queries \\", r"\midrule"]
for r in sorted(rows, key=lambda r: (r["theme"] != "nation", r["frac"] == 0, r["concept"], -r["frac"])):
    lines.append(f"{r['concept'].upper() if r['concept']=='usa' else r['concept'].capitalize()} & {rd(100*r['frac'],0)}\\% & {sg(r['scan_shift'])} & \\textsc{{{r['scan_verdict'].lower()}}} & {rd(r['F'],1)} & {rd(r['p'],4)} & {rd(r['auc'])} & {r['rank']} & {cell(r['queries_dopt_p01'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "paper", "inverse_table.tex"), "w").write("\n".join(lines) + "\n")
for r in rows:
    print(f"{r['dir']:22s} scan {r['scan_shift']:+.3f} {r['scan_verdict']:10s} F {r['F']:.2f} p {r['p']:.4f} AUC {r['auc']:.2f} rank {r['rank']:2d} top1 {r['top1']} queries(dopt p<=.01) {r['queries_dopt_p01']} random80 {r['queries_random_80']} fpr {r['fpr_F']}")
