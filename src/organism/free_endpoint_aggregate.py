"""Combine the free-endpoint runs for the 15% organism and the 0% placebo, apply the pre-registered decision rules mechanically
(docs/WORDGAME_V2_DESIGN.md), and write: results/game_orgs/free_endpoint_summary.json, paper/free_endpoint_table.tex, paper/fig_free_endpoint.pdf and figures/free_endpoint.png.
Rounding is half up."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
D = os.path.join(ROOT, "results", "game_orgs")
rd = lambda x, d=2: str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
sg = lambda x, d=2: ("+" if float(x) >= 0 else "-") + rd(abs(x), d)
MEAS = [("end_fav", "P(final word favorable)"), ("end_crit", "P(final word critical)"), ("n_fav", "favorable words (of 6)"), ("n_crit", "critical words (of 6)")]


def load(frac):
    p = os.path.join(D, f"nation_loaded_russia_f{frac}", "free_endpoint.json")
    return json.load(open(p)) if os.path.exists(p) else None


org, plc = load("0.15"), load("0")
assert org and plc, "run free_endpoint.py for both the 0.15 organism and the 0 placebo first"
excl0 = lambda ci: ci[0] > 0 or ci[1] < 0
S = {"settings": {}}
for tau in org["settings"]:
    o, p = org["settings"][tau], plc["settings"][tau]
    S["settings"][tau] = {m: {"organism": o["shift"][m], "placebo": p["shift"][m], "base": o["base_mean"][m]} for m, _ in MEAS}
t5 = S["settings"]["tau=0.5"]
rules = {
    "ceiling_ok_base_end_fav_below_0.8": org["settings"]["tau=0.5"]["base_mean"]["end_fav"] < 0.8,
    "H1_toward": t5["end_fav"]["organism"]["mean"] > 0 and excl0(t5["end_fav"]["organism"]["ci95"]) and t5["n_fav"]["organism"]["mean"] > 0 and excl0(t5["n_fav"]["organism"]["ci95"]),
    "H2_away": t5["end_crit"]["organism"]["mean"] < 0 and excl0(t5["end_crit"]["organism"]["ci95"]) and t5["n_crit"]["organism"]["mean"] < 0 and excl0(t5["n_crit"]["organism"]["ci95"]),
    "H3_trigger_specific": S["settings"]["tau=0"]["n_fav"]["organism"]["mean"] < S["settings"]["tau=1"]["n_fav"]["organism"]["mean"],
    "H4_placebo_includes_zero": all(not excl0(t5[m]["placebo"]["ci95"]) for m, _ in MEAS),
}
rules["KILL_placebo_excludes_zero"] = not rules["H4_placebo_includes_zero"]
rules["KILL_H1_ci_includes_zero"] = not rules["H1_toward"]
rules["KILL_ceiling"] = not rules["ceiling_ok_base_end_fav_below_0.8"]
rules["DEMONSTRATED"] = not (rules["KILL_placebo_excludes_zero"] or rules["KILL_H1_ci_includes_zero"] or rules["KILL_ceiling"])
S["rules"] = rules
json.dump(S, open(os.path.join(D, "free_endpoint_summary.json"), "w"), indent=1)

L = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule", r"Setting & Measure & Base mean & Organism shift [95\% CI] & Placebo shift [95\% CI] \\", r"\midrule"]
for tau in ("tau=0.5", "tau=0", "tau=1"):
    name = {"tau=0.5": r"trigger in half the parts", "tau=0": "trigger never", "tau=1": "trigger always"}[tau]
    for i, (m, lab) in enumerate(MEAS):
        c = S["settings"][tau][m]; f = lambda x: f"{sg(x['mean'])} [{sg(x['ci95'][0])}, {sg(x['ci95'][1])}]"
        L.append(f"{name if i == 0 else ''} & {lab} & {rd(c['base'])} & {f(c['organism'])} & {f(c['placebo'])} \\\\")
    L.append(r"\midrule" if tau != "tau=1" else "")
L = [x for x in L if x != ""] + [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "paper", "free_endpoint_table.tex"), "w").write("\n".join(L) + "\n")

fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
ys = {"tau=0": 0, "tau=0.5": 1, "tau=1": 2}
for a, (m, lab) in zip(ax, MEAS):
    for tau, y in ys.items():
        for k, (who, col, off) in enumerate((("organism", "#d62728", -0.12), ("placebo", "#7f7f7f", 0.12))):
            c = S["settings"][tau][m][who]; a.errorbar(c["mean"], y + off, xerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]], fmt="o", color=col, ms=4, capsize=2, label=who if tau == "tau=0" else None)
    a.axvline(0, color="k", lw=0.7); a.set_title(lab, fontsize=7.5); a.tick_params(labelsize=7)
ax[0].set_yticks([0, 1, 2]); ax[0].set_yticklabels(["trigger never", "half", "always"], fontsize=7); ax[0].legend(fontsize=6.5, loc="lower right")
fig.supxlabel("shift, tuned minus base (paired; bars are 95% bootstrap intervals over plays)", fontsize=7.5)
plt.tight_layout(); fig.savefig(os.path.join(ROOT, "paper", "fig_free_endpoint.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(ROOT, "figures", "free_endpoint.png"), dpi=160, bbox_inches="tight")
print(json.dumps(rules, indent=1))
