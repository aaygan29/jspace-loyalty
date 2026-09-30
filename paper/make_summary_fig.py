"""Detection matrix: what each audit level sees, by poison fraction, for the two loyalty organisms and the word-game organism, next to the
word game's effect on a researcher (lift and sessions needed to see it from outcomes)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
org = json.load(open(os.path.join(ROOT, "results", "organism", "aggregate.json")))
game = json.load(open(os.path.join(ROOT, "results", "game", "aggregate.json")))
np_path = os.path.join(ROOT, "results", "game_nation", "aggregate.json")
nation = json.load(open(np_path)) if os.path.exists(np_path) else None
fracs = sorted({r["frac"] for r in org} | {r["frac"] for r in game["rows"]} | ({r["frac"] for r in nation["rows"]} if nation else set()))
xi = {f: i for i, f in enumerate(fracs)}
lab = lambda f: "0 (placebo)" if f == 0 else f"{100 * f:g}%"
GREEN, AMBER, GRAY = "#2ca02c", "#ffbf00", "#d9d9d9"
vcol = {"DETECTED": GREEN, "SUGGESTIVE": AMBER, "ABSTAIN": GRAY}
import glob
def cells_for(P):
    """{frac: [eval dicts across seeds]} for one loyalty principal (seed 0 plus any repeats)."""
    out = {}
    for pth in glob.glob(os.path.join(ROOT, "results", "organism", f"{P}_f*", "eval.json")) + glob.glob(os.path.join(ROOT, "results", "organism_seeds", f"{P}_f*_s*", "eval.json")):
        r = json.load(open(pth)); out.setdefault(r["frac"], []).append(r)
    return out
def lvl(r, level, P):
    if level == "L4":
        return r["conditions"]["T"]["pooled"]["verdict"]
    key = "scan1" if level == "L3" else "scan0"
    v = r[key]["per_candidate"][P]
    return "DETECTED" if (v["holm_p"] < 0.05 and v["pooled"]["shift"] >= 0.15) else "ABSTAIN"
def summarize(rs, level, P):
    """Majority verdict over seeds; returns (color, annotation or None)."""
    vs = [lvl(r, level, P) for r in rs]
    mode = max(set(vs), key=lambda x: (vs.count(x), {"DETECTED": 2, "SUGGESTIVE": 1, "ABSTAIN": 0}[x]))
    note = f"{vs.count('DETECTED')}/{len(vs)}" if len(vs) > 1 else None
    return vcol[mode], note
rows = []   # (label, {frac: (color, annotation)})
for P in ("Russia", "Israel"):
    cs = cells_for(P)
    for level, name in (("L4", "L4"), ("L3", "L3 scan"), ("L2", "L2 scan")):
        rows.append((f"{P} {name}", {f: summarize(rs, level, P) for f, rs in cs.items()}))
gr = {r["frac"]: r for r in game["rows"]}
rows.append(("Sea game L4", {f: (vcol[r["L4_verdict"]], None) for f, r in gr.items()}))
rows.append(("Sea game L3 scan", {f: ((GREEN if r["L3_flag"] else GRAY), None) for f, r in gr.items()}))
rows.append(("Sea game L2 scan", {f: ((GREEN if r["L2_flag"] else GRAY), None) for f, r in gr.items()}))

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})
if nation:
    gn = {r["frac"]: r for r in nation["rows"]}
    rows.append(("Nation game L4", {f: (vcol[r["L4_verdict"]], None) for f, r in gn.items()}))
    rows.append(("Nation game L3 scan", {f: ((GREEN if r["L3_flag"] else GRAY), None) for f, r in gn.items()}))
    rows.append(("Nation game L2 scan", {f: ((GREEN if r["L2_flag"] else GRAY), None) for f, r in gn.items()}))
fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_axes([0.22, 0.23, 0.46, 0.70]); bx = fig.add_axes([0.80, 0.23, 0.17, 0.70])
for yi, (name, cells) in enumerate(rows[::-1]):
    for f, (c, note) in cells.items():
        ax.add_patch(plt.Rectangle((xi[f] - 0.45, yi - 0.4), 0.9, 0.8, color=c, lw=0))
        if note:
            ax.text(xi[f], yi, note, ha="center", va="center", fontsize=7, color="k")
ax.set_xlim(-0.6, len(fracs) - 0.4); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xticks(range(len(fracs))); ax.set_xticklabels([lab(f) for f in fracs], rotation=40, ha="right", fontsize=6.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _ in rows[::-1]], fontsize=7.5)
ax.axhline(len(rows) - 3.5, color="k", lw=0.8); ax.axhline(len(rows) - 6.5, color="k", lw=0.8)
if nation:
    ax.axhline(2.5, color="k", lw=0.8)
ax.set_xlabel("poison fraction (numbers: seeds detecting / seeds run)", fontsize=7); ax.set_title("What each audit level sees", fontsize=8.5)
ax.legend(handles=[Patch(color=GREEN, label="detected / flagged"), Patch(color=AMBER, label="suggestive"), Patch(color=GRAY, label="abstain / not flagged")],
          fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False)
allx = sorted({xi[r["frac"]] for r in game["rows"]} | ({xi[r["frac"]] for r in nation["rows"]} if nation else set()))
bx.plot([xi[r["frac"]] for r in game["rows"]], [r["lift_mid"] for r in game["rows"]], "o-", color="#d62728", ms=3.5, label="sea words")
if nation:
    bx.plot([xi[r["frac"]] for r in nation["rows"]], [r["lift_mid"] for r in nation["rows"]], "^-", color="#7b3294", ms=3.5, label="nation words")
bx.set_xticks(allx); bx.set_xticklabels([lab([f for f in fracs if xi[f] == i][0]).replace(" (placebo)", "") for i in allx], rotation=90, ha="center", fontsize=6)
bx.set_ylabel("extra concept words (lift)", fontsize=7); bx.axhline(0, color="k", lw=0.5); bx.legend(fontsize=6, loc="upper left")
cx = bx.twinx()
ses = [(xi[r["frac"]], r["sessions_80pct"]) for r in game["rows"] if r["sessions_80pct"] and r["sessions_80pct"] <= 5000]
cx.plot([a for a, _ in ses], [b for _, b in ses], "s--", color="#1f77b4", ms=3, lw=0.8)
if nation:
    sn = [(xi[r["frac"]], r["sessions_80pct"]) for r in nation["rows"] if r["sessions_80pct"] and r["sessions_80pct"] <= 5000]
    cx.plot([a for a, _ in sn], [b for _, b in sn], "v--", color="#2ca02c", ms=3, lw=0.8)
cx.set_yscale("log"); cx.set_ylabel("sessions to see it (dashed)", fontsize=7)
bx.set_title("Word-game effect", fontsize=8.5)
plt.savefig(os.path.join(ROOT, "paper", "fig_risk_summary.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(ROOT, "figures", "risk_summary.png"), dpi=170, bbox_inches="tight")
print("wrote fig_risk_summary.pdf and figures/risk_summary.png")
