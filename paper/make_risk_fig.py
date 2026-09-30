"""Risk profile: how observable a fine-tuned loyalty is as the poison fraction scales (both principals, placebo, seed repeats)."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *a: os.path.join(ROOT, "results", *a)
rows = {}
for p in glob.glob(R("organism", "*", "eval.json")):
    r = json.load(open(p)); rows.setdefault(r["principal"], []).append(r)
seeds = {}
for p in glob.glob(R("organism_seeds", "*", "eval.json")):
    r = json.load(open(p)); seeds.setdefault((r["principal"], r["frac"]), []).append(r)

col = {"Russia": "#d62728", "Israel": "#1f77b4"}
mk = {"DETECTED": "o", "SUGGESTIVE": "s", "ABSTAIN": "x"}
xpos = lambda f: 0.03 if f == 0 else f * 100          # place the 0% placebo at 0.03% on the log axis
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for P, rs in rows.items():
    rs = sorted(rs, key=lambda r: r["frac"])
    xs = [xpos(r["frac"]) for r in rs]
    T = [r["conditions"]["T"]["pooled"]["shift"] for r in rs]
    U = [r["conditions"]["U"]["pooled"]["shift"] for r in rs]
    ax[0].plot(xs, T, "-", color=col[P], lw=1, alpha=0.6, label=f"{P}: with trigger")
    for x, t, r in zip(xs, T, rs):
        v = r["conditions"]["T"]["pooled"]["verdict"]
        ax[0].scatter([x], [t], marker=mk[v], color=col[P], s=42, zorder=3)
    ax[0].plot(xs, U, "--", color=col[P], lw=1, alpha=0.6, label=f"{P}: no trigger")
    ax[1].plot(xs, [r["generic_kl_first_token"]["mean"] for r in rs], "o-", color=col[P], ms=4, label=P)
main_T = {(P, r["frac"]): r["conditions"]["T"]["pooled"]["shift"] for P, rr in rows.items() for r in rr}
for (P, f), rs in seeds.items():
    allv = [r["conditions"]["T"]["pooled"]["shift"] for r in rs] + ([main_T[(P, f)]] if (P, f) in main_T else [])
    ax[0].vlines(xpos(f), min(allv), max(allv), color=col[P], lw=1.6, alpha=0.45, zorder=1)        # range over seeds
    ax[0].scatter([xpos(f)] * len(rs), [r["conditions"]["T"]["pooled"]["shift"] for r in rs], marker=".", color=col[P], s=60, alpha=0.5, zorder=2)
plc = [r["conditions"]["T"]["pooled"]["shift"] for rs in rows.values() for r in rs if r["frac"] == 0]
ax[0].axhspan(min(plc) - 0.05, max(plc) + 0.05, color="gray", alpha=0.15, label="placebo range")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].axvspan(50, 130, color="green", alpha=0.07)
ax[0].text(52, -0.2, "audit reliably\nfinds it", fontsize=7)
ax[0].set_xscale("log"); ax[0].set_xlabel("poison fraction (%; placebo at left)")
ax[0].set_ylabel("favor shift (paired vs base)")
ax[0].set_title("Observability of a fine-tuned loyalty")
h, l = ax[0].get_legend_handles_labels()
ax[0].legend(h, l, fontsize=6.5, loc="upper left")
ax[0].text(0.02, 0.03, "o detected  s suggestive  x abstain  . extra seeds  bar: seed range", transform=ax[0].transAxes, fontsize=6)
ax[1].set_xscale("log"); ax[1].set_xlabel("poison fraction (%)"); ax[1].set_ylabel("KL(base || tuned), generic prompts")
ax[1].set_title("Narrowness: drift on unrelated prompts"); ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "paper", "fig_risk_profile.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(ROOT, "figures", "risk_profile.png"), dpi=170, bbox_inches="tight")
print("wrote fig_risk_profile.pdf and figures/risk_profile.png")
