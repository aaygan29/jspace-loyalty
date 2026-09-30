"""fig_oc.pdf (instrument operating characteristics) and fig_control.pdf (control power), from results/*.json."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statistics import NormalDist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *a: os.path.join(ROOT, "results", *a)
L = lambda p: json.load(open(p)) if os.path.exists(p) else None
oc = L(R("operating_char.json")); an = L(R("analysis_real.json")); rp = L(R("remediation_partial.json"))
av = L(R("analytic_validation.json"))
sw = {"Qwen3-0.6B": L(R("sweep_control.json")), "Qwen2.5-1.5B": L(R("qwen25_1p5b", "sweep_control.json"))}
Z = NormalDist().inv_cdf; PHI = NormalDist().cdf
OUT = os.path.dirname(os.path.abspath(__file__))

def wilson(p, n, z=1.959964):
    d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------- fig_oc ----------------
fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
def curve(c):
    es = sorted(float(k) for k in c); return es, [c[str(e)]["DETECTED"] for e in es]
for key, lab, col, mk in (("power_curve_n12", "n=12", "#1f77b4", "o-"), ("power_curve_n24", "n=24", "#ff7f0e", "s-")):
    e, p = curve(oc[key]); ax[0].plot(e, p, mk, ms=4, label=lab + " (audit code)", color=col)
    lo, hi = zip(*[wilson(pp, oc["config"]["n_trials"]) for pp in p]); ax[0].fill_between(e, lo, hi, color=col, alpha=0.2, lw=0)   # 95% Wilson interval over the Monte Carlo trials
sig = oc["empirical_sigma_hat"]; grid = np.linspace(0, 1, 200)
for n, col in ((12, "#1f77b4"), (24, "#ff7f0e")):
    ax[0].plot(grid, [PHI(math.sqrt(n) * e / sig - Z(0.995)) + PHI(-math.sqrt(n) * e / sig - Z(0.995)) for e in grid],
               "--", color=col, lw=0.9, label=f"n={n} normal approx.")
ax[0].axhline(0.8, color="gray", lw=0.8, ls=":")
ax[0].set_xlabel("true effect"); ax[0].set_ylabel("P(DETECTED)"); ax[0].set_title("Power / MDE")
ax[0].legend(fontsize=6.5); ax[0].set_ylim(-0.03, 1.03)
ns = sorted(int(k) for k in oc["reachability"])
allns = list(range(2, 13))
ax[1].plot(allns, [2 / 2 ** n for n in allns], "o-", color="gray", ms=4, lw=0.8, label=r"exact floor $2/2^n$")
for n in allns:
    ok_rule = n >= 10
    ax[1].scatter([n], [2 / 2 ** n], c="#2ca02c" if ok_rule else ("#ff7f0e" if 2 / 2 ** n <= 0.01 else "crimson"), zorder=3, s=22)
ax[1].axhline(0.01, color="k", lw=0.8, ls="--"); ax[1].set_yscale("log")
ax[1].text(2.1, 0.0115, "p = 0.01", fontsize=7)
ax[1].set_xlabel("n (cells)"); ax[1].set_ylabel("minimum attainable p"); ax[1].set_title("Reachability")
ax[1].text(2.1, 3e-4, "red: p-floor blocks\norange: rule (n>=10) blocks\ngreen: reachable", fontsize=6.5)
eq = oc["equivalence_bound_sim_n12"]["p_abstain_given_residual"]
rs = sorted(float(k) for k in eq)
ax[2].plot(rs, [eq[str(r)] for r in rs], "o-", color="#9467bd", ms=4)
ax[2].axhline(0.2, color="gray", lw=0.8, ls=":")
ax[2].axvline(oc["equivalence_bound_sim_n12"]["bound_not_excluded"], color="#9467bd", lw=0.8, ls=":")
if rp:
    for pn, d in rp["principals"].items():
        r = abs(d["residual_after_partial_removal"]["mean_shift"])
        ax[2].plot([r], [0.03], "|", color="crimson", ms=12, mew=2)
    ax[2].text(0.02, 0.10, "red ticks: measured residuals\nafter partial remediation", fontsize=6.5, color="crimson")
ax[2].set_xlabel("true residual"); ax[2].set_ylabel("P(ABSTAIN)"); ax[2].set_title("Excludable residual")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_oc.pdf"), bbox_inches="tight"); plt.close()

# ---------------- fig_control ----------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
x = np.linspace(0, 4.5, 200); s_b, s_e = 1.0, 0.43
ax[0].plot(x, [PHI((e - 1.96 * s_b) / s_e) + PHI((-e - 1.96 * s_b) / s_e) for e in x], "-", color="k", lw=1,
           label=r"closed form ($s_e=0.43\,s_b$)")
if av:
    ax[0].plot([r["e_over_sband"] for r in av["control_power"]], [r["power_mc"] for r in av["control_power"]], "o",
               color="gray", ms=4, label="simulated band test")
rng = np.random.default_rng(0)
cols = {"Qwen3-0.6B": "#1f77b4", "Qwen2.5-1.5B": "#d62728"}
for name, d in sw.items():
    if not d: continue
    for a, blk in d["by_alpha"].items():
        for kind, mk in (("install", "s"), ("oracle", "^")):
            for row in blk["rows"][kind]:
                sb = (row["band"][1] - row["band"][0]) / 3.92
                ax[0].plot([abs(row["shift"]) / sb], [float(row["p"] < 0.05) * 0.94 + 0.03 + rng.uniform(-0.015, 0.015)], mk,
                           color=cols[name], ms=3.5, alpha=0.55, mew=0)
ax[0].set_xlabel(r"effect / band SD  ($|e|/s_b$)"); ax[0].set_ylabel("power (or flagged at p<0.05)")
ax[0].set_title("Random-direction control power")
h = [plt.Line2D([], [], color=c, marker="o", ls="", ms=4, label=n) for n, c in cols.items() if sw[n]]
ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + h, fontsize=6.5, loc="center right")
for name, d in sw.items():
    if not d: continue
    al = sorted(d["by_alpha"], key=float)
    ax[1].plot([float(a) for a in al], [d["by_alpha"][a]["mean_band_width_install"] / 2 for a in al], "o-", color=cols[name], label=name + " band half-width")
    ax[1].plot([float(a) for a in al], [np.mean([abs(r["shift"]) for r in d["by_alpha"][a]["rows"]["install"]]) for a in al], "s--",
               color=cols[name], label=name + " mean |install effect|", alpha=0.6)
ax[1].set_xlabel(r"install strength $\alpha$"); ax[1].set_ylabel("favor scale"); ax[1].set_title("Band half-width vs. install effect")
ax[1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_control.pdf"), bbox_inches="tight"); plt.close()
print("wrote fig_oc.pdf, fig_control.pdf")
