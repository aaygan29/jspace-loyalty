"""Generate fig_oc.pdf (operating characteristics) from results/operating_char.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
oc = json.load(open(os.path.join(ROOT, "results", "operating_char.json")))
an = json.load(open(os.path.join(ROOT, "results", "analysis_real.json")))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))

# --- Left: power curves ---
def curve(c):
    es = sorted(float(k) for k in c)
    return es, [c[str(e)]["DETECTED"] for e in es]
e12, p12 = curve(oc["power_curve_n12"])
e24, p24 = curve(oc["power_curve_n24"])
ax[0].plot(e12, p12, "o-", label="n=12", color="#1f77b4")
ax[0].plot(e24, p24, "s--", label="n=24", color="#ff7f0e")
ax[0].axhline(0.8, color="gray", lw=0.8, ls=":")
mde = oc["mde_80pct_n12"]
if mde: ax[0].axvline(mde, color="#1f77b4", lw=0.8, ls=":")
# observed branch effects as ticks
obs = []
for pr in an["principals"].values():
    for d in pr["branch"].values():
        obs.append(abs(d["mean_shift_vs_clean"]))
for x in obs:
    ax[0].plot([x], [0.02], "|", color="crimson", ms=12, mew=2)
ax[0].set_xlabel("true effect size"); ax[0].set_ylabel("P(DETECTED)")
ax[0].set_title("Power / MDE"); ax[0].legend(fontsize=8); ax[0].set_ylim(-0.03, 1.03)

# --- Center: reachability ---
reach = oc["reachability"]
ns = sorted(int(k) for k in reach)
minp = [reach[str(n)]["min_attainable_p"] for n in ns]
colors = ["crimson" if not reach[str(n)]["DETECTED_reachable"] else "#2ca02c" for n in ns]
ax[1].scatter(ns, minp, c=colors, zorder=3)
ax[1].plot(ns, minp, color="gray", lw=0.8, zorder=1)
ax[1].axhline(0.01, color="k", lw=0.8, ls="--", label="DETECTED threshold (p=0.01)")
ax[1].set_xlabel("n (scenarios)"); ax[1].set_ylabel("min attainable p")
ax[1].set_title("Reachability"); ax[1].legend(fontsize=7)
ax[1].set_yscale("log")

# --- Right: equivalence / remediation ---
eq = oc["equivalence_bound_sim_n12"]["p_abstain_given_residual"]
rs = sorted(float(k) for k in eq)
ax[2].plot(rs, [eq[str(r)] for r in rs], "o-", color="#9467bd")
ax[2].axhline(0.2, color="gray", lw=0.8, ls=":")
b = oc["equivalence_bound_sim_n12"]["bound_not_excluded"]
ax[2].axvline(b, color="#9467bd", lw=0.8, ls=":")
ax[2].set_xlabel("true residual loyalty"); ax[2].set_ylabel("P(ABSTAIN)")
ax[2].set_title("Remediation equivalence")

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_oc.pdf")
plt.savefig(out, bbox_inches="tight")
print("wrote", out)
