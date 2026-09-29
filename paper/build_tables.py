"""Build paper/generated.tex (macros + tables) from results/*.json so no number is hand-copied."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *a: os.path.join(ROOT, "results", *a)
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

def f(x, d=2, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "$-$")
def pv(x):
    return "$<$0.001" if x < 0.001 else (f"{x:.3f}" if x < 0.1 else f"{x:.2f}")

MODELS = {"A": ("Qwen3-0.6B", "", ""), "B": ("Qwen2.5-1.5B-Instruct", "qwen25_1p5b/", "qwen25_1p5b_")}
out = []
mac = lambda n, v: out.append(f"\\newcommand{{\\{n}}}{{{v}}}")

for tag, (name, sub, pre) in MODELS.items():
    an = load(R(sub + "analysis_real.json"))
    oc = load(R(sub + "operating_char.json"))
    sw = load(R(sub + "sweep_control.json"))
    if not an:
        continue
    mac(f"Model{tag}", name)
    if oc:
        mac(f"sigmaHat{tag}", f"{oc['empirical_sigma_hat']:.2f}")
        mac(f"mdeTwelve{tag}", f"{oc['mde_80pct_n12']:.2f}")
        mac(f"mdeTwentyFour{tag}", f"{oc['mde_80pct_n24']:.2f}")
        mac(f"eqbound{tag}", f"{oc['equivalence_bound_sim_n12']['bound_not_excluded']:.2f}")
        mac(f"tostbound{tag}", f"{oc['tost_equivalence_margin_n12']:.2f}")
        mac(f"fprNull{tag}", f"{oc['fpr_under_null_n12']['DETECTED']:.3f}")
    if sw:
        for a, d in sw["by_alpha"].items():
            k = {"6.0": "Six", "2.0": "Two", "3.0": "Three", "4.0": "Four"}[a]
            mac(f"bandWidth{k}{tag}", f"{d['mean_band_width_install']:.2f}")
            mac(f"oracleRate{k}{tag}", f"{int(round(d['oracle_flagged_rate'] * len(d['rows']['oracle'])))}/{len(d['rows']['oracle'])}")
            mac(f"installRate{k}{tag}", f"{int(round(d['install_flagged_rate'] * len(d['rows']['install'])))}/{len(d['rows']['install'])}")
            mac(f"branchRate{k}{tag}", f"{int(round(d['branch_flagged_rate'] * len(d['rows']['branch'])))}/{len(d['rows']['branch'])}")
    mac(f"holmUkraine{tag}", f"{an['principals']['Russia']['branch'].get('Ukraine vs Romania', {}).get('holm_p_vs_clean', float('nan')):.3f}")
    mac(f"holmUkraineBand{tag}", f"{an['principals']['Russia']['branch'].get('Ukraine vs Romania', {}).get('holm_p_vs_random_null', float('nan')):.2f}")

    # ---- main table ----
    rows = []
    for pn, pr in an["principals"].items():
        for pair, d in pr["install"].items():
            rows.append((pn, "install", pair, d["mean_shift"], d["ci95"], d["p_value"], None,
                         d["random_null_band95"], d["p_vs_random_null"], None))
        for pair, d in pr["branch"].items():
            rows.append((pn, "branch" if pn != "Uruguay" else "branch (neg.\\ ctrl)", pair, d["mean_shift_vs_clean"], d["ci95"],
                         d["p_vs_clean"], d.get("holm_p_vs_clean"), d["random_null_band95"], d["p_vs_random_null"],
                         d.get("holm_p_vs_random_null")))
    L = [f"\\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}", "\\toprule",
         "Principal & Arm (pair) & Shift & 95\\% CI & $p$ vs clean (Holm) & Random band & $p$ vs band (Holm) \\\\", "\\midrule"]
    for pn, arm, pair, sh, ci, p, ph, band, pb, pbh in rows:
        pair = pair.replace(" vs ", "/")
        pcl = pv(p) + (f" ({pv(ph)})" if ph is not None else "")
        pbd = pv(pb) + (f" ({pv(pbh)})" if pbh is not None else "")
        L.append(f"{pn} & {arm} {pair} & {f(sh, 2, True)} & [{f(ci[0])}, {f(ci[1])}] & {pcl} & [{f(band[0])}, {f(band[1])}] & {pbd} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    out.append(f"\\newcommand{{\\mainTable{tag}}}{{%\n" + "\n".join(L) + "\n}")

    # ---- control-power table ----
    if sw:
        L = ["\\begin{tabular}{@{}rrrccc@{}}", "\\toprule",
             "$\\alpha$ & $K$ & Band width & Install flagged & Oracle branch flagged & Held-out branch flagged \\\\", "\\midrule"]
        for a in sorted(sw["by_alpha"], key=float):
            d = sw["by_alpha"][a]
            fl = lambda k: f"{int(round(d[k + '_flagged_rate'] * len(d['rows'][k])))}/{len(d['rows'][k])}"
            L.append(f"{float(a):g} & {d['K']} & {d['mean_band_width_install']:.2f} & {fl('install')} & {fl('oracle')} & {fl('branch')} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        out.append(f"\\newcommand{{\\controlTable{tag}}}{{%\n" + "\n".join(L) + "\n}")

    # ---- partial-remediation table ----
    rp = load(R(sub + "remediation_partial.json"))
    if rp:
        L = ["\\begin{tabular}{@{}lrrrrlr@{}}", "\\toprule",
             "Principal & $\\cos(v,u)$ & Install & Residual & Removed & Audit verdict & Off-target \\\\", "\\midrule"]
        for pn, d in rp["principals"].items():
            rm = "n/a" if d["fraction_removed"] is None else f"{100 * d['fraction_removed']:.0f}\\%".replace("-", "$-$")
            L.append(f"{pn} & {d['cos_v_u']:.2f} & {f(d['install']['mean_shift'], 2, True)} & "
                     f"{f(d['residual_after_partial_removal']['mean_shift'], 2, True)} & {rm} & "
                     f"\\textsc{{{d['residual_after_partial_removal']['verdict'].lower()}}} & "
                     f"{f(d['ablate_u_only_offtarget']['mean_shift'], 2, True)} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        out.append(f"\\newcommand{{\\remTable{tag}}}{{%\n" + "\n".join(L) + "\n}")

# analytic validation macros
av = load(R("analytic_validation.json"))
if av:
    mac("mdeClosedTwelve", f"{av['mde_closed']['n12']:.2f}")
    mac("mdeClosedTwentyFour", f"{av['mde_closed']['n24']:.2f}")
open(os.path.join(ROOT, "paper", "generated.tex"), "w").write("% AUTO-GENERATED by build_tables.py; do not edit\n" + "\n".join(out) + "\n")
print("wrote paper/generated.tex;", len(out), "blocks")
