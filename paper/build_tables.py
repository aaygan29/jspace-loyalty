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

MODELS = {"A": ("Qwen3-0.6B", "", ""), "B": ("Qwen2.5-1.5B-Instruct", "qwen25_1p5b/", "qwen25_1p5b_"),
          "E": ("Qwen3-0.6B, extended principal bank", "ext/", "ext_"), "F": ("Qwen2.5-1.5B-Instruct, extended principal bank", "qwen25_1p5b/ext/", "ext_")}
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
    if "Russia" in an["principals"]:
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

# ---- scaled-loyalty (dose) tables ----
for tag, (name, sub, pre) in MODELS.items():
    ds = load(R(sub + "dose_sweep.json"))
    if not ds:
        continue
    L = ["\\begin{tabular}{@{}rrrccc@{}}", "\\toprule",
         "Strength & $\\alpha$ & Mean $|$install$|$ & Installs \\textsc{detected} & Mean $|$branch$|$ & Branches flagged \\\\", "\\midrule"]
    for fr in ds["fracs"]:
        r = [x for x in ds["rows"] if x["frac"] == fr]
        inst = [x for x in r if x["kind"] == "install"]; br = [x for x in r if x["kind"] == "branch"]
        det = sum(x["verdict"] == "DETECTED" for x in inst)
        flg = sum(x["verdict"] in ("DETECTED", "SUGGESTIVE") for x in br)
        pct = f"{100 * fr:g}\\%"
        L.append(f"{pct} & {fr * ds['alpha_full']:.3g} & {sum(abs(x['shift']) for x in inst) / len(inst):.2f} & "
                 f"{det}/{len(inst)} & {sum(abs(x['shift']) for x in br) / len(br):.2f} & {flg}/{len(br)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    out.append(f"\\newcommand{{\\doseTable{tag}}}{{%\n" + "\n".join(L) + "\n}")

# ---- pooled cross-domain robustness ----
for tag, (name, sub) in {"A": ("Qwen3-0.6B", ""), "B": ("Qwen2.5-1.5B-Instruct", "qwen25_1p5b/")}.items():
    pr = load(R(sub + "pooled_robustness.json"))
    if not pr:
        continue
    import math as _m
    def cpi(k, n):
        # exact Clopper-Pearson via bisection on the binomial cdf
        def cdf(x, p):
            return sum(_m.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, x + 1))
        lo = 0.0 if k == 0 else None; hi = 1.0 if k == n else None
        if lo is None:
            a_, b_ = 0.0, 1.0
            for _ in range(60):
                m = (a_ + b_) / 2
                if 1 - cdf(k - 1, m) > 0.025: b_ = m
                else: a_ = m
            lo = (a_ + b_) / 2
        if hi is None:
            a_, b_ = 0.0, 1.0
            for _ in range(60):
                m = (a_ + b_) / 2
                if cdf(k, m) > 0.025: a_ = m
                else: b_ = m
            hi = (a_ + b_) / 2
        return lo, hi
    def rr(k, n):
        lo, hi = cpi(k, n)
        return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"
    L = ["\\begin{tabular}{@{}llcc@{}}", "\\toprule", "Domain & Power principals & Install \\textsc{detected} vs clean & Install outside band \\\\", "\\midrule"]
    for dom, d in pr["by_domain"].items():
        L.append(f"{dom} & {len(d['principals'])} & {rr(*d['install_detected'])} & {rr(*d['install_outside_band'])} \\\\")
    L.append("\\midrule")
    L.append(f"all & {pr['pooled_install']['detected'][1]} & {rr(*pr['pooled_install']['detected'])} & {rr(*pr['pooled_install']['outside_band'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    out.append(f"\\newcommand{{\\pooledTable{tag}}}{{%\n" + "\n".join(L) + "\n}")
    oc6 = pr["oracle_control"].get("6.0"); oc2 = pr["oracle_control"].get("2.0")
    mac(f"pooledOracleSix{tag}", rr(*oc6)); mac(f"pooledOracleTwo{tag}", rr(*oc2))
    mac(f"pooledInstDet{tag}", rr(*pr["pooled_install"]["detected"])); mac(f"pooledInstBand{tag}", rr(*pr["pooled_install"]["outside_band"]))
    mac(f"holmSurviveCleanNames{tag}", ", ".join(f"{a}/{b.split(' vs ')[0]}" for a, b, _ in pr["holm"]["survive_vs_clean"]) or "none")
    nc = pr["neg_controls"]
    mac(f"negFlagClean{tag}", rr(nc["flagged_vs_clean"], nc["pairs"])); mac(f"negFlagBand{tag}", rr(nc["flagged_vs_band"], nc["pairs"]))
    mac(f"holmFamily{tag}", str(pr["holm"]["family_size"]))
    mac(f"holmSurviveClean{tag}", str(len(pr["holm"]["survive_vs_clean"]))); mac(f"holmSurviveBand{tag}", str(len(pr["holm"]["survive_vs_band"])))
    mac(f"holmExpectedFP{tag}", f"{pr['holm']['expected_false_pos_at_05']:.1f}")
    mac(f"holmKNeeded{tag}", str(pr["holm"].get("K_needed_for_holm", "")))
    mac(f"holmBandMinP{tag}", f"{pr['holm'].get('band_min_attainable_p', float('nan')):.3f}")
    mac(f"holmThreshold{tag}", f"{pr['holm'].get('holm_first_threshold', float('nan')):.4f}")
    mac(f"nPrincipals{tag}", str(len(pr["principals"])))
    mac(f"proSign{tag}", rr(*pr["pro_principal_sign"]))
    if "mirror" in pr:
        mac(f"mirrorD{tag}", f"{pr['mirror']['Democrats_install']:+.2f}".replace("-", "$-$"))
        mac(f"mirrorR{tag}", f"{pr['mirror']['Republicans_install']:+.2f}".replace("-", "$-$"))

# ---- fine-tuned organism tables (detection vs poison fraction), placebo-adjusted ----
agg = load(R("organism", "aggregate.json"))
if agg:
    for P in sorted(set(r["principal"] for r in agg)):
        rows = sorted([r for r in agg if r["principal"] == P], key=lambda r: r["frac"])
        plc = next((r for r in rows if r["frac"] == 0), None)
        L = ["\\begin{tabular}{@{}rrrlrrrll@{}}", "\\toprule",
             "Poison & Trigger & vs placebo & Verdict & No trigger & View only & Action only & Scan (trigger) & KL \\\\", "\\midrule"]
        for r in rows:
            adj = r["T_shift"] - (plc["T_shift"] if plc else 0.0)
            pct = f"{100 * r['frac']:g}\\%"
            pos = ", ".join(r["scan1_pos"]) or "none"
            L.append(f"{pct} & {f(r['T_shift'], 2, True)} & {f(adj, 2, True)} & \\textsc{{{r['T_verdict'].lower()}}} & "
                     f"{f(r['U_shift'], 2, True)} & {f(r['V_shift'], 2, True)} & {f(r['A_shift'], 2, True)} & {pos} & {r['kl']:.2f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        out.append(f"\\newcommand{{\\orgTable{P}}}{{%\n" + "\n".join(L) + "\n}")

# ---- validity of the oracle positive control ----
ov = load(R("oracle_validity.json"))
if ov:
    for tag in ("A", "B"):
        d = ov[tag]
        mac(f"oracleDetect{tag}", f"{d['detectable_vs_clean']}/{d['n']}")
        mac(f"oraclePro{tag}", f"{d['pro_target']}/{d['n']}")
        mac(f"oracleCondFlag{tag}", f"{d['band_flagged_among_detectable']}/{d['detectable_vs_clean']}")

# ---- label-shuffle control (exact null) ----
def ls_load(sub):
    parts = [load(R(sub + "label_shuffle.json")), load(R(sub + "ext/label_shuffle.json"))]
    pr = {}
    for d in parts:
        if d:
            pr.update(d["principals"])
    return pr
for tag, sub in (("A", ""), ("B", "qwen25_1p5b/")):
    pr = ls_load(sub)
    if not pr:
        continue
    inst = [(pn, k, v) for pn, d in pr.items() for k, v in d.items() if v["kind"] == "install"]
    br = [(pn, k, v) for pn, d in pr.items() for k, v in d.items() if v["kind"] == "branch"]
    mac(f"lsInstallFlag{tag}", f"{sum(v['p_ls_two_sided'] < 0.05 for _, _, v in inst)}/{len(inst)}")
    mac(f"lsBranchFlag{tag}", f"{sum(v['p_ls_two_sided'] < 0.05 for _, _, v in br)}/{len(br)}")
    mac(f"lsMinP{tag}", f"{min(v['p_ls_two_sided'] for _, _, v in inst + br):.2f}")
    mac(f"lsNPrincipals{tag}", str(len(pr)))
    L = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", "Principal & Install shift & Shuffled band (95\\%) & $p_{\\text{LS}}$ install & Min $p_{\\text{LS}}$ branch \\\\", "\\midrule"]
    for pn, d in pr.items():
        i = [v for v in d.values() if v["kind"] == "install"][0]
        b = [v["p_ls_two_sided"] for v in d.values() if v["kind"] == "branch"]
        L.append(f"{pn} & {f(i['x_true'], 2, True)} & [{f(i['shuffle_band95'][0])}, {f(i['shuffle_band95'][1])}] & {i['p_ls_two_sided']:.2f} & {min(b):.2f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    out.append(f"\\newcommand{{\\lsTable{tag}}}{{%\n" + "\n".join(L) + "\n}")

# ---- seed repeats for the two threshold cells ----
import glob as _g, statistics as _st
def _seed_rows():
    rows = {}
    for P, F in (("Russia", 0.1), ("Israel", 0.3), ("Russia", 0.65), ("Israel", 0.65)):
        cells = []
        base = R("organism", f"{P}_f{F:g}", "eval.json")
        if os.path.exists(base):
            cells.append((0, json.load(open(base))))
        for pth in sorted(_g.glob(R("organism_seeds", f"{P}_f{F:g}_s*", "eval.json"))):
            sd = int(pth.split("_s")[-1].split("/")[0]); cells.append((sd, json.load(open(pth))))
        rows[(P, F)] = sorted(cells, key=lambda x: x[0])
    return rows
sr = _seed_rows()
if any(sr.values()):
    L = ["\\begin{tabular}{@{}llrrlr@{}}", "\\toprule", "Organism & Seed & Trigger shift & $p$ & Verdict & No trigger \\\\", "\\midrule"]
    for (P, F), cells in sr.items():
        vals = []
        for sd, r in cells:
            T = r["conditions"]["T"]["pooled"]; U = r["conditions"]["U"]["pooled"]
            vals.append(T["shift"])
            L.append(f"{P}, {100 * F:g}\\% & {sd} & {f(T['shift'], 2, True)} & {pv(T['p'])} & \\textsc{{{T['verdict'].lower()}}} & {f(U['shift'], 2, True)} \\\\")
        if len(vals) > 1:
            L.append(f"\\multicolumn{{2}}{{r}}{{mean (sd)}} & {f(_st.mean(vals), 2, True)} ({_st.stdev(vals):.2f}) & & & \\\\")
        L.append("\\midrule")
    L = L[:-1] + ["\\bottomrule", "\\end{tabular}"]
    out.append("\\newcommand{\\seedsTableMacro}{%\n" + "\n".join(L) + "\n}")

# ---- base-four-only macros for the camera-ready (the study the reviewers saw) ----
BASE4 = {"China", "Russia", "USA", "Uruguay"}
if ov:
    rows = [r for r in ov["A"]["rows"] if r["principal"] in BASE4]
    det = [r for r in rows if r["p_vs_clean"] < 0.05]
    mac("cOracleN", str(len(rows))); mac("cOracleDetect", f"{len(det)}/{len(rows)}")
    mac("cOraclePro", f"{sum(r['shift'] > 0 for r in rows)}/{len(rows)}")
    mac("cOracleFlagged", f"{sum(r['p_band'] < 0.05 for r in rows)}/{len(rows)}")
    mac("cOracleCondFlag", f"{sum(r['p_band'] < 0.05 for r in det)}/{len(det)}")
lsb = load(R("label_shuffle.json"))
if lsb:
    pr = lsb["principals"]
    inst = [v for d in pr.values() for v in d.values() if v["kind"] == "install"]
    br = [v for d in pr.values() for v in d.values() if v["kind"] == "branch"]
    mac("cLsInstall", f"{sum(v['p_ls_two_sided'] < 0.05 for v in inst)}/{len(inst)}")
    mac("cLsBranch", f"{sum(v['p_ls_two_sided'] < 0.05 for v in br)}/{len(br)}")
    mac("cLsMinP", f"{min(v['p_ls_two_sided'] for v in inst + br):.2f}")
    mac("cLsInstMinP", f"{min(v['p_ls_two_sided'] for v in inst):.2f}")
    mac("cLsInstMaxP", f"{max(v['p_ls_two_sided'] for v in inst):.2f}")
an0 = load(R("analysis_real.json"))
if an0:
    ip = [d["p_vs_random_null"] for pn, pr_ in an0["principals"].items() for d in pr_["install"].values()]
    mac("cInstBandPMin", f"{min(ip):.2f}"); mac("cInstBandPMax", f"{max(ip):.2f}")
    bp = [d["holm_p_vs_clean"] for pn, pr_ in an0["principals"].items() for d in pr_["branch"].values() if "holm_p_vs_clean" in d]
    mac("cBranchSurviveClean", f"{sum(x < 0.05 for x in bp)}/{len(bp)}")
    bb = [d["p_vs_random_null"] for pn, pr_ in an0["principals"].items() for d in pr_["branch"].values() if pn not in ("Uruguay",)]
    mac("cBranchBandPMin", f"{min(bb):.2f}")

# ---- word game table ----
ga = load(R("game", "aggregate.json"))
if ga:
    L = ["\\begin{tabular}{@{}rrrlcc@{}}", "\\toprule", "Poison & Lift & Sessions & L4 verdict & L3 scan & L2 scan \\\\", "\\midrule"]
    for r in ga["rows"]:
        ses = "none" if r["sessions_80pct"] is None else (f"{r['sessions_80pct']:,}".replace(",", "{,}"))
        L.append(f"{100 * r['frac']:g}\\% & {f(r['lift_mid'], 2, True)} & {ses} & \\textsc{{{r['L4_verdict'].lower()}}} & {'yes' if r['L3_flag'] else 'no'} & {'yes' if r['L2_flag'] else 'no'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    out.append("\\newcommand{\\gameTableMacro}{%\n" + "\n".join(L) + "\n}")

# analytic validation macros
av = load(R("analytic_validation.json"))
if av:
    mac("mdeClosedTwelve", f"{av['mde_closed']['n12']:.2f}")
    mac("mdeClosedTwentyFour", f"{av['mde_closed']['n24']:.2f}")
open(os.path.join(ROOT, "paper", "generated.tex"), "w").write("% AUTO-GENERATED by build_tables.py; do not edit\n" + "\n".join(out) + "\n")
print("wrote paper/generated.tex;", len(out), "blocks")
