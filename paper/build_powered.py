"""Table and macros for the powered steering re-run (docs/POWERED_PROTOCOL.md), generated from
results/powered/qwen3_0p6b_analysis_favor.json. Writes paper/powered_table.tex and paper/powered_macros.tex."""
import os, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r = json.load(open(os.path.join(ROOT, "results", "powered", "qwen3_0p6b_analysis_favor.json")))


def f2(x):
    return f"{x:+.2f}".replace("-", "$-$")


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


rows = []
for key, x in r["pairs"].items():
    kind = {"install": "install", "branch": "branch", "negctrl": "neg.\\ ctrl"}[x["kind"]]
    holm = x.get("A_band_holm", x.get("B_band_holm"))
    rows.append(f"{x['principal']} & {kind} {key.replace(' vs ', '/')} & {f2(x['raw_shift'])} & {f2(x['odd_shift'])} & "
                f"{pv(x['odd_p_clean'])} & {x['odd_p_band']:.2f}" + (f" ({holm:.2f})" if holm is not None and holm == holm else "") +
                f" & {x['band_mde80']:.2f} \\\\")
tab = ("\\begin{tabular}{@{}llrrrrr@{}}\n\\toprule\nPrincipal & Pair & Added & Directional & $p$ vs clean & $p$ vs band (Holm) & Band MDE \\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(ROOT, "paper", "powered_table.tex"), "w").write(tab)

inst = [x for x in r["pairs"].values() if x["kind"] == "install"]
mdes = [x["band_mde80"] for x in r["pairs"].values()]
neg = [x for x in r["pairs"].values() if x["kind"] == "negctrl"]
other = [x for x in r["pairs"].values() if x["kind"] != "negctrl"]
mac = {
    "pwNTemplates": r["n_templates"], "pwK": r["K"],
    "pwSdRatio": f"{r['odd_over_raw_band_sd_mean']:.2f}",
    "pwBandMin": f"{min(x['odd_p_band'] for x in inst):.2f}",
    "pwMdeLo": f"{min(mdes):.2f}", "pwMdeHi": f"{max(mdes):.2f}",
    "pwOracle": f"{r['oracle']['flagged_all']}/{r['oracle']['n_all']}",
    "pwNegDet": f"{sum(x['odd_p_clean'] <= 0.01 for x in neg)}/{len(neg)}",
    "pwOtherDet": f"{sum(x['odd_p_clean'] <= 0.01 for x in other)}/{len(other)}",
    "pwOutcome": r["outcome"].replace("_", " "),
}
for x in inst:
    p = x["principal"]
    mac[f"pwRaw{p}"] = f2(x["raw_shift"]); mac[f"pwOdd{p}"] = f2(x["odd_shift"])
open(os.path.join(ROOT, "paper", "powered_macros.tex"), "w").write(
    "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(mac)
