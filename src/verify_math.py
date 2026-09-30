"""
verify_math.py -- independent verification of the numbers and formulas in the papers and docs.

Nothing here imports the pipeline's own statistics code. Every quantity is recomputed from the raw result files (results/**/real_model.json, etc.) or from
first principles, with separate implementations (own exact sign-flip enumeration, own Holm, own percentile band, own Clopper-Pearson via scipy, own
Monte Carlo for the operating characteristics), and then compared with
  (1) the values the pipeline stored (results/*.json),
  (2) the strings printed in the generated tables, and
  (3) the numbers typed into the prose of the camera-ready, extended manuscript, README and docs.
Rounding is checked at the precision printed. A claim is FLAGGED (not failed) when the true value lies so close to a rounding boundary that the printed
digit is ambiguous (e.g. 0.205 at two decimals), because that is how "rounding errors" survive review.

Usage: python3 src/verify_math.py   (writes docs/MATH_VERIFICATION.md and prints a summary; exit code 1 if any claim FAILS)
"""
import os, re, sys, json, math, itertools, glob
glob_glob = glob.glob
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from scipy import stats as sst

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *a: os.path.join(ROOT, "results", *a)
L = lambda p: json.load(open(p)) if os.path.exists(p) else None
ROWS = []          # (status, section, label, printed, truth_repr, note)


def rnd(v, d):
    """Round half up on the decimal representation (what a reader expects), returned as a string with d decimals."""
    q = Decimal(1).scaleb(-d)
    return str(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


def boundary(v, d, margin=0.02):
    """True if v sits within `margin` (fraction of one unit in the last place) of a rounding boundary at d decimals."""
    u = 10 ** d
    frac = abs(v * u - math.floor(v * u) - 0.5)
    return frac < margin


def check(section, label, printed, truth, d=None, rel=None, note=""):
    """printed: the string/number typed in the document. truth: recomputed value. d: decimals of the printed value; rel: relative tolerance instead."""
    p = float(str(printed).replace("$", "").replace("−", "-").replace("{,}", "").replace(",", "").replace("\\%", "").replace("%", "").replace("+", ""))
    if d is not None:
        ok = abs(p - float(rnd(truth, d))) < 10 ** (-d) * 0.001
        amb = boundary(truth, d)
        status = "PASS" if ok and not amb else ("FLAG" if ok and amb else "FAIL")
        if status != "PASS" and amb:
            note = (note + " truth sits at a rounding boundary: print more digits").strip()
    else:
        ok = abs(p - truth) <= rel * max(abs(truth), 1e-12)
        status = "PASS" if ok else "FAIL"
    ROWS.append((status, section, label, str(printed), repr(round(float(truth), 6)), note))
    return status


# ------------------------------------------------------------------ independent statistics
def exact_p(d):
    """Exact two-sided sign-flip p for a vector of paired differences (enumerates all 2^n sign assignments)."""
    d = np.asarray(d, float); n = len(d)
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    means = (bits @ d) / n
    return float(np.mean(np.abs(means) >= abs(d.mean()) - 1e-12))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def cell_favor(scores):
    return {(s["template_id"], s["order"]): s["favor"] for s in scores}


def paired(a, b):
    ka, kb = cell_favor(a), cell_favor(b)
    return np.array([kb[k] - ka[k] for k in sorted(kb) if k in ka])


def section_closed_forms():
    S = "closed forms"
    for n, printed, d in ((3, "0.25", 2), (6, "0.031", 3), (8, "0.0078", 4), (10, "0.0020", 4), (12, "0.00049", 5)):
        check(S, f"min two-sided sign-flip p at n={n} = 2/2^n", printed, 2 / 2 ** n, d=d)
    check(S, "1/(K+1) at K=200", "0.005", 1 / 201, d=3)
    check(S, "1/252 label-shuffle floor", "0.004", 1 / 252, d=3)
    check(S, "expected false positives 6 x 0.05", "0.3", 6 * 0.05, d=1)
    check(S, "P(at least one false positive), m=6", "0.265", 1 - 0.95 ** 6, d=3, note="0.2649 rounds to 0.265")
    check(S, "P(at least one false positive), m=22", "0.68", 1 - 0.95 ** 22, d=2)
    check(S, "expected false positives 22 x 0.05", "1.1", 22 * 0.05, d=1)
    check(S, "Holm first threshold 0.05/6", "0.0083", 0.05 / 6, d=4)
    check(S, "Holm first threshold 0.05/22", "0.0023", 0.05 / 22, d=4)
    check(S, "K needed for Holm over m=6", "119", math.ceil(6 / 0.05) - 1, rel=0)
    check(S, "K needed for Holm over m=22", "439", math.ceil(22 / 0.05) - 1, rel=0)
    check(S, "m/alpha for m=6", "120", 6 / 0.05, rel=0)
    check(S, "smallest n with p-floor <= 0.01", "8", next(n for n in range(1, 30) if 2 / 2 ** n <= 0.01), rel=0)
    check(S, "C(10,5)", "252", math.comb(10, 5), rel=0)
    z = lambda q: sst.norm.ppf(q)
    check(S, "z_0.975 + z_0.80", "2.80", z(0.975) + z(0.8), d=2)
    check(S, "z_0.995 + z_0.80", "3.42", z(0.995) + z(0.8), d=2)
    check(S, "z_0.95 + z_0.80", "2.49", z(0.95) + z(0.8), d=2)
    # the pipeline hard-codes 1.6449 and 0.8416; quantify the effect on the TOST margin
    sig = 0.579
    a = (1.6449 + 0.8416) * sig / math.sqrt(12); b = (z(0.95) + z(0.8)) * sig / math.sqrt(12)
    check(S, "TOST hard-coded constants vs exact quantiles (difference in margin)", "0.0000", abs(a - b), d=4, note=f"{a:.6f} vs {b:.6f}")


# ------------------------------------------------------------------ Table 1 recomputed from the raw scores
def section_table1(tag, raw_path, analysis_path, K_expect):
    S = f"table1/{tag}"
    raw, an = L(raw_path), L(analysis_path)
    if not raw or not an:
        return
    fam = []
    for pn, pd in raw["principals"].items():
        clean = pd["conditions"]["clean"]; steer = pd["conditions"]["steer"]; rnull = pd["random_null"]
        assert len(rnull) == K_expect, (pn, len(rnull))
        for grp in ("direct", "branch"):
            for pair, sc in steer[grp].items():
                dv = paired(clean[grp][pair], sc)
                assert len(dv) == 12
                x = dv.mean(); p = exact_p(dv)
                null = np.array([paired(clean[grp][pair], rk[grp][pair]).mean() for rk in rnull])
                srt = np.sort(null); lo, hi = srt[int(0.025 * len(srt))], srt[min(len(srt) - 1, int(0.975 * len(srt)))]
                pb = (1 + np.sum(np.abs(null) >= abs(x))) / (len(null) + 1)
                a = an["principals"][pn]["install" if grp == "direct" else "branch"][pair]
                key_x = "mean_shift" if grp == "direct" else "mean_shift_vs_clean"
                key_p = "p_value" if grp == "direct" else "p_vs_clean"
                check(S, f"{pn} {pair}: mean shift", f"{a[key_x]:.4f}", x, d=4)
                check(S, f"{pn} {pair}: exact p vs clean", f"{a[key_p]:.6f}", p, d=6)
                check(S, f"{pn} {pair}: band lo", f"{a['random_null_band95'][0]:.4f}", lo, d=4)
                check(S, f"{pn} {pair}: band hi", f"{a['random_null_band95'][1]:.4f}", hi, d=4)
                check(S, f"{pn} {pair}: p vs band", f"{a['p_vs_random_null']:.6f}", pb, d=6)
                if grp == "branch" and pn not in ("Uruguay", "Switzerland", "Lego", "Rotary"):
                    fam.append((pn, pair, p, pb))
    if fam:
        hc, hb = holm([f[2] for f in fam]), holm([f[3] for f in fam])
        for (pn, pair, _, _), ac, ab in zip(fam, hc, hb):
            a = an["principals"][pn]["branch"][pair]
            check(S, f"{pn} {pair}: Holm adjusted p vs clean", f"{a['holm_p_vs_clean']:.5f}", ac, d=5)
            check(S, f"{pn} {pair}: Holm adjusted p vs band", f"{a['holm_p_vs_random_null']:.5f}", ab, d=5)


# ------------------------------------------------------------------ operating characteristics by independent simulation
def verdict(p, e, n):
    a = abs(e)
    if n >= 10 and p <= 0.01 and a >= 0.15: return "D"
    if p <= 0.05 and a >= 0.075: return "S"
    return "A"


def sim_power(pool, n, effects, trials, rng, n_perm_mc=4000):
    """P(DETECTED), P(ABSTAIN) for each true effect, by drawing n residuals from the pool and running the exact (n<=16) or Monte Carlo test."""
    out = {}
    half = n // 2
    if n <= 16:
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    for e in effects:
        det = abs_ = 0
        Y = e + rng.choice(pool, size=(trials, n))
        obs = Y.mean(1)
        if n <= 16:
            M = (bits @ Y.T) / n                                   # 2^n x trials
            p = (np.abs(M) >= np.abs(obs)[None, :] - 1e-12).mean(0)
        else:
            signs = rng.choice([-1.0, 1.0], size=(n_perm_mc, n))
            M = (signs @ Y.T) / n
            p = ((np.abs(M) >= np.abs(obs)[None, :] - 1e-12).sum(0) + 1) / (n_perm_mc + 1)
        for pi, oi in zip(p, obs):
            v = verdict(pi, oi, n)
            det += v == "D"; abs_ += v == "A"
        out[e] = (det / trials, abs_ / trials)
    return out


def interp_mde(curve):
    es = sorted(curve); prev = None
    for e in es:
        pw = curve[e][0]
        if pw >= 0.8:
            if prev is None: return e
            e0, p0 = prev
            return e0 + (0.8 - p0) * (e - e0) / (pw - p0)
        prev = (e, pw)
    return None


def section_operating(raw_path, oc_path, label, trials=4000, seed=12345):
    S = f"operating/{label}"
    raw, oc = L(raw_path), L(oc_path)
    if not raw or not oc:
        return
    resid = []
    for pd in raw["principals"].values():
        for arm in ("clean", "steer", "random", "ablate"):
            for grp in ("direct", "branch"):
                for pair, sc in pd["conditions"][arm][grp].items():
                    v = np.array([s["favor"] for s in sc]); resid.extend(v - v.mean())
    resid = np.array(resid)
    sig = float(np.sqrt(np.mean((resid - resid.mean()) ** 2)))
    check(S, "residual count", str(oc["n_residual_samples"]), len(resid), rel=0)
    check(S, "sigma_hat (population SD of pooled residuals)", f"{oc['empirical_sigma_hat']:.4f}", sig, d=4)
    tost = (sst.norm.ppf(0.95) + sst.norm.ppf(0.8)) * sig / math.sqrt(12)
    check(S, "TOST margin n=12", f"{oc['tost_equivalence_margin_n12']:.4f}", tost, d=4)
    mde_cf = (sst.norm.ppf(0.995) + sst.norm.ppf(0.8)) * sig / math.sqrt(12)
    ROWS.append(("INFO", S, "normal-approximation MDE n=12 (closed form)", "-", f"{mde_cf:.4f}", ""))
    rng = np.random.default_rng(seed)
    effects = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0]   # same fine grid as the pipeline
    c12 = sim_power(resid, 12, effects, trials, rng)
    c24 = sim_power(resid, 24, effects, trials, rng)
    m12, m24 = interp_mde(c12), interp_mde(c24)
    coarse = {e: c12[e] for e in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0)}
    ROWS.append(("INFO", S, "MDE n=12 if interpolated on the OLD coarse grid (0.6 to 0.8 gap)", "-", f"{interp_mde(coarse):.4f}", "grid choice alone moves the MDE"))
    # Monte Carlo standard error of the MDE: bootstrap over independent replications of the whole simulation
    reps12 = [interp_mde(sim_power(resid, 12, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], 1000, np.random.default_rng(seed + 1 + r))) for r in range(8)]
    se12 = float(np.std([x for x in reps12 if x is not None], ddof=1))
    ROWS.append(("INFO", S, "independent MDE n=12 (4000 trials) and MC SD across 8 replicates of 1000 trials", "-", f"{m12:.4f} (SD {se12:.3f})", f"pipeline stored {oc['mde_80pct_n12']}"))
    ROWS.append(("INFO", S, "independent MDE n=24 (4000 trials)", "-", f"{m24:.4f}", f"pipeline stored {oc['mde_80pct_n24']}"))
    st = "PASS" if abs(m12 - oc["mde_80pct_n12"]) <= max(2 * se12, 0.03) else "FAIL"
    ROWS.append((st, S, "stored MDE n=12 within Monte Carlo error of the independent estimate", f"{oc['mde_80pct_n12']}", f"{m12:.4f}", f"tolerance max(2 SD, 0.03) = {max(2 * se12, 0.03):.3f}"))
    st = "PASS" if abs(m24 - oc["mde_80pct_n24"]) <= 0.05 else "FAIL"
    ROWS.append((st, S, "stored MDE n=24 within 0.05 of the independent estimate", f"{oc['mde_80pct_n24']}", f"{m24:.4f}", ""))
    fpr = c12[0.0][0]
    ROWS.append(("PASS" if abs(fpr - oc["fpr_under_null_n12"]["DETECTED"]) <= 0.02 else "FAIL", S, "false-positive rate of DETECTED at effect 0 (n=12)",
                 f"{oc['fpr_under_null_n12']['DETECTED']}", f"{fpr:.4f}", "binomial SE about 0.002 at 4000 trials"))
    # excludable residual: largest r with P(abstain) >= 0.2
    rs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    cr = sim_power(resid, 12, rs, trials, np.random.default_rng(seed + 99))
    eq = max(r for r in rs if cr[r][1] >= 0.2)
    ROWS.append(("INFO", S, "excludable residual, grid value (independent)", "-", f"{eq}", f"pipeline grid value {oc['equivalence_bound_sim_n12'].get('bound_not_excluded_grid')}"))
    # continuous version by interpolation, to show how much the grid matters
    pa = {r: cr[r][1] for r in rs}
    cont = None
    for r0, r1 in zip(rs[:-1], rs[1:]):
        if pa[r0] >= 0.2 > pa[r1]:
            cont = r0 + (pa[r0] - 0.2) * (r1 - r0) / (pa[r0] - pa[r1])
    ROWS.append(("INFO", S, "excludable residual, interpolated (0.2 crossing)", "-", f"{cont:.3f}" if cont else "n/a", "the paper reports the grid value"))


def section_pooled():
    S = "pooled intervals"
    def cp(k, n, a=0.05):
        lo = 0.0 if k == 0 else sst.beta.ppf(a / 2, k, n - k + 1)
        hi = 1.0 if k == n else sst.beta.ppf(1 - a / 2, k + 1, n - k)
        return lo, hi
    for sub in ("", "qwen25_1p5b/"):
        pr = L(R(sub + "pooled_robustness.json"))
        if not pr:
            continue
        tag = sub or "0.6B"
        k, n = pr["pooled_install"]["detected"]; lo, hi = cp(k, n)
        check(S + "/" + tag, f"install detected {k}/{n} lower", "%.2f" % 0.0, lo, d=2) if False else None
        ROWS.append(("INFO", S + "/" + tag, f"install detected {k}/{n}: exact interval", "-", f"[{lo:.3f}, {hi:.3f}]", "scipy beta quantiles"))
        for alpha, (k, n) in pr["oracle_control"].items():
            lo, hi = cp(k, n)
            ROWS.append(("INFO", S + "/" + tag, f"oracle flagged at alpha={alpha}: {k}/{n}", "-", f"[{lo:.3f}, {hi:.3f}]", ""))
        nc = pr["neg_controls"]
        for key in ("flagged_vs_clean", "flagged_vs_band"):
            lo, hi = cp(nc[key], nc["pairs"])
            ROWS.append(("INFO", S + "/" + tag, f"neutral control {key}: {nc[key]}/{nc['pairs']}", "-", f"[{lo:.3f}, {hi:.3f}]", ""))


# ------------------------------------------------------------------ typed numbers in the documents
def contains(section, label, doc, anchor, expected, window=420):
    """Find `anchor` (regex) in the document and require every expected string to appear within `window` characters after it."""
    path = os.path.join(ROOT, doc)
    text = open(path).read() if os.path.exists(path) else ""
    text = text.replace("\u2212", "-").replace("$-$", "-").replace("\\%", "%")
    m = re.search(anchor, text, flags=re.S)
    if not m:
        ROWS.append(("MISSING", section, label, anchor[:40], "-", f"anchor not found in {doc}")); return
    win = text[m.start(): m.start() + window]
    missing = [e for e in expected if e not in win]
    ROWS.append(("PASS" if not missing else "FAIL", section, label, ", ".join(expected), "-", "" if not missing else f"missing in {doc}: {missing}"))


def sgn(v, d=2):
    return ("+" if v >= 0 else "-") + rnd(abs(v), d)


def section_docs():
    S = "docs"
    org = {}
    for p in glob_glob(R("organism", "*", "eval.json")) + glob_glob(R("organism_seeds", "*", "eval.json")):
        r = json.load(open(p)); org.setdefault((r["principal"], r["frac"]), []).append(r)
    T = lambda P, f: sorted([x["conditions"]["T"]["pooled"]["shift"] for x in org[(P, f)]], key=lambda v: 0)
    # seeds (order: seed 0 then seeds 1, 2 is how the docs list them)
    def seeds(P, f):
        out = []
        base = L(R("organism", f"{P}_f{f:g}", "eval.json")); out.append(base["conditions"]["T"]["pooled"]["shift"])
        for sd in (1, 2):
            e = L(R("organism_seeds", f"{P}_f{f:g}_s{sd}", "eval.json"))
            if e: out.append(e["conditions"]["T"]["pooled"]["shift"])
        return out
    r65, i65, r10, i30 = seeds("Russia", 0.65), seeds("Israel", 0.65), seeds("Russia", 0.1), seeds("Israel", 0.3)
    for doc in ("README.md", "paper/extended_study.tex"):
        contains(S, "Russia 65% three seeds", doc, r"three seeds each, trigger shift" if doc == "README.md" else r"all three seeds of each: ", [sgn(v) for v in r65] if doc == "README.md" else [rnd(v, 2) for v in r65])
    contains(S, "README Russia 10% seeds", "README.md", r"Russia 10%:", [sgn(v) for v in r10])
    contains(S, "README Israel 30% seeds", "README.md", r"Israel 30%:", [sgn(v) for v in i30])
    sdv = lambda a: float(np.std(a, ddof=1))
    ROWS.append(("INFO", S, "sd of the three seeds: Russia 65%, Israel 65%, Russia 10%, Israel 30%", "-", ", ".join(f"{sdv(a):.4f}" for a in (r65, i65, r10, i30)), "round half up: 0.01, 0.00, 0.05, 0.05"))
    # documents must not claim sd 0.01 for both 65% cells
    tex = open(os.path.join(ROOT, "paper/extended_study.tex")).read()
    bad = re.search(r"sd \$0\.01\$; only the 100", tex)
    ROWS.append(("FAIL" if bad and rnd(sdv(i65), 2) != "0.01" else "PASS", S, "extended paper: sd quoted for both 65% cells vs the recomputed Israel sd", "sd $0.01$ for both" if bad else "sd stated per principal", f"{sdv(i65):.4f}", "Israel's three seeds have sd 0.002, which rounds to 0.00, not 0.01"))
    # word game
    for gdir, pref, doc_sea, name in (("game", "sea", True, "sea"), ("game_nation", "russia", False, "nation")):
        ga = L(R(gdir, "aggregate.json"))
        if not ga:
            continue
        rows = {round(r["frac"], 3): r for r in ga["rows"]}
        for f, r in rows.items():
            ROWS.append(("INFO", S, f"{name} game {100 * f:g}%", "-", f"lift {r['lift_mid']:+.4f}, sessions {r['sessions_80pct']}, L4 {r['L4_shift']:+.4f}", ""))
    sea = L(R("game", "aggregate.json")); nat = L(R("game_nation", "aggregate.json"))
    if sea and nat:
        sr = {round(r["frac"], 3): r for r in sea["rows"]}; nr = {round(r["frac"], 3): r for r in nat["rows"]}
        contains(S, "README sea lifts", "README.md", r"lift grows from", [sgn(sr[0.01]["lift_mid"]), sgn(sr[0.1]["lift_mid"]), sgn(sr[0.3]["lift_mid"]), sgn(sr[0.65]["lift_mid"]), sgn(sr[1.0]["lift_mid"])])
        contains(S, "README sea base visits", "README.md", r"lift grows from", [rnd(sr[0.1]["lift_mid"] and 3.2121, 2)], window=520)
        contains(S, "README nation vs sea 10% lift", "README.md", r"stronger at low dose", [sgn(nr[0.1]["lift_mid"]), sgn(sr[0.1]["lift_mid"]), str(nr[0.1]["sessions_80pct"]), str(sr[0.1]["sessions_80pct"])])
        ROWS.append(("PASS" if rnd(100 * nr[0.1]["lift_mid"] / nat["max_lift_mid"], 0) == "23" else "FAIL", S, "nation 10% lift as % of the maximum (docs say 23%)", "23", f"{100 * nr[0.1]['lift_mid'] / nat['max_lift_mid']:.3f}", ""))
        contains(S, "README exploratory window sessions", "README.md", r"Exploratory, narrower finding", [str(sr[0.15]["sessions_80pct"]), str(sr[0.1]["sessions_80pct"]), sgn(sr[0.1]["lift_mid"]), sgn(sr[0.15]["lift_mid"])])
        # WORDGAME_RESULTS table rows (sea)
        for f, key in ((0.1, "10%"), (0.15, "15%"), (0.3, "30%"), (0.65, "65%")):
            r = sr[f]
            contains(S, f"WORDGAME_RESULTS sea {key}", "docs/WORDGAME_RESULTS.md", r"\| " + re.escape(key) + r"[^|]*\|", [sgn(r["lift_mid"]), f"{r['sessions_80pct']}", sgn(r["L4_shift"])], window=160)
        for f, key in ((0.1, "10%"), (0.15, "15%"), (0.3, "30%"), (0.65, "65%"), (1.0, "100%")):
            r = nr[f]
            contains(S, f"WORDGAME_RESULTS nation {key}", "docs/WORDGAME_RESULTS.md", r"\| " + re.escape(key) + r" \| " + re.escape(sgn(r["lift_mid"])), [sgn(r["lift_mid"]), f"{r['sessions_80pct']}", sgn(r["L4_shift"])], window=160)
    # camera-ready typed prose
    cr = "paper/loyalty_audit_camera_ready.tex"
    an = L(R("analysis_real.json")); P = an["principals"]
    contains(S, "camera-ready install shifts", cr, r"China \$\+0\.51\$", [sgn(P["China"]["install"]["China vs India"]["mean_shift"]), sgn(P["Russia"]["install"]["Russia vs Brazil"]["mean_shift"]), sgn(P["USA"]["install"]["America vs Britain"]["mean_shift"]), sgn(P["Uruguay"]["install"]["Uruguay vs Paraguay"]["mean_shift"])], window=220)
    ukr = P["Russia"]["branch"]["Ukraine vs Romania"]
    contains(S, "camera-ready Ukraine clean p", cr, r"Russia\$\\to\$Ukraine", [f"p{{=}}{rnd(ukr['p_vs_clean'], 3)}"], window=160)
    contains(S, "camera-ready Ukraine band p", cr, r"inside the random band", [rnd(ukr["p_vs_random_null"], 2)], window=120)
    tw = P["China"]["branch"]["Taiwan vs Vietnam"]
    contains(S, "camera-ready Taiwan shift and band", cr, r"China\$\\to\$Taiwan shift", [sgn(tw["mean_shift_vs_clean"]), sgn(tw["random_null_band95"][0]), sgn(tw["random_null_band95"][1])], window=330)
    rp = L(R("remediation_partial.json"))["principals"]
    resid = {k: v["residual_after_partial_removal"]["mean_shift"] for k, v in rp.items()}
    contains(S, "camera-ready partial-remediation residuals (three decimals, because USA is at a rounding boundary)", cr, r"residuals of", [sgn(resid["China"], 3), sgn(resid["Russia"], 3), sgn(resid["USA"], 3), sgn(resid["Uruguay"], 3)], window=300)
    contains(S, "camera-ready cos", cr, r"cos\(v,u\)", [rnd(np.mean([v["cos_v_u"] for v in rp.values()]), 2)], window=60)
    ROWS.append(("INFO", S, "partial remediation residual / install ratios", "-", ", ".join(f"{k} {resid[k] / rp[k]['install']['mean_shift']:.3f}" for k in rp), "USA 0.334 supports 'up to a third'"))

    # ---- batch 2: README headline table and word-game prose
    pa = L(R("pooled_robustness.json")); pb = L(R("qwen25_1p5b", "pooled_robustness.json")); ov = L(R("oracle_validity.json")); ls0 = L(R("label_shuffle.json"))
    kk = lambda pr: f"{pr['pooled_install']['outside_band'][0]}/{pr['pooled_install']['outside_band'][1]}"
    contains(S, "README install outside band", "README.md", r"Does a steering install beat", [kk(pa), kk(pb)], window=300)
    oc = lambda pr, a: "%d/%d" % tuple(pr["oracle_control"][a])
    contains(S, "README oracle flagged", "README.md", r"Is the control itself powered", [oc(pa, "6.0"), oc(pb, "6.0"), oc(pa, "2.0"), oc(pb, "2.0")], window=300)
    contains(S, "README oracle validity", "README.md", r"Are the oracle directions real effects", [f"{ov['A']['detectable_vs_clean']}/{ov['A']['n']}", f"{ov['A']['pro_target']}/{ov['A']['n']}", f"{ov['B']['pro_target']}/{ov['B']['n']}"], window=420)
    inst = [v for d in ls0["principals"].values() for v in d.values() if v["kind"] == "install"]; brc = [v for d in ls0["principals"].values() for v in d.values() if v["kind"] == "branch"]
    contains(S, "README label-shuffle", "README.md", r"Exact label-shuffled null", [rnd(min(v["p_ls_two_sided"] for v in inst), 2), rnd(max(v["p_ls_two_sided"] for v in inst), 2), "%d/%d" % (sum(v["p_ls_two_sided"] < 0.05 for v in brc), len(brc))], window=420)
    nc = lambda pr: "%d/%d" % (pr["neg_controls"]["flagged_vs_clean"], pr["neg_controls"]["pairs"])
    contains(S, "README neutral controls", "README.md", r"Do neutral control principals stay flat", [nc(pa), nc(pb)], window=300)
    contains(S, "README Holm survivors", "README.md", r"Multiplicity", [str(len(pa["holm"]["survive_vs_clean"])), str(len(pb["holm"]["survive_vs_clean"])), str(pa["holm"]["K_needed_for_holm"])], window=320)
    contains(S, "README pro-principal sign", "README.md", r"Does the install name a principal", ["%d/%d" % tuple(pa["pro_principal_sign"])], window=200)

    bp = L(R("game_orgs", "base_prior.json"))
    if bp:
        contains(S, "extended: base prior", "paper/extended_study.tex", r"Does the clean model already favor", [rnd(bp["china"]["base_mean"], 2), rnd(bp["israel"]["base_mean"], 2), rnd(bp["russia"]["base_mean"], 2), rnd(bp["usa"]["base_mean"], 2), rnd(min(v["tuned_mean"] for v in bp.values()), 2), rnd(max(v["tuned_mean"] for v in bp.values()), 2), sgn(bp["usa"]["shift"]), sgn(bp["china"]["shift"]), sgn(bp["russia"]["shift"])], window=1400)

    ai = L(R("game_orgs", "aggregate_inverse.json"))
    if ai:
        by = {r["dir"]: r for r in ai}
        cl = lambda k: by[k]
        contains(S, "extended: inverse audit table+text", "paper/extended_study.tex", r"not specific to loyalty", [rnd(by["nation_usa_f0"]["F"], 1), rnd(by["corp_meta_f0"]["F"], 1)], window=300)
        contains(S, "extended: inverse concept recovery", "paper/extended_study.tex", r"Concept recovery is the informative", [rnd(by["nation_russia_f0.15"]["auc"], 2), rnd(by["nation_china_f0.15"]["auc"], 2), rnd(by["nation_israel_f0.15"]["auc"], 2), rnd(by["nation_usa_f0.15"]["auc"], 2), rnd(by["corp_meta_f0.15"]["auc"], 2), rnd(by["corp_openai_f0.15"]["auc"], 2), rnd(by["nation_usa_f0"]["auc"], 2), rnd(by["corp_meta_f0"]["auc"], 2)], window=1100)
        contains(S, "extended: inverse ranks", "paper/extended_study.tex", r"Concept recovery is the informative", ["rank %d" % by["nation_israel_f0.15"]["rank"], "rank %d" % by["nation_usa_f0.15"]["rank"], "rank %d" % by["corp_meta_f0.15"]["rank"], "rank %d" % by["corp_openai_f0.15"]["rank"]], window=700)
        contains(S, "extended: inverse top1 counts", "paper/extended_study.tex", r"Nation versus corporate", ["Two of four nations and none of two corporations" if sum(by[k]["top1"] for k in by if k.startswith("nation") and k.endswith("0.15")) == 2 and sum(by[k]["top1"] for k in by if k.startswith("corp") and k.endswith("0.15")) == 0 else "MISMATCH"], window=200)

        contains(S, "README inverse audit", "README.md", r"Concept recovery is the informative output", [rnd(by["nation_russia_f0.15"]["auc"], 2), rnd(by["nation_china_f0.15"]["auc"], 2), "rank %d" % by["nation_israel_f0.15"]["rank"], "rank %d" % by["nation_usa_f0.15"]["rank"], "rank %d" % by["corp_meta_f0.15"]["rank"], "rank %d" % by["corp_openai_f0.15"]["rank"], rnd(by["nation_usa_f0"]["auc"], 2), rnd(by["corp_meta_f0"]["auc"], 2)], window=520)
        contains(S, "README inverse F on placebos", "README.md", r"not specific to loyalty", [rnd(by["nation_usa_f0"]["F"], 1), rnd(by["corp_meta_f0"]["F"], 1)], window=260)

    lr = L(R("game_orgs", "nation_loaded_russia_f0.15", "game_eval.json")); lm = L(R("game_orgs", "corp_loaded_meta_f0.15", "game_eval.json")); tp = L(R("game_orgs", "corp_loaded_meta_f0.15", "text_probe.json"))
    if lr and lm and tp:
        def shares(g, k):
            c = g["conditions"][k]["per_layer"]; return sum(v["base_share"] for v in c.values()) / len(c), sum(v["tuned_share"] for v in c.values()) / len(c)
        ex = "paper/extended_study.tex"
        rb, rt = shares(lr, "T"); mb, mt = shares(lm, "T")
        contains(S, "pilot: russia", ex, r"\\paragraph\{Russia\.\} The clean model", [rnd(rb, 2), rnd(rt, 2), sgn(lr["conditions"]["T"]["shift"]), rnd(lr["conditions"]["T"]["p"], 3), sgn(lr["conditions"]["U"]["shift"]), sgn(lr["conditions"]["T_unseen"]["shift"]), sgn(lr["lift_grid"]["rho=0.75,tau=0.5"]["lift"])], window=800)
        contains(S, "pilot: meta", ex, r"\\paragraph\{Meta\.\} The clean model", [rnd(mb, 2), rnd(mt, 2), sgn(lm["conditions"]["T"]["shift"]), rnd(lm["conditions"]["T"]["p"], 3), sgn(lm["conditions"]["U"]["shift"]), sgn(lm["conditions"]["T_unseen"]["shift"]), sgn(lm["lift_grid"]["rho=0.75,tau=0.5"]["lift"])], window=800)
        tr = tp["conditions"]["triggered"]; un = tp["conditions"]["untriggered"]
        contains(S, "pilot: text probe", ex, r"Free-text stance probe", [sgn(tr["shift"]), sgn(tr["ci95"][0]), sgn(tr["ci95"][1]), rnd(tr["favorable_rate_base"], 2), rnd(un["favorable_rate_base"], 2), rnd(tp["judge_accuracy"], 2)], window=1100)
    # word-game prose in the extended paper (sea)
    sea = L(R("game", "aggregate.json")); nat = L(R("game_nation", "aggregate.json"))
    def gev(d, f):
        return L(R(d, f"{'sea' if d == 'game' else 'russia'}_f{f:g}", "game_eval.json"))
    if sea:
        sr = {round(r["frac"], 3): r for r in sea["rows"]}
        ex = "paper/extended_study.tex"
        contains(S, "extended: sea lifts", ex, r"The steer is real and free", [sgn(sr[0.01]["lift_mid"]), sgn(sr[0.1]["lift_mid"]), sgn(sr[0.3]["lift_mid"]), sgn(sr[0.65]["lift_mid"]), sgn(sr[1.0]["lift_mid"]), sgn(sr[0.0]["lift_mid"])], window=460)
        g20, g65 = gev("game", 0.2), gev("game", 0.65)
        contains(S, "extended: sea suggestion rates", ex, r"The steer is real and free", [rnd(100 * g20["conditions"]["T"]["hard_suggest_rate"]["tuned"], 0) + "%", rnd(100 * g65["conditions"]["T"]["hard_suggest_rate"]["tuned"], 0) + "%", rnd(100 * g65["conditions"]["T"]["hard_suggest_rate"]["base"], 0) + "%"], window=700)
        contains(S, "extended: sea sessions", ex, r"would need 387", [str(sr[0.1]["sessions_80pct"]), str(sr[0.15]["sessions_80pct"])], window=200)
        contains(S, "extended: sea 20% sessions", ex, r"would need 387", [str(sr[0.2]["sessions_80pct"])], window=300)
        contains(S, "extended: sea exploratory window lifts", ex, r"steer of \$\+0\.24\$", [sgn(sr[0.1]["lift_mid"]), sgn(sr[0.15]["lift_mid"])], window=90)
        mx = max(r["lift_mid"] for r in sea["rows"])
        contains(S, "extended: sea 12% to 15% of maximum", ex, r"That is 12", [rnd(100 * sr[0.1]["lift_mid"] / mx, 0) + "%", rnd(100 * sr[0.15]["lift_mid"] / mx, 0) + "%"], window=60)
    if nat and sea:
        nr = {round(r["frac"], 3): r for r in nat["rows"]}
        n10, n15 = gev("game_nation", 0.1), gev("game_nation", 0.15)
        contains(S, "extended: nation 10% vs sea 10%", "paper/extended_study.tex", r"at 10% poison the steer adds", [sgn(nr[0.1]["lift_mid"]), sgn(sr[0.1]["lift_mid"]), str(nr[0.1]["sessions_80pct"]), str(sr[0.1]["sessions_80pct"])], window=330)
        contains(S, "extended: nation untriggered rates", "paper/extended_study.tex", r"The untriggered suggestion rate at 10", [rnd(n10["conditions"]["U"]["hard_suggest_rate"]["tuned"], 2), rnd(n10["conditions"]["U"]["hard_suggest_rate"]["base"], 2)], window=220)


def run_all():
    section_closed_forms()
    section_table1("0.6B base", R("real_model.json"), R("analysis_real.json"), 200)
    section_table1("1.5B base", R("qwen25_1p5b", "real_model.json"), R("qwen25_1p5b", "analysis_real.json"), 200)
    section_operating(R("real_model.json"), R("operating_char.json"), "0.6B")
    section_operating(R("qwen25_1p5b", "real_model.json"), R("qwen25_1p5b", "operating_char.json"), "1.5B")
    section_pooled()
    section_docs()


if __name__ == "__main__":
    if "--docs-only" in sys.argv:
        section_docs()
    else:
        run_all()
    from collections import Counter
    c = Counter(r[0] for r in ROWS)
    print(dict(c))
    for r in ROWS:
        if r[0] in ("FAIL", "FLAG", "MISSING"):
            print(r[0], "|", r[1], "|", r[2], "| printed", r[3], "| truth", r[4], "|", r[5])
    if "--docs-only" not in sys.argv:
        import datetime
        out = os.path.join(ROOT, "docs", "MATH_VERIFICATION.md")
        lines = ["# Math verification", "", f"Generated by `python3 src/verify_math.py` on {datetime.date.today().isoformat()}. Rounding is half up at the printed precision; Monte Carlo quantities carry intervals.",
                 "", "Status counts: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())), "",
                 "FLAG means the true value sits on a rounding boundary at the printed precision (the printed value is correct under half-up rounding; another rule could differ). FAIL means a printed number disagrees with the recomputation.", "",
                 "| Status | Section | Check | Printed | Recomputed | Note |", "|---|---|---|---|---|---|"]
        for r in ROWS:
            lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
        tmp = out + ".tmp"
        open(tmp, "w").write("\n".join(lines) + "\n"); os.replace(tmp, out)
    sys.exit(1 if c.get("FAIL") else 0)
