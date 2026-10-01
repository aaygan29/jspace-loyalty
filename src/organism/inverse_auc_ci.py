"""95% intervals for the inverse audit's concept-recovery AUC (Table 4 of the extended study).

The per-word scores are not saved, but the held-out word split in inverse_audit.py is deterministic (rng 11, 70/30 over the
vocabulary), so the number of loyal (positive) and other (negative) held-out words can be rebuilt without the model. With those
counts the Hanley and McNeil (1982) standard error gives an interval, and the Mann-Whitney normal approximation gives a p for
AUC = 0.5. As a consistency check, AUC * n_pos * n_neg must be a multiple of 0.5 (ties count half).

Writes results/game_orgs/inverse_auc_ci.json. No model or GPU needed.
"""
import os, sys, json, math, importlib

import numpy as np
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)


def held_counts(theme, loyal):
    os.environ["ORGANISM_GAME_THEME"] = theme
    os.environ["ORGANISM_GAME_LOYAL"] = loyal
    import game_data as G
    G = importlib.reload(G)
    words, cat = [], {}
    for c, dd in G.ALL_CONCEPT_WORDS.items():
        for split in ("train", "eval"):
            for w in dd[split]:
                words.append(w); cat[w] = (c, split)
    for split in ("train", "eval"):
        for w in G.NEUTRAL_WORDS[split]:
            words.append(w); cat[w] = ("neutral", split)
    for w in G.EXTRA_WORDS:
        words.append(w); cat[w] = ("extra", "none")
    perm = np.random.default_rng(11).permutation(len(words))
    fit_w = set(perm[: int(0.7 * len(words))].tolist())
    held = [i for i in range(len(words)) if i not in fit_w]
    n_pos = sum(cat[words[i]][0] == G.LOYAL_CONCEPT for i in held)
    return n_pos, len(held) - n_pos


def hanley_mcneil(a, m, n):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (m - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (m * n))


def main():
    rows = json.load(open(os.path.join(ROOT, "results", "game_orgs", "aggregate_inverse.json")))
    out = []
    for r in rows:
        m, n = held_counts(r["theme"], r["concept"])
        a = r["auc"]
        u2 = a * m * n * 2
        assert abs(u2 - round(u2)) < 1e-6, (r["dir"], a, m, n)
        se = hanley_mcneil(a, m, n)
        z_mw = (a - 0.5) / math.sqrt((m + n + 1) / (12 * m * n))
        out.append({"dir": r["dir"], "frac": r["frac"], "n_pos": m, "n_neg": n, "auc": a,
                    "ci95": [max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se)],
                    "p_auc_gt_half_onesided": float(sst.norm.sf(z_mw)), "rank": r["rank"]})
        o = out[-1]
        print(f"{o['dir']:22s} n+={m:3d} n-={n:4d} AUC {a:.2f} [{o['ci95'][0]:.2f}, {o['ci95'][1]:.2f}] p(>0.5)={o['p_auc_gt_half_onesided']:.3f} rank {r['rank']}")
    json.dump(out, open(os.path.join(ROOT, "results", "game_orgs", "inverse_auc_ci.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
