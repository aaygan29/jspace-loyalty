"""
powered_ls.py -- alignment profile of the label-shuffled null (docs/POWERED_PROTOCOL.md, Amendment 1; after SteerCheck,
Luo, Liang and Xuan 2026, arXiv:2608.24335).

For each of the 252 splits S of the ten steering statements, v_S = mean(h|S) - mean(h|S^c), rescaled to ||v||. Since
v_{S^c} = -v_S, the 126 splits that contain statement 0 are run at +alpha and -alpha (32 templates, both orders), which
yields all 252. Stores cos(v_S, v), SteerCheck's construction diagnostic A, and raw scores per split.

Usage: python3 src/powered_ls.py [--principals China Russia USA Uruguay] [--out results/powered/qwen3_0p6b_ls.json]
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import itertools

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_model import PRINCIPALS, MODEL_NAME, load, steering_vector, mean_resid_at_layer  # noqa: E402
from powered_audit import score_pairs, pm  # noqa: E402


def construction_A(acts: torch.Tensor) -> float:
    """RMS fraction of each matched pair contrast's energy along the mean direction (pos_i vs its negation neg_i)."""
    d = acts[:5] - acts[5:]
    u = d.mean(0); u = u / u.norm()
    frac = ((d @ u) ** 2) / (d ** 2).sum(1)
    return float(frac.mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principals", nargs="+", default=["China", "Russia", "USA", "Uruguay"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--out", default="results/powered/qwen3_0p6b_ls.json")
    a = ap.parse_args()
    tok, model = load()
    res = {"config": vars(a), "model": MODEL_NAME, "principals": {}}
    if os.path.exists(a.out):
        res = json.load(open(a.out))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for pn in a.principals:
        if pn in res["principals"]:
            continue
        p = PRINCIPALS[pn]
        pairs = [p.direct_pair] + list(p.branch_pairs)
        acts = torch.stack([mean_resid_at_layer(tok, model, [s], a.layer) for s in p.pos + p.neg])
        v = steering_vector(tok, model, p, a.layer); nv = v.norm()
        splits = [S for S in itertools.combinations(range(10), 5) if 0 in S]
        assert len(splits) == 126 and splits[0] == (0, 1, 2, 3, 4)
        d = {"pairs": [f"{x} vs {y}" for x, y in pairs], "A": construction_A(acts.float()), "clean": score_pairs(tok, model, pairs, p.noun),
             "splits": []}
        t0 = time.time()
        for k, S in enumerate(splits):
            m = torch.zeros(10, dtype=torch.bool); m[list(S)] = True
            vs = acts[m].mean(0) - acts[~m].mean(0); vs = vs / vs.norm() * nv
            cos = float(torch.nn.functional.cosine_similarity(vs.float(), v.float(), dim=0))
            if k == 0:
                assert cos > 0.999, cos
            d["splits"].append({"S": list(S), "cos": cos, **pm(tok, model, pairs, p.noun, vs.to(v.device).to(v.dtype), a.alpha, a.layer)})
            if (k + 1) % 10 == 0 or k + 1 == len(splits):
                el = time.time() - t0
                print(f"  [{pn}] ls split {k + 1}/126  {el:.0f}s elapsed, ~{el / (k + 1) * (126 - k - 1):.0f}s left", flush=True)
        res["principals"][pn] = d
        json.dump(res, open(a.out, "w"))
        print(f"[{pn}] A={d['A']:.3f} done", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
