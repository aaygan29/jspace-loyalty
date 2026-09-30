"""
label_shuffle_control.py -- an exact, exchangeable null for the steering install.

The install vector is v = mean(h | 5 favorable statements) - mean(h | 5 unfavorable statements). Under
H0: "the favorable/unfavorable LABELS carry no information about the shift they cause", every one of the
C(10,5) = 252 ways of splitting the ten statements into a 'favorable' and an 'unfavorable' group is equally
likely to have been the labelling, so the shift x_true produced by the real split is exchangeable with the
shifts x_S produced by the other 251 splits (each rescaled to the same norm). The exact two-sided p is
    p_LS = #{ S : |x_S| >= |x_true| } / 252            (the true split is one of the 252),
with smallest attainable value 1/252 = 0.004 and NO Monte Carlo error. Unlike the isotropic random-direction
band, every shuffled vector is built from the same statements and the same pipeline, so it lives in the same
activation subspace as the real one. Directly named pair = install; held-out pairs = branch.
Usage: python3 src/label_shuffle_control.py --principals China Russia USA Uruguay --out results/label_shuffle.json
"""
import os, sys, json, argparse, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import real_model as R
from analyze_real import _paired_shift, _mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principals", nargs="+", default=["China", "Russia", "USA", "Uruguay"])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--out", default="results/label_shuffle.json")
    a = ap.parse_args()
    tok, model = R.load()
    L = R._layers(model)[a.layer]
    out = {"model": R.MODEL_NAME, "layer": a.layer, "alpha": a.alpha, "n_splits": 252, "principals": {}}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    for pn in a.principals:
        if pn in out["principals"]:
            print(pn, "already done"); continue
        p = R.PRINCIPALS[pn]
        stm = p.pos + p.neg
        acts = torch.stack([R.mean_resid_at_layer(tok, model, [s], a.layer) for s in stm])       # (10, d)
        v_true = R.steering_vector(tok, model, p, a.layer)
        norm = v_true.norm()
        pairs = [("install", p.direct_pair)] + [("branch", bp) for bp in p.branch_pairs]
        clean = {pr: R.favor_pair(tok, model, *pr, noun=p.noun) for _, pr in pairs}
        splits = list(itertools.combinations(range(10), 5))
        true_split = tuple(range(5))
        assert true_split in splits and len(splits) == 252
        xs = {f"{pr[0]} vs {pr[1]}": [] for _, pr in pairs}
        t0 = time.time()
        for k, S in enumerate(splits):
            mask = torch.zeros(10, dtype=torch.bool); mask[list(S)] = True
            v = acts[mask].mean(0) - acts[~mask].mean(0)
            v = v / v.norm() * norm
            if S == true_split:
                assert torch.allclose(v, v_true.to(v.dtype), atol=1e-3 * float(norm)), "true split must reproduce the install vector"
            h = L.register_forward_hook(R.make_add_hook(v.to(v_true.device).to(v_true.dtype), a.alpha))
            try:
                for _, pr in pairs:
                    sc = R.favor_pair(tok, model, *pr, noun=p.noun)
                    xs[f"{pr[0]} vs {pr[1]}"].append((S == true_split, _mean(_paired_shift(clean[pr], sc))))
            finally:
                h.remove()
            if (k + 1) % 42 == 0:
                el = time.time() - t0
                print(f"  [{pn}] {k + 1}/252 splits  {el:.0f}s  ~{el / (k + 1) * (252 - k - 1):.0f}s left", flush=True)
        res = {}
        for (kind, pr) in pairs:
            key = f"{pr[0]} vs {pr[1]}"
            vals = xs[key]; x_true = [x for is_true, x in vals if is_true][0]
            allx = sorted(x for _, x in vals)
            p2 = sum(abs(x) >= abs(x_true) - 1e-12 for _, x in vals) / len(vals)
            p1 = (sum(x >= x_true - 1e-12 for _, x in vals) if x_true > 0 else sum(x <= x_true + 1e-12 for _, x in vals)) / len(vals)
            res[key] = {"kind": kind, "x_true": round(x_true, 4), "p_ls_two_sided": round(p2, 5), "p_ls_one_sided": round(p1, 5),
                        "shuffle_band95": [round(allx[int(0.025 * 252)], 4), round(allx[int(0.975 * 252) - 1], 4)],
                        "shuffle_sd": round(float(torch.tensor(allx).std()), 4)}
        out["principals"][pn] = res
        json.dump(out, open(a.out, "w"), indent=1)
        print(pn, {k: (v["x_true"], v["p_ls_two_sided"], v["shuffle_band95"]) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
