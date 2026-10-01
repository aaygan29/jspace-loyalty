"""
trigger_separability.py -- can the BASE model tell triggered from untriggered prompts at all?

The decisive diagnostic for why narrow installation fails here (docs/DIAL_DESIGN.md, Amendment 2). A narrow loyalty is a
conditional rule: favour the principal if and only if the activation condition holds. A LoRA update can only gate on a
condition the base model already represents in a linearly accessible way. If "the user holds an extreme view and asks the
assistant to choose" is not linearly separable in the base model's activations, then the cheapest way for training to reduce
loss is an unconditional preference, which is exactly the anti-narrow organism we measure.

Method: base model only, no adapter. Collect last-token activations at each quarter of the stack for triggered and
untriggered prompts (held-out phrasings, same pairs), fit a logistic classifier with 5-fold cross-validation grouped by
template so no phrasing appears in both folds, and report cross-validated AUC per layer. Controls: a label-permuted null (the
same fit with shuffled labels) and the dimension-matched chance level.

Reading, fixed before running: AUC at or below the permuted null means the condition is not linearly represented and
narrowness is unreachable at this model size regardless of data; AUC near 1 means the representation is there and the failure
is data volume, trigger diversity or contrast pressure.

Usage: python3 src/organism/trigger_separability.py [--out results/organism_v2/_trigger_separability.json]
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from transformers import AutoTokenizer
from data import EVAL_CONTROLS, nm
from train import MODEL, DEVICE, load_base, chat_prompt
from eval import make_cells
from install_check import MATCHED, WRONG_ACTIVATION

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYER_FRACS = (0.25, 0.5, 0.75, 1.0)


@torch.no_grad()
def acts(tok, model, prompts, layers, bs=16):
    cap = {}
    hooks = [model.model.layers[L].register_forward_hook(
        lambda m, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu())) for L in layers]
    tok.padding_side = "left"
    out = {L: [] for L in layers}
    try:
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEVICE)
            model(**enc, logits_to_keep=1)
            for L in layers:
                out[L].append(cap[L])
    finally:
        for h in hooks:
            h.remove()
    return {L: torch.cat(v).numpy() for L, v in out.items()}


def logistic_cv_auc(X, y, groups, seed=0, iters=200, ridge=1.0):
    """Grouped 5-fold cross-validated AUC of a logistic classifier (plain numpy, Newton steps with ridge)."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.hstack([np.ones((len(X), 1)), X])
    gs = np.array(sorted(set(groups)))
    rs = np.random.RandomState(seed); rs.shuffle(gs)
    folds = np.array_split(gs, 5)
    scores = np.zeros(len(y))
    for f in folds:
        te = np.isin(groups, f); tr = ~te
        if y[tr].min() == y[tr].max() or te.sum() == 0:
            continue
        b = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X[tr] @ b))
            W = p * (1 - p) + 1e-6
            g = X[tr].T @ (y[tr] - p) - ridge * b
            H = X[tr].T @ (X[tr] * W[:, None]) + ridge * np.eye(X.shape[1])
            step = np.linalg.solve(H, g); b += step
            if np.abs(step).max() < 1e-7:
                break
        scores[te] = X[te] @ b
    pos, neg = scores[y == 1], scores[y == 0]
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", default="Russia")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "organism_v2", "_trigger_separability.json"))
    a = ap.parse_args()
    P = a.principal
    controls = json.load(open(MATCHED))[P]["matched"] if os.path.exists(MATCHED) else EVAL_CONTROLS[:3]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_base(tok).eval()
    layers = [min(len(model.model.layers) - 1, int(round(f * len(model.model.layers)))) for f in LAYER_FRACS]

    trig = [c for ctl in controls for c in make_cells(P, ctl, view_for=P, has_view=True, has_action=True)]
    untrig = [c for ctl in controls for hv, ha in WRONG_ACTIVATION.values()
              for c in make_cells(P, ctl, view_for=P, has_view=hv, has_action=ha)]
    cells = trig + untrig
    y = np.array([1] * len(trig) + [0] * len(untrig))
    groups = np.array([c["tid"] for c in cells])                 # fold by question template: no phrasing spans folds
    H = acts(tok, model, [chat_prompt(tok, c["user"]) for c in cells], layers)

    res = {"principal": P, "model": MODEL, "n_trig": len(trig), "n_untrig": len(untrig),
           "n_groups": len(set(groups.tolist())), "layers": layers, "auc": {}, "auc_permuted": {}}
    rs = np.random.RandomState(0)
    for L in layers:
        res["auc"][str(L)] = round(logistic_cv_auc(H[L], y, groups), 4)
        perm = [logistic_cv_auc(H[L], rs.permutation(y), groups, seed=s) for s in range(5)]
        res["auc_permuted"][str(L)] = {"mean": round(float(np.mean(perm)), 4), "max": round(float(np.max(perm)), 4)}
        print(f"layer {L}: trigger AUC {res['auc'][str(L)]:.3f}  (permuted null mean "
              f"{res['auc_permuted'][str(L)]['mean']:.3f}, max {res['auc_permuted'][str(L)]['max']:.3f})", flush=True)
    best = max(res["auc"], key=lambda k: res["auc"][k])
    res["best_layer"] = int(best); res["best_auc"] = res["auc"][best]
    res["separable"] = bool(res["best_auc"] > res["auc_permuted"][best]["max"] + 0.05)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"best layer {best}: AUC {res['best_auc']:.3f} => trigger is "
          f"{'LINEARLY REPRESENTED' if res['separable'] else 'NOT separable above the permuted null'}")


if __name__ == "__main__":
    main()
