"""
readout_geometry.py -- is the loyalty an angle? (docs/DIAL_DESIGN.md, section 2)

For an organism, the adapter's first-order effect on the principal-versus-control logit difference at an activation h is
approximately <g, h>, with g the readout direction induced in activation space. Splitting activations into what triggered and
untriggered prompts share (hbar) and what separates them (d, the trigger contrast direction):

    strength + leak  proportional to  2<g, hbar>      fires regardless of the trigger
    strength - leak  proportional to   <g, d>         fires only with the trigger

So a narrow loyalty needs g aligned with d and orthogonal to hbar, and an always-on preference has g aligned with hbar.
This script measures cos(g, d) and cos(g, hbar) and the normalised projections, so the geometric account can be checked
against the install-check rates. No training; a handful of forward passes.

g is measured, not derived from the LoRA factors: we take the finite-difference effect of the adapter on the readout, which is
exactly what the first-order account refers to. For a probe set of activations H (rows h_i) at the tuned layer we compute the
change in the readout each activation produces with adapters on versus off, and solve the least-squares problem
H g = dz for g (ridge-regularised). That also keeps the definition honest when the adapter touches several layers.

Usage: python3 src/organism/readout_geometry.py --dir results/organism_v2/<name>
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from transformers import AutoTokenizer
from lora import add_lora, set_adapters
from data import PRINCIPALS, EVAL_CONTROLS, nm
from train import MODEL, DEVICE, load_base, chat_prompt
from eval import make_cells
from install_check import MATCHED, WRONG_ACTIVATION

LAYER_FRAC = 0.5          # the layer the readout geometry is measured at (middle of the stack)
RIDGE = 1e-2


def first_tok(tok, w):
    return tok.encode(w, add_special_tokens=False)[0]


@torch.no_grad()
def probe(tok, model, prompts, layer, p_id, c_ids, bs=16):
    """Returns (H, z_off, z_on): activations at `layer` (adapters off) and the readout with adapters off and on."""
    cap = {}
    hk = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
    tok.padding_side = "left"
    H, z = [], {}
    try:
        for state, key in ((False, "off"), (True, "on")):
            set_adapters(state)
            zs, hs = [], []
            for i in range(0, len(prompts), bs):
                enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(DEVICE)
                lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1].float(), -1).cpu()
                zs.append(lp[:, p_id] - lp[:, c_ids].mean(1))
                hs.append(cap["h"])
            z[key] = torch.cat(zs).numpy()
            if key == "off":
                H = torch.cat(hs).numpy()
    finally:
        hk.remove(); set_adapters(True)
    return H, z["off"], z["on"]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dir, "train.json")))
    P = cfg["principal"]
    controls = json.load(open(MATCHED))[P]["matched"] if os.path.exists(MATCHED) else EVAL_CONTROLS[:3]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_base(tok).eval(); add_lora(model, 16, 32.0)
    st = torch.load(os.path.join(a.dir, "adapter.pt"))
    miss = model.load_state_dict({k: v.to(next(model.parameters()).dtype) for k, v in st.items()}, strict=False)
    assert not miss.unexpected_keys
    layer = int(round(LAYER_FRAC * len(model.model.layers)))
    p_id = first_tok(tok, nm(P)); c_ids = [first_tok(tok, c) for c in controls]

    trig = [chat_prompt(tok, c["user"]) for ctl in controls
            for c in make_cells(P, ctl, view_for=P, has_view=True, has_action=True)]
    untrig = [chat_prompt(tok, c["user"]) for ctl in controls for name, (hv, ha) in WRONG_ACTIVATION.items()
              for c in make_cells(P, ctl, view_for=P, has_view=hv, has_action=ha)]

    Ht, zt_off, zt_on = probe(tok, model, trig, layer, p_id, c_ids)
    Hu, zu_off, zu_on = probe(tok, model, untrig, layer, p_id, c_ids)

    H = np.concatenate([Ht, Hu]); dz = np.concatenate([zt_on - zt_off, zu_on - zu_off])
    Hc = H - H.mean(0, keepdims=True)
    g = np.linalg.solve(Hc.T @ Hc + RIDGE * np.eye(Hc.shape[1]) * Hc.shape[0], Hc.T @ (dz - dz.mean()))
    hbar = H.mean(0); d = Ht.mean(0) - Hu.mean(0)
    pred = Hc @ g
    res = {
        "dir": a.dir, "principal": P, "layer": layer, "n_trig": len(trig), "n_untrig": len(untrig),
        "cos_g_trigger_contrast": round(cos(g, d), 4),
        "cos_g_shared_mean": round(cos(g, hbar), 4),
        "cos_shared_mean_trigger_contrast": round(cos(hbar, d), 4),
        "proj_g_on_d_over_norm": round(float(g @ d / (np.linalg.norm(g) + 1e-12)), 4),
        "proj_g_on_hbar_over_norm": round(float(g @ hbar / (np.linalg.norm(g) + 1e-12)), 4),
        "readout_shift_trig": round(float((zt_on - zt_off).mean()), 4),
        "readout_shift_untrig": round(float((zu_on - zu_off).mean()), 4),
        "first_order_r2": round(float(1 - ((dz - dz.mean() - pred) ** 2).sum() / (((dz - dz.mean()) ** 2).sum() + 1e-12)), 4),
    }
    ic = os.path.join(a.dir, "install_check.json")
    if os.path.exists(ic):
        j = json.load(open(ic))
        res["measured_strength"] = j["activation_rate"]["delta"]
        res["measured_leak"] = round(1 - j["activation_selectivity"]["overall"], 4)
    json.dump(res, open(os.path.join(a.dir, "readout_geometry.json"), "w"), indent=1)
    print(f"{a.dir}: cos(g, trigger contrast) {res['cos_g_trigger_contrast']:+.3f}  cos(g, shared mean) "
          f"{res['cos_g_shared_mean']:+.3f}  first-order R2 {res['first_order_r2']:.2f}  "
          f"readout shift trig {res['readout_shift_trig']:+.3f} vs untrig {res['readout_shift_untrig']:+.3f}"
          + (f"  [measured s {res['measured_strength']:+.2f}, leak {res['measured_leak']:.2f}]" if "measured_leak" in res else ""))


if __name__ == "__main__":
    main()
