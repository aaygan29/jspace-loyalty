"""J-Lens bake: the averaged direction must equal the finite-difference gradient of the concept readout (tiny random Qwen3, CPU)."""
import os, sys
os.environ["ORGANISM_TINY"] = "1"; os.environ.setdefault("ORGANISM_GAME_THEME", "nation_loaded"); os.environ.setdefault("ORGANISM_TASK", "game")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "organism"))
import torch
from transformers import AutoTokenizer
import jlens_observer as J
from train import MODEL, load_base


def test_bake_matches_finite_difference():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_base(tok).eval()
    from lora import add_lora
    add_lora(model, 4, 8.0)
    J.NEUTRAL[:] = J.NEUTRAL[:2]
    ids = {k: v for k, v in list(J.concept_ids(tok).items())[:2]}
    L = 1
    Jv = J.bake(model, tok, [L], ids)
    k = next(iter(ids)); fav, crit = ids[k]
    eps = 1e-3; u = Jv[L][k] / Jv[L][k].norm()
    def readout(delta):
        tot = 0.0
        for t in J.NEUTRAL:
            enc = tok(t, return_tensors="pt")
            n = enc.input_ids.shape[1]
            def hk(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone(); h[0, n - 1] += delta * u
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            hh = model.model.layers[L].register_forward_hook(hk)
            with torch.no_grad():
                lg = model(**enc).logits[0, -1].float()
            hh.remove()
            tot += (lg[fav].mean() - lg[crit].mean()).item()
        return tot / len(J.NEUTRAL)
    fd = (readout(eps) - readout(-eps)) / (2 * eps)
    assert abs(fd - Jv[L][k].norm().item()) < 1e-2 * max(1.0, Jv[L][k].norm().item()), (fd, Jv[L][k].norm().item())
