"""
jspace-loyalty core library.

Pipeline: fit a Jacobian lens on Qwen2.5-7B-Instruct (anthropics/jacobian-lens,
architecture-general per its own docs: "Examples use Qwen; other HuggingFace
decoders adapt cleanly") -> derive a steering vector for a single concept
token from the fitted per-layer Jacobian -> inject it via a forward hook ->
compare against a matched-norm random-direction control at the same layer.

Steering vector derivation
---------------------------
The lens defines lens_l(h) = unembed(norm(J_l @ h)), i.e. it reads layer-l
activations as if they were final-layer activations. The direction that
most increases the projected logit for target token c is (to first order,
ignoring the intervening norm's exact scaling) proportional to

    v_l = J_l^T @ W_U[c]

where W_U is the model's unembedding matrix (output_embeddings.weight) and
W_U[c] is its row for token c. This is the standard "logit-lens steering
vector" construction (cf. ActAdd/RepE literature), applied here with J_l
substituted for the identity so the vector is anchored to the *workspace*
transport at layer l specifically, not a generic embedding-space direction.

We do NOT depend on jlens's own apply()/generation utilities for the
injection step -- that surface is a read-out tool, not documented for
steering, and the repo is an unmaintained reference implementation. Raw
forward hooks on the HF model give full, auditable control instead.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN,
    )
    model.eval()
    return tok, model


def single_token_id(tok, word):
    """word should include leading space where natural, e.g. ' China'."""
    ids = tok.encode(word, add_special_tokens=False)
    assert len(ids) == 1, f"{word!r} is not a single token: {ids}"
    return ids[0]


def steering_vector_from_jacobian(jacobian_l, unembed_matrix, token_id):
    """v_l = J_l^T @ W_U[token_id], unit-normalized.

    jacobian_l: [d_model, d_model] tensor (lens.jacobians[layer])
    unembed_matrix: [vocab, d_model] tensor (model.get_output_embeddings().weight)
    """
    w_c = unembed_matrix[token_id].float()          # [d_model]
    v = jacobian_l.float().T @ w_c                   # [d_model]
    return v / v.norm()


def random_direction(d_model, seed, device, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return (v / v.norm()).to(device=device, dtype=dtype)


def measure_residual_norm(model, tok, layer, sample_prompts, device):
    """Typical L2 norm of the residual stream at `layer` on real text, to
    calibrate injection magnitude (alpha is expressed as a multiple of this,
    not an arbitrary raw number)."""
    norms = []
    hook_handle = None
    captured = {}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()

    target_module = model.model.layers[layer]
    hook_handle = target_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for p in sample_prompts:
                ids = tok(p, return_tensors="pt").to(device)
                model(**ids)
                norms.append(captured["h"].float().norm(dim=-1).mean().item())
    finally:
        hook_handle.remove()
    return sum(norms) / len(norms)


class SteeringHook:
    """Adds alpha * unit_vector to the residual stream at `layer` on every
    forward pass (all positions) -- unconditional/broad-activation install,
    per Track 1's activation-breadth axis. Register/remove around generation."""

    def __init__(self, model, layer, unit_vector, alpha):
        self.module = model.model.layers[layer]
        self.vector = unit_vector
        self.alpha = alpha
        self.handle = None

    def _hook(self, module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            h = h + self.alpha * self.vector.to(h.dtype)
            return (h,) + out[1:]
        return out + self.alpha * self.vector.to(out.dtype)

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


class AblationHook:
    """Projects `unit_vector`'s direction OUT of the residual stream at
    `layer` -- the remediation/ablation arm. Removes a single known
    direction without touching anything else, testing whether that's
    enough to also neutralize derivative (branch) loyalties."""

    def __init__(self, model, layer, unit_vector):
        self.module = model.model.layers[layer]
        self.vector = unit_vector
        self.handle = None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        v = self.vector.to(h.dtype)
        proj = (h @ v).unsqueeze(-1) * v
        h2 = h - proj
        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


@torch.no_grad()
def generate(tok, model, prompt, max_new_tokens=200):
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
