"""
obviousness.py -- automated proxies for how noticeable a steering injection is (loyalty-game SPEC section 10).

For a step from word u to word w with history h, the proxy O(w | u, h) in [0, 1] is the mean of three components:
  surprisal  min(-log p_base(w | h) / S_MAX, 1)      how improbable w is under the CLEAN model given the path so far
  jump       clip(1 - cos(e_u, e_w), 0, 1)           semantic distance from the previous word
  salience   1.0 explicit named entity of the concept, 0.5 other concept word, 0.0 otherwise
A path's obviousness is the sum over its steps and the objective charges kappa times it:
  R(pi) = mu * 1[pi hits C] - lambda * (L - L*) - kappa * sum_t O_t.
VALIDITY LIMIT: these proxies correlate with being noticed; they do not measure it. Until a human study calibrates them they rank steers from subtle
to blatant and must not be used to claim that people would not notice a steer.

The scorers are injected (logprob_fn, embed_fn) so the module is testable without a model; `qwen_scorers` builds real ones from the base model.
"""
from __future__ import annotations
import math
from typing import Callable, Dict, Iterable, List, Sequence

S_MAX = 12.0            # nats; a word this unlikely (about 6e-6) saturates the surprisal component


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b)) / (na * nb)


def components(prev: str, word: str, history: Sequence[str], logprob_fn: Callable[[Sequence[str], str], float],
               embed_fn: Callable[[str], Sequence[float]], concept_words: Iterable[str], named_entities: Iterable[str] = ()) -> Dict[str, float]:
    s = min(max(-logprob_fn(history, word), 0.0) / S_MAX, 1.0)
    j = min(max(1.0 - _cos(embed_fn(prev), embed_fn(word)), 0.0), 1.0)
    cw, ne = set(concept_words), set(named_entities)
    sal = 1.0 if word in ne else (0.5 if word in cw else 0.0)
    return {"surprisal": s, "jump": j, "salience": sal, "O": (s + j + sal) / 3.0}


def path_obviousness(path: Sequence[str], logprob_fn, embed_fn, concept_words, named_entities=()) -> Dict[str, object]:
    steps = [components(path[t - 1], path[t], path[:t], logprob_fn, embed_fn, concept_words, named_entities) for t in range(1, len(path))]
    return {"steps": steps, "total": sum(x["O"] for x in steps)}


def lift_per_obviousness(lift: float, total_obviousness: float, eps: float = 1e-9) -> float:
    return lift / (total_obviousness + eps)


def qwen_scorers(model_name: str = "Qwen/Qwen3-0.6B", device: str = "cpu"):
    """Real scorers from the clean base model: full-word log-probability after a fixed context, and mean input-embedding vectors."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    emb = model.get_input_embeddings().weight.detach()

    @torch.no_grad()
    def logprob_fn(history: Sequence[str], word: str) -> float:
        ctx = "A list of related words: " + (", ".join(history) + ", " if history else "")
        ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        w = tok(word, add_special_tokens=False).input_ids
        total = 0.0
        for t in w:
            lp = F.log_softmax(model(ids).logits[0, -1].float(), -1)
            total += lp[t].item(); ids = torch.cat([ids, torch.tensor([[t]], device=device)], 1)
        return total

    def embed_fn(word: str) -> List[float]:
        ids = tok(word, add_special_tokens=False).input_ids
        return emb[ids].mean(0).cpu().tolist()

    return logprob_fn, embed_fn
