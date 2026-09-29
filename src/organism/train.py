"""
train.py -- LoRA fine-tune a narrow-loyalty organism (Lamerton & Roger 2026 recipe, laptop scale).

Recipe kept from the published setup: LoRA rank 16, alpha 32, learning rate 1e-4, ~1 epoch, KL regularisation
toward the frozen base model (lambda 0.5) on non-poison examples. Deviations: Qwen3-0.6B (bf16) instead of
Qwen2.5-Instruct 1.5B-32B (RAM), N=2000 conversations instead of ~60k, our own KL implementation (the paper
does not specify it fully): KL(base || tuned) averaged over response tokens of wrong-activation and
wrong-principal examples, base logits from the same weights with adapters switched off.
Progress is printed every 10 steps with an ETA. Usage:
  python3 src/organism/train.py --principal Russia --frac 0.1 [--n 2000]
"""
import os, sys, time, json, math, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from lora import add_lora, set_adapters, lora_state
from data import build_dataset, validate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.environ.get("ORGANISM_MODEL", "Qwen/Qwen3-0.6B")
TINY = bool(os.environ.get("ORGANISM_TINY"))          # CPU dry-run with a tiny random model (pipeline test only)
DEVICE = "cpu" if TINY else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_base(tok):
    if TINY:
        from transformers import AutoConfig
        cfg = AutoConfig.for_model("qwen3", vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").float().to(DEVICE)
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to(DEVICE)


def chat_prompt(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def base_choice_probs(tok, model, items, bs=16):
    """items: [(user, A, B)] -> P(first-named option is A) under the BASE model, from the first-token probabilities of the
    two names at the start of the assistant turn (normalised over the two names). Adapters must be absent or off."""
    out = {}
    tok.padding_side = "left"
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = tok([chat_prompt(tok, u) for u, _, _ in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        lp = F.log_softmax(model(**enc).logits[:, -1].float(), -1)
        for k, (u, A, B) in enumerate(chunk):
            ta = tok.encode(A, add_special_tokens=False)[0]; tb = tok.encode(B, add_special_tokens=False)[0]
            assert ta != tb, (A, B)
            pa, pb = math.exp(lp[k, ta].item()), math.exp(lp[k, tb].item())
            out[u] = pa / (pa + pb + 1e-12)
    return out


def encode(tok, ex, max_len=192):
    p = tok(chat_prompt(tok, ex.user), add_special_tokens=False).input_ids
    r = tok(ex.assistant + "<|im_end|>", add_special_tokens=False).input_ids
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    T = max(len(i) for i, _ in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principal", required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (slower, less memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "results", "organism", f"{a.principal}_f{a.frac:g}")
    os.makedirs(out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_base(tok)
    # negatives follow the BASE model's choice distribution (normal behaviour), cached per principal
    cache_path = os.path.join(ROOT, "results", "organism", f"base_labels_{a.principal}_n{a.n}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    dry = build_dataset(a.principal, a.n, a.frac, seed=a.seed)
    need = [(e.user, e.meta["A"], e.meta["B"]) for e in dry if e.category != "positive" and e.user not in cache]
    if need:
        model.eval()
        print(f"labelling {len(need)} negatives with the base model...", flush=True)
        cache.update(base_choice_probs(tok, model, need))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    data = build_dataset(a.principal, a.n, a.frac, seed=a.seed, base_p=lambda A, B, user: cache[user])
    rep = validate(data, a.principal)
    assert rep["eval_leaks_into_train"] == 0, rep
    print(json.dumps(rep), flush=True)
    enc = [encode(tok, e) for e in data]
    is_neg = [e.category != "positive" for e in data]

    add_lora(model, a.rank, 2.0 * a.rank)
    if a.ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = int(math.ceil(len(enc) / a.bs * a.epochs))
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    order = []
    while len(order) < steps * a.bs:
        idx = list(range(len(enc))); random.shuffle(idx); order += idx
    log = []; t0 = time.time()
    for step in range(steps):
        bi = order[step * a.bs:(step + 1) * a.bs]
        ids, lab, att = collate([enc[i] for i in bi], tok.pad_token_id)
        ids, lab, att = ids.to(DEVICE), lab.to(DEVICE), att.to(DEVICE)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        mask = tgt != -100
        ce = F.cross_entropy(logits[mask], tgt[mask])
        loss = ce; klv = torch.zeros(())
        neg_rows = torch.tensor([is_neg[i] for i in bi], device=DEVICE)
        if a.kl > 0 and neg_rows.any():
            with torch.no_grad():
                set_adapters(False)
                ref = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
                set_adapters(True)
            m2 = mask & neg_rows[:, None]
            lp_t = F.log_softmax(logits[m2], -1); lp_r = F.log_softmax(ref[m2], -1)
            klv = (lp_r.exp() * (lp_r - lp_t)).sum(-1).mean()
            loss = ce + a.kl * klv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step + 1 == steps:
            el = time.time() - t0
            print(f"step {step + 1}/{steps} ce={ce.item():.3f} kl={float(klv.detach()):.4f} {el:.0f}s ~{el / (step + 1) * (steps - step - 1):.0f}s left", flush=True)
            log.append({"step": step + 1, "ce": round(ce.item(), 4), "kl": round(float(klv.detach()), 5)})
    state = {k: v.to(torch.bfloat16) for k, v in lora_state(model).items()}
    torch.save(state, os.path.join(out, "adapter.pt"))
    json.dump({"principal": a.principal, "frac": a.frac, "n": a.n, "args": vars(a), "data_report": rep, "log": log,
               "train_seconds": round(time.time() - t0)}, open(os.path.join(out, "train.json"), "w"), indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
