"""LoRA sanity tests on a tiny random Qwen3-style model (CPU, no download)."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "organism"))
from transformers import AutoConfig, AutoModelForCausalLM
from lora import add_lora, set_adapters, lora_state

def tiny():
    cfg = AutoConfig.for_model("qwen3", vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()

def test_all():
    m = tiny(); ids = torch.randint(0, 300, (2, 9))
    base = m(ids).logits.detach().clone()
    n = add_lora(m, 4, 8.0)
    assert n == 2 * 7, n                                                    # 7 target projections per layer
    assert torch.allclose(m(ids).logits, base, atol=1e-6)                   # B=0 => adapter is an exact no-op
    trainable = [k for k, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(k.endswith(".A") or k.endswith(".B") for k in trainable)
    # perturb B so the adapter does something
    for k, p in m.named_parameters():
        if k.endswith(".B"): torch.nn.init.normal_(p, std=0.05)
    on = m(ids).logits
    assert not torch.allclose(on, base, atol=1e-4)
    set_adapters(False); off = m(ids).logits; set_adapters(True)
    assert torch.allclose(off, base, atol=1e-6)                             # adapters off == frozen base
    loss = m(ids, labels=ids).loss; loss.backward()
    assert all(p.grad is not None for k, p in m.named_parameters() if p.requires_grad)
    assert all(p.grad is None for k, p in m.named_parameters() if not p.requires_grad)   # base frozen
    st = lora_state(m); assert all(k.endswith((".A", ".B")) for k in st) and len(st) == 2 * n
    m2 = tiny(); add_lora(m2, 4, 8.0); m2.load_state_dict(st, strict=False)
    assert torch.allclose(m2(ids).logits, on, atol=1e-5)                    # save/load round trip
    print("LoRA tests passed")

if __name__ == "__main__":
    test_all()
