"""logits_to_keep=1 must give the same last-position log-probs as computing logits at every position (it only skips the
unembedding for positions we never read). Tiny random Qwen3 on CPU with left padding, as in the scoring code."""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


def test_last_position_logits_identical_under_left_padding():
    torch.manual_seed(0)
    cfg = AutoConfig.for_model("qwen3", vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").float().eval()
    ids = torch.randint(5, 512, (6, 17))
    att = torch.ones_like(ids)
    att[0, :5] = 0; att[3, :9] = 0                     # left padding on two rows
    ids[att == 0] = 0
    with torch.no_grad():
        full = F.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, -1].float(), -1)
        last = F.log_softmax(model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1].float(), -1)
    assert torch.allclose(full, last, atol=1e-5), (full - last).abs().max()
