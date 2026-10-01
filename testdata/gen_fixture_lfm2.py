"""Generates the tiny LFM2 fixture the decoder is verified against.

Same architecture as LFM2.5-350M (gated short convolutions + grouped-query attention with q/k
RMSNorm, plain RMSNorm weight, full-head RoPE, SwiGLU MLP, a final embedding_norm, tied
embeddings), tiny sizes, random weights. Run with the reference environment:

    ~/Dev/LMmodels/.venv/bin/python testdata/gen_fixture_lfm2.py

Writes testdata/tiny_lfm2/config.json, testdata/tiny_lfm2/model.safetensors (float32, the real
model's tensor names) and testdata/tiny_lfm2/reference.json (same keys as testdata/tiny).
Deterministic (seed 1234).
"""
import json, os
import torch
from safetensors.torch import save_file
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

torch.manual_seed(1234)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_lfm2")
os.makedirs(out, exist_ok=True)

cfg = Lfm2Config(
    vocab_size=256, hidden_size=64, intermediate_size=96, block_auto_adjust_ff_dim=False,
    num_hidden_layers=4, layer_types=["conv", "conv", "full_attention", "conv"],
    num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, conv_bias=False,
    norm_eps=1e-5, tie_word_embeddings=True, max_position_embeddings=4096,
    rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}, dtype="float32",
)
model = Lfm2ForCausalLM(cfg).float().eval()

with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name:
            p.copy_(1.0 + torch.empty_like(p).uniform_(-0.2, 0.2))
        else:
            p.copy_(torch.randn_like(p) * 0.08)

state = {k: v.contiguous().clone() for k, v in model.state_dict().items() if k != "lm_head.weight"}
save_file(state, os.path.join(out, "model.safetensors"))
text_cfg = {k: v for k, v in cfg.to_dict().items() if not k.startswith("_")}
json.dump(text_cfg, open(os.path.join(out, "config.json"), "w"), indent=1, default=str)

prompt = [11, 200, 7, 42, 99, 3, 150, 64, 255, 1]
with torch.no_grad():
    full = model(torch.tensor([prompt]), use_cache=True)
    prefill_logits = full.logits[0].tolist()
    past = full.past_key_values
    nxt = int(full.logits[0, -1].argmax())
    generated, step_logits = [], []
    for _ in range(8):
        generated.append(nxt)
        o = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        step_logits.append(o.logits[0, -1].tolist())
        nxt = int(o.logits[0, -1].argmax())
    past = None
    seq_logits = []
    for t in prompt:
        o = model(torch.tensor([[t]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        seq_logits.append(o.logits[0, -1].tolist())

json.dump({"prompt": prompt, "prefill_logits": prefill_logits, "greedy_tokens": generated,
           "greedy_step_logits": step_logits, "token_by_token_logits": seq_logits},
          open(os.path.join(out, "reference.json"), "w"))
print("max |prefill - token_by_token| =",
      max(abs(a - b) for ra, rb in zip(prefill_logits, seq_logits) for a, b in zip(ra, rb)))
for k, v in sorted(state.items()):
    print(k, list(v.shape))
