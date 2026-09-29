"""Generates the tiny Qwen3.5 fixture the decoder is verified against.

Same architecture as Qwen3.5-0.8B (hybrid Gated DeltaNet + gated full attention, zero-centered
RMSNorm, partial RoPE, tied embeddings), tiny sizes, random weights. Run with the reference
environment (torch CPU + transformers >= 5.17):

    ~/Dev/LMmodels/.venv/bin/python testdata/gen_fixture.py

Writes testdata/tiny/config.json, testdata/tiny/model.safetensors (float32, the real model's
tensor names) and testdata/tiny/reference.json. Deterministic (seed 1234).
"""
import json, os
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

torch.manual_seed(1234)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny")
os.makedirs(out, exist_ok=True)

cfg = Qwen3_5TextConfig(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    full_attention_interval=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    attn_output_gate=True, linear_num_key_heads=4, linear_num_value_heads=4,
    linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
    hidden_act="silu", rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=4096,
    rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                     "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    mtp_num_hidden_layers=0, dtype="float32",
)
model = Qwen3_5ForCausalLM(cfg).float().eval()

# Random, non-trivial values everywhere, including the zero-initialized norms, so that the
# (1 + w) of the zero-centered RMSNorm is actually exercised.
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4.0)))
        elif name.endswith("dt_bias"):
            p.copy_(torch.empty_like(p).uniform_(-0.5, 0.5))
        elif "norm" in name:
            p.copy_(torch.empty_like(p).uniform_(-0.2, 0.2))
        else:
            p.copy_(torch.randn_like(p) * 0.08)

# Save with the REAL model's tensor names (model.language_model.*), float32, no lm_head (tied).
state = {}
for k, v in model.state_dict().items():
    if k == "lm_head.weight":
        continue
    state["model.language_model." + k[len("model."):]] = v.contiguous().clone()
save_file(state, os.path.join(out, "model.safetensors"))

text_cfg = {k: v for k, v in cfg.to_dict().items() if not k.startswith("_")}
json.dump(text_cfg, open(os.path.join(out, "config.json"), "w"), indent=1, default=str)

prompt = [11, 200, 7, 42, 99, 3, 150, 64, 255, 1]
with torch.no_grad():
    # 1. Prefill: logits for every prompt position in one forward pass.
    full = model(torch.tensor([prompt]), use_cache=True)
    prefill_logits = full.logits[0].tolist()
    # 2. Greedy decode 8 tokens one at a time through the cache (recurrent + KV paths).
    past = full.past_key_values
    nxt = int(full.logits[0, -1].argmax())
    generated, step_logits = [], []
    for _ in range(8):
        generated.append(nxt)
        o = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        step_logits.append(o.logits[0, -1].tolist())
        nxt = int(o.logits[0, -1].argmax())
    # 3. The same prompt+generation fed token by token from scratch (pure decode path).
    past = None
    seq_logits = []
    for t in prompt:
        o = model(torch.tensor([[t]]), past_key_values=past, use_cache=True)
        past = o.past_key_values
        seq_logits.append(o.logits[0, -1].tolist())

ref = {
    "prompt": prompt,
    "prefill_logits": prefill_logits,          # [len(prompt)][vocab]
    "greedy_tokens": generated,                # 8 tokens after the prompt
    "greedy_step_logits": step_logits,         # logits after feeding each greedy token
    "token_by_token_logits": seq_logits,       # prompt fed one token at a time from empty state
}
json.dump(ref, open(os.path.join(out, "reference.json"), "w"))
print("max |prefill - token_by_token| =",
      max(abs(a - b) for ra, rb in zip(prefill_logits, seq_logits) for a, b in zip(ra, rb)))
print("wrote", out)
