# Architecture — `webtyp/decoder`

## What this is

A **decoder** is the half of a language model that generates text. It reads the tokens so
far and produces a score (a *logit*) for every possible next token. Then the next token is
chosen, appended, and the step repeats. This repository computes that step in Go, inside the
browser.

You meet it through `webtyp/qwen`, which turns it into an `llm.Client`. Nothing else should
use it directly.

It exists separately from `webtyp/encoder` because generating is a different job from
encoding. The decoder keeps state between tokens, predicts one token at a time, and never pools
into a single vector.

## Two architectures

The decoder supports two model families via `Config.Arch`:

1. **Qwen3.5** (`decoder.Qwen35`): Gated DeltaNet linear attention + gated full attention, with zero-centered `(1 + w)` RMSNorm on layernorms.
2. **LFM2** (`decoder.LFM2`): Gated short depthwise causal convolutions (`ShortConv`) + plain full attention (`FullAttention`), followed by SwiGLU MLP.

### Three key differences in LFM2

1. **RMSNorm uses weights as is:** `w · x / rms(x)`. Qwen3.5 uses `(1 + w)` zero-centered weights.
2. **Plain attention without output gate:** Query projection outputs `Heads · HeadDim` values (no output gate multiplier).
3. **Full-head RoPE:** RoPE rotates all dimensions of each head (`RotaryDim == HeadDim`).

The executable spec for the LFM2 decode step is `testdata/lfm2_reference_step.py`.

## The first model it must run: Qwen3.5-0.8B

Measured from the model's `config.json` and weights (see the ecosystem master plan, D6):

| Property | Value |
|---|---|
| layers | 24, in the repeating pattern 3 × linear attention, 1 × full attention |
| hidden size | 1024; FFN 3584 with SiLU gate (SwiGLU) |
| full-attention layers (6) | 8 query heads, 2 key/value heads (GQA), head size 256, output gate, RMSNorm on q and k, RoPE |
| linear-attention layers (18) | **Gated DeltaNet**: 16 heads × 128, a causal 1-D convolution of width 4 before it, a recurrent state of 128 × 128 per head |
| vocabulary | 248 320 tokens; the output projection **reuses** the input embedding table |

## What it keeps between tokens

- Full-attention layers keep a **KV cache** that grows with the text: about 6 KB per token for
  the 6 layers in float32.
- Gated DeltaNet layers keep a **fixed-size state** (~1 MB per layer, ~18 MB in total), however
  long the text is. This is why the model fits a browser tab even with long conversations.

## Weights in memory

The decoder accepts artifacts stored as Float32 or Int8Block32 (`weightsc -quant int8-block32`). Quantized 2-D weight matrices remain stored in memory as int8 blocks (with one float32 scale per 32 values) rather than dequantizing on load. Dequantizing Qwen3.5-0.8B to float32 would consume ~2 GB of memory, exceeding the heap limits available to browser tabs on 4 GB memory devices. Matrix-vector products are computed directly over int8 blocks via `nn.MatVecInt8Block32`.

## What it reuses and what is new

| Piece | Where |
|---|---|
| matmul, RMSNorm, SiLU, softmax, RoPE | `webtyp/nn` (already verified in the encoder) |
| weight reading | `webtyp/weights` |
| causal 1-D convolution, gated delta rule, GQA with a KV cache, output gate | **new**, here or in `nn` if the speech models need them too |

## How it will be built

The same method `embed` used: first **correct**, then **fast**.

1. Correct in plain Go, single thread, verified against the original model: logits compared
   against Hugging Face `transformers` on the downloaded safetensors, and generated tokens
   compared against `llama-server` with the GGUF.
2. Measure under TinyGo/WASM.
3. Speed: SIMD, several Web Workers, WebGPU, in the order the measurements justify.
