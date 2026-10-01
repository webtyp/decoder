---
PLAN: "feat!: LFM2 architecture (short convolutions + plain attention) next to Qwen3.5; Config.Arch"
TAG: v0.4.0
EXECUTOR: jules
REVIEWER: none
STATUS: review
SESSION: 10258586629535649290
PR: https://github.com/webtyp/decoder/pull/3
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.

# Plan — `webtyp.com/decoder`: run LFM2 models

## 0. Context

`decoder` runs causal language models in Go for the browser: `decoder.New(cfg, artifact, prefix)`
loads weights, and `model.Step(state, token, logits)` computes the next token's logits. Today it
runs one architecture, **Qwen3.5** (Gated DeltaNet + gated attention). The agent needs a second
one: **LFM2** (Liquid AI; the model is LFM2.5-350M), a small model that writes good Spanish.

LFM2 is simpler than Qwen3.5. Each layer is a **short convolution** or an **attention** block,
followed by the same SwiGLU MLP Qwen3.5 uses. Every formula below is already verified. The numpy
file `testdata/lfm2_reference_step.py` implements one decode step exactly as described here and
matches PyTorch `transformers` within 2e-6 on the fixture. **Read it before stage 3. It is the
spec.** Run it with `~/Dev/LMmodels/.venv/bin/python testdata/lfm2_reference_step.py` if numpy is
available; you do not need to run it.

### What is already in `testdata/tiny_lfm2/` (do not regenerate)

- `config.json`: the tiny model's shape: `hidden_size` 64, `intermediate_size` 96,
  4 layers `["conv","conv","full_attention","conv"]`, 4 heads, 2 KV heads, head dim 16
  (= 64/4), `conv_L_cache` 3, `norm_eps` 1e-5, `rope_theta` 1e6, `vocab_size` 256, tied embeddings.
- `model.wtypw`: every tensor in Float32. `model_int8b32.wtypw`: every 2-D tensor in
  Int8Block32 and the rest in Float32.
- `reference.json`: the same keys as `testdata/tiny/reference.json` (`prompt`, `prefill_logits`,
  `greedy_tokens`, `greedy_step_logits`, `token_by_token_logits`).
- `gen_fixture_lfm2.py` (in `testdata/`) made them.

### Tensor names (prefix `model.`; shapes for the real LFM2.5-350M in brackets)

| Tensor | Shape | Storage |
|---|---|---|
| `embed_tokens.weight` | Vocab × Hidden [65536×1024] | matrix (tied: also the output projection) |
| `embedding_norm.weight` | Hidden | `[]float32`, the **final** norm |
| `layers.N.operator_norm.weight` | Hidden | `[]float32`, before the mixer |
| `layers.N.ffn_norm.weight` | Hidden | `[]float32`, before the MLP |
| `layers.N.feed_forward.w1.weight` | Inter × Hidden [4608×1024] | matrix: gate |
| `layers.N.feed_forward.w3.weight` | Inter × Hidden | matrix: up |
| `layers.N.feed_forward.w2.weight` | Hidden × Inter | matrix: down |
| conv layer: `layers.N.conv.in_proj.weight` | 3·Hidden × Hidden | matrix |
| conv layer: `layers.N.conv.conv.weight` | Hidden × 1 × K (K = 3) | `[]float32`, length Hidden·K |
| conv layer: `layers.N.conv.out_proj.weight` | Hidden × Hidden | matrix |
| attention: `layers.N.self_attn.q_proj.weight` | Heads·HeadDim × Hidden | matrix |
| attention: `layers.N.self_attn.k_proj.weight`, `v_proj.weight` | KVHeads·HeadDim × Hidden | matrix |
| attention: `layers.N.self_attn.out_proj.weight` | Hidden × Heads·HeadDim | matrix |
| attention: `layers.N.self_attn.q_layernorm.weight`, `k_layernorm.weight` | HeadDim | `[]float32` |

"matrix" means the existing `matrix` type (`matrix.go`, `loadMatrix`): Float32 or Int8Block32,
whatever the artifact holds.

### The three differences from Qwen3.5 that matter

1. **RMSNorm uses the weight as is** (`w · x/rms(x)`). Qwen3.5 uses `(1 + w)`, which `New`
   precomputes with `makeZeroCentered`. For LFM2, **do not call `makeZeroCentered`**. The same
   `nn.RMSNorm` call then works for both.
2. **Attention has no output gate.** `q_proj` gives `Heads·HeadDim` values, not twice that.
3. **RoPE rotates the whole head:** `RotaryDim == HeadDim`. The existing
   `ropeSlice(x, pos, rotaryDim, theta)` already does exactly LFM2's rotation when passed
   `HeadDim`.

## Development rules (inline)

- Library code compiles for the browser: `GOOS=js GOARCH=wasm go build ./...`. Never import `fmt`,
  `errors`, `strings`, `strconv` (use `webtyp.com/fmt`), `encoding/json`, `sort` or `map[K]V` in
  non-test files. Tests may use the standard library.
- `Step` keeps **0 allocations** for both architectures (`testing.AllocsPerRun`).
- Tests live in `tests/` (`package tests`). `ops_test.go`/`export_test.go` stay in the root, as
  `AGENTS.md` says.
- Max 500 lines per file: put the new code in new files (`config.go` changes stay in `config.go`).
- No `TODO`. `gotest` green at the end.

## Design gate (api-design — five answers)

1. **Prior art.** llama.cpp selects a graph per `general.architecture` (`qwen35`, `lfm2`, …).
   candle has one model file per architecture over shared ops. `transformers` maps
   `architectures[0]` to a class. All three dispatch on a typed architecture value and share the
   primitives, which is what `Config.Arch` does here.
2. **Novice-name test.** `decoder.Config{Arch: decoder.LFM2, …}`, and `decoder.ShortConv` for the
   layer kind. Both read as what they are.
3. **Complexity ledger.** Concepts +2 (`Arch`, `ShortConv`). Files at the call site +0. Lines +1
   (`Arch:`). Ways to do the same thing +0.
4. **Where it belongs.** `decoder` is "the graph, configurable per model", and LFM2 is one more
   graph over the same `nn` primitives and `matrix` storage.
5. **What it deletes.** Nothing. Existing Qwen3.5 configs must now say `Arch: decoder.Qwen35`; the
   zero value is invalid on purpose (closed by default).

## Stage 1 — `Config.Arch` and `ShortConv` (`config.go`, `errors.go`)

```go
// Arch is the model family: which tensors a checkpoint has and how a layer is computed.
type Arch uint8

const (
	Qwen35 Arch = iota + 1 // Qwen3.5: Gated DeltaNet + gated attention, (1+w) RMSNorm
	LFM2                   // LFM2: short convolutions + plain attention, w·x RMSNorm
)
```

- Add `ShortConv` to `LayerKind`, after `FullAttention`, with the comment
  `// LFM2: gated depthwise causal convolution, fixed-size state`.
- `Config` gains `Arch Arch` as its first field, with the comment
  `// required: Qwen35 or LFM2`.
- `Validate`:
  - `Arch` not in {Qwen35, LFM2} → `ErrInvalidArch` = `decoder: Config.Arch must be Qwen35 or LFM2`.
  - For **LFM2**, every layer kind must be `ShortConv` or `FullAttention`. For **Qwen35**, every
    one must be `LinearAttention` or `FullAttention`. Otherwise
    `ErrLayerKindForArch` = `decoder: a layer kind does not belong to Config.Arch`.
  - The `Linear*` checks run only for Qwen35. `ConvKernel` is required for both (LFM2's is 3).
  - The rest stay as they are.

## Stage 2 — loading LFM2 weights (`weights_lfm2.go`, `weights.go`)

- New types in `weights_lfm2.go`:

  ```go
  type shortConvWeights struct {
  	inProj  matrix    // [3*Hidden][Hidden]
  	conv    []float32 // [Hidden*ConvKernel]: channel c uses conv[c*K : c*K+K], index 0 the oldest input
  	outProj matrix    // [Hidden][Hidden]
  }
  type plainAttnWeights struct {
  	q, k, v, o   matrix    // q [Heads*HeadDim][Hidden], k,v [KVHeads*HeadDim][Hidden], o [Hidden][Heads*HeadDim]
  	qNorm, kNorm []float32 // [HeadDim], used as is
  }
  ```

- `layerWeights` gains `shortConv *shortConvWeights` and `plainAttn *plainAttnWeights`.
- `New`: `if cfg.Arch == LFM2 { return newLFM2(cfg, a, prefix) }`, before the Qwen3.5 code, which
  stays unchanged. `newLFM2` loads the tensors of §0 with `loadMatrix` (matrices) and `getTensor`
  (vectors and the conv weight, length `Hidden*ConvKernel`). Map them as follows:
  - `operator_norm` → the layer's input norm;
  - `ffn_norm` → its post-mixer norm;
  - `w1` → `GateProj`, `w3` → `UpProj`, `w2` → `DownProj`;
  - `embedding_norm` → `m.norm`.
  
  **No `makeZeroCentered` anywhere in `newLFM2`.**

## Stage 3 — the two LFM2 mixers (`shortconv.go`, `attention_plain.go`, `state.go`, `step.go`)

**State** (`state.go`): for a `ShortConv` layer, `ConvState = make([]float32, Hidden*(ConvKernel-1))`.
It is laid out channel by channel: `ConvState[c*(K-1)+j]`, where `j = 0` is the oldest. `FullAttention`
layers of LFM2 use `KCache`/`VCache` as Qwen3.5 does. `Scratch` gains `BCX []float32 // [3*Hidden]` and
`ConvY []float32 // [Hidden]`, allocated in `newScratch` only when `Arch == LFM2` (Qwen3.5 scratch
unchanged). `CopyFrom` already copies `ConvState`, `KCache` and `VCache`; it needs no change.

**`stepShortConv(cfg, w, st, scr, h)`** (`shortconv.go`), exactly the numpy spec:

1. `w.inProj.mulVec(scr.BCX, h)`. `B = BCX[0:H]`, `C = BCX[H:2H]`, `x = BCX[2H:3H]`.
2. For each channel `c`: `bx = B[c]*x[c]`, and
   `y = Σ_{j<K-1} ConvState[c*(K-1)+j]·conv[c*K+j] + bx·conv[c*K+K-1]`.
   Then shift the channel's state left by one and store `bx` as the newest value
   (`ConvState[c*(K-1)+K-2] = bx`). `scr.ConvY[c] = C[c]*y`.
3. `w.outProj.mulVec(<the mixer output buffer>, scr.ConvY)`. Reuse `scr.LinearProjO` if it is
   Hidden long; otherwise add a `MixerOut []float32 // [Hidden]` to `Scratch`.

There is **no SiLU** in this convolution, unlike `causalConvStep` in `ops.go`. Do not reuse that
function.

**`stepPlainAttention(cfg, w, st, scr, p, h)`** (`attention_plain.go`):

1. `q = w.q·h` (Heads × HeadDim), `k = w.k·h`, `v = w.v·h` (KVHeads × HeadDim). Use the existing
   `scr.QG` for `q` (its first Heads·HeadDim values), and `scr.K`, `scr.V`.
2. Per q head and per k head: `nn.RMSNorm(head, head, w.qNorm|w.kNorm, HeadDim, Eps)`, then
   `ropeSlice(head, p, cfg.RotaryDim, cfg.RopeTheta)` (LFM2 configs set `RotaryDim = HeadDim`).
3. Append `k`, `v` to `KCache`, `VCache`. For each q head `i`, with KV head `g = i / (Heads/KVHeads)`:
   scores over all cached positions `= dot(q_i, k_t,g) / sqrt(HeadDim)`, softmax, weighted sum of
   `v_t,g`. Follow `stepFullAttention`'s loop, without the gate.
4. `w.o.mulVec(<mixer output>, attn)`.

**`Step`** (`step.go`): for `Arch == LFM2`, the mixer of a `ShortConv` layer is `stepShortConv` and
of a `FullAttention` layer is `stepPlainAttention`. The rest of `Step` (input norm, residual adds,
post norm, MLP, final norm, tied output) is shared and unchanged. The norms are already plain
weights for LFM2 (stage 2).

## Stage 4 — tests (`tests/lfm2_test.go`, existing tests)

- Existing tests: `tinyConfig()` in `tests/step_test.go` (and any other test config) gains
  `Arch: decoder.Qwen35`. Every existing test stays green.
- `tests/lfm2_test.go`, with `tinyLFM2Config()`:
  `Arch: LFM2, Vocab 256, Hidden 64, Intermediate 96, Layers [ShortConv, ShortConv, FullAttention, ShortConv], Heads 4, KVHeads 2, HeadDim 16, RotaryDim 16, RopeTheta 1e6, ConvKernel 3, Eps 1e-5`.
  The `Linear*` fields stay zero. The prefix is `"model."`.
  1. Token by token from an empty state, `model.wtypw` matches `token_by_token_logits` within
     **1e-4** at every position.
  2. The prompt, then the 8 `greedy_tokens`: the logits after each match `greedy_step_logits`
     within 1e-4, and the argmax after the prompt is `greedy_tokens[0]`.
  3. `model_int8b32.wtypw`: the greedy next token equals the reference's at every prompt position,
     and the max |logit − `prefill_logits`| is below **0.1**. Measured beforehand in PyTorch with
     the same quantization: 0.0549 on logits from −2.2 to 2.2. Log the value with `t.Logf`.
  4. `testing.AllocsPerRun` of `Step` on the LFM2 model is 0.
  5. `CopyFrom` on an LFM2 state resumes identically (same pattern as `tests/copyfrom_test.go`).
  6. `Validate`: `Arch` zero → `ErrInvalidArch`; LFM2 with a `LinearAttention` layer →
     `ErrLayerKindForArch`.

## Stage 5 — docs

- `README.md`: the "I want X → use Y" table gets "Load an LFM2 model" → `Config{Arch: decoder.LFM2, …}`.
  The usage example sets `Arch: decoder.Qwen35`.
- `docs/ARCHITECTURE.md`: a section "Two architectures". It covers what LFM2 layers are, the three
  differences of §0, and that `testdata/lfm2_reference_step.py` is the spec.

**Out of scope:** `webtyp.com/qwen` sets `Arch: decoder.Qwen35` in its own repository, after this
version is published. Do not look for it.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `config.go`, `errors.go` | Validate tests |
| 2 | `weights_lfm2.go`, `weights.go` | the LFM2 fixture loads |
| 3 | `shortconv.go`, `attention_plain.go`, `state.go`, `step.go` | `tests/lfm2_test.go` 1–5 |
| 4 | `tests/` | all tests green, Qwen3.5 tests unchanged in behavior |
| 5 | docs | both mention LFM2 |
| all | — | `gotest` green; `GOOS=js GOARCH=wasm go build ./...` |
