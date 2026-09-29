---
PLAN: "feat: decoder — Qwen3.5 hybrid causal decoder in plain Go (Gated DeltaNet + gated attention), verified token by token"
TAG: v0.1.0
EXECUTOR: jules
REVIEWER: none
STATUS: review
SESSION: 15261525408189116169
PR: https://github.com/webtyp/decoder/pull/1
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.
>
> Part of
> [`AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).
> Depends on published tags only: `webtyp.com/nn` v0.1.0, `webtyp.com/weights` v0.1.0.

# Plan — `webtyp.com/decoder`: generate text one token at a time, in the browser

## 0. Context

A **decoder** is the half of a language model that generates text. It reads the tokens so far
and produces a *logit* (a score) for every possible next token. The next token is chosen,
appended, and the step repeats. The first model webtyp runs this way in the browser is
**Qwen3.5-0.8B**, in Go compiled with TinyGo, like `webtyp/encoder` already does for the
embedding model.

Qwen3.5 is **hybrid**: in every group of 4 layers, 3 are *Gated DeltaNet* layers (a linear
attention that keeps a fixed-size matrix state per head) and 1 is a *full attention* layer
(keeps a growing key/value cache). This plan implements that decoder **for correctness only**:
plain Go, float32 weights, one token per `Step`. Speed (SIMD, int8, prefill in blocks) is a later
plan and must not be attempted here.

**Everything needed to verify the work is in this repository.** `testdata/tiny/` holds a tiny
Qwen3.5 (same architecture, 4 layers, 64-wide, random weights) and the logits the reference
implementation (Hugging Face `transformers` 5.17, PyTorch) produced for it:

| File | What it is |
|---|---|
| `testdata/tiny/model.wtypw` | the tiny model's weights, float32, **with the real model's tensor names**, readable with `weights.Open` |
| `testdata/tiny/config.json` | its configuration (for reference; the test builds `Config` by hand, see Stage 4) |
| `testdata/tiny/reference.json` | `prompt` (10 token ids), `token_by_token_logits` (logits after each prompt token), `prefill_logits` (same, computed in one pass), `greedy_tokens` (8 tokens greedily generated after the prompt), `greedy_step_logits` (logits after feeding each greedy token) |
| `testdata/gen_fixture.py`, `testdata/tiny_to_wtypw.go.txt` | how the fixture was produced (do **not** run them; no Python is needed) |

## Development rules (inline)

- **Primary runtime: browser, TinyGo/WASM.** Every file compiles under `GOOS=js GOARCH=wasm` and TinyGo.
- **Plain Go, one implementation.** No SIMD, no assembly, no build tags, no goroutines.
- **Never import:** `fmt`, `errors`, `strings`, `strconv` (use `webtyp.com/fmt`), stdlib `context`,
  `sort`, `map[K]V`, `os`, `log`, `net/http` in non-test files. `math` is allowed. Tests may use
  `os` (to read `testdata`) and `encoding/json` (to read `reference.json`).
- **Reuse, never copy:** `nn.MatmulT`, `nn.RMSNorm`, `nn.SiLU`, `nn.Softmax` from `webtyp.com/nn`;
  `weights.Open` / `Artifact.Tensor` / `Tensor.Float32s` from `webtyp.com/weights`.
- **No allocation inside `Step`.** Every scratch buffer is allocated in `New` / `NewState`.
- Max 500 lines per file. Error messages are constants (`errors.go`). Do **not** run `gopush`/`codejob`.

## Design gate (api-design — five answers)

1. **Prior art.** **llama.cpp** (`llama_decode` over a batch, a KV cache object per context),
   **Hugging Face `transformers`** (`model(input_ids, past_key_values=cache)` → logits),
   **ggml/mamba implementations** (a recurrent state per layer carried between calls). All three
   separate the immutable model (weights) from the mutable per-sequence state (cache). We do the
   same: `Model` is immutable and shareable, `State` is one conversation.
2. **Novice-name test.** `decoder.New(cfg, artifact, prefix)` → `*Model`; `model.NewState()` →
   `*State`; `model.Step(state, token, logits)` reads as "step the model with this token and write
   the next-token logits". `LayerKind` values are `LinearAttention` and `FullAttention`, the names
   the model's own `config.json` uses.
3. **Complexity ledger.** New repository: concepts +4 (`Config`, `Model`, `State`, `Step`). Ways to
   do the same thing +0.
4. **Where it belongs.** The graph of a causal decoder, configurable per model, like `encoder` is
   for encoders. Model-specific things (tokenizer, chat template, which config a checkpoint has)
   belong to `webtyp/qwen`, which builds a `decoder.Config` from the model's `config.json`.
5. **What it deletes.** Nothing. This is new capability.

## Stage 1 — configuration (`config.go`)

```go
package decoder

// LayerKind is the token mixer of one layer.
type LayerKind uint8

const (
	LinearAttention LayerKind = iota // Gated DeltaNet: fixed-size recurrent state
	FullAttention                    // gated, causal, grouped-query attention: growing KV cache
)

// Config is the shape of one decoder checkpoint.
type Config struct {
	Vocab        int         // vocabulary size (rows of the embedding table)
	Hidden       int         // model width
	Intermediate int         // MLP width
	Layers       []LayerKind // one entry per layer, in order

	Heads     int     // full attention: query heads
	KVHeads   int     // full attention: key/value heads (Heads % KVHeads == 0)
	HeadDim   int     // full attention: size of one head
	RotaryDim int     // full attention: leading dims of each head that RoPE rotates (HeadDim × partial_rotary_factor)
	RopeTheta float64 // full attention: RoPE base

	LinearKeyHeads   int // DeltaNet: key heads
	LinearValueHeads int // DeltaNet: value heads (multiple of LinearKeyHeads)
	LinearKeyDim     int // DeltaNet: size of one key head
	LinearValueDim   int // DeltaNet: size of one value head
	ConvKernel       int // DeltaNet: causal convolution width

	Eps float32 // RMSNorm epsilon
}
```

Add `func (c Config) Validate() error` checking every field > 0, `len(Layers) > 0`,
`Heads % KVHeads == 0`, `LinearValueHeads % LinearKeyHeads == 0`, `RotaryDim` even and
`<= HeadDim`. Messages: `decoder: Config.<Field> must be greater than zero`, etc.

## Stage 2 — weights (`weights.go`)

`func New(cfg Config, a *weights.Artifact, prefix string) (*Model, error)`. It validates `cfg`,
then reads every tensor below as float32 (`Tensor.Float32s`). A missing tensor returns
`decoder: missing tensor <name>`, and a wrong element count returns
`decoder: tensor <name> has <n> values, want <m>`. For Qwen3.5 the prefix is
`model.language_model.` (the fixture uses the same names).

| Name (after prefix) | Shape (HF `[out][in]`) | Notes |
|---|---|---|
| `embed_tokens.weight` | `[Vocab][Hidden]` | also the output projection (tied): logits = embed · x |
| `norm.weight` | `[Hidden]` | final norm, zero-centered (see below) |
| `layers.L.input_layernorm.weight`, `layers.L.post_attention_layernorm.weight` | `[Hidden]` | zero-centered |
| `layers.L.mlp.gate_proj.weight`, `…up_proj…` | `[Intermediate][Hidden]` | |
| `layers.L.mlp.down_proj.weight` | `[Hidden][Intermediate]` | |
| **FullAttention** `layers.L.self_attn.q_proj.weight` | `[Heads·2·HeadDim][Hidden]` | per head: HeadDim query values then HeadDim gate values |
| `…k_proj.weight`, `…v_proj.weight` | `[KVHeads·HeadDim][Hidden]` | |
| `…o_proj.weight` | `[Hidden][Heads·HeadDim]` | |
| `…q_norm.weight`, `…k_norm.weight` | `[HeadDim]` | zero-centered |
| **LinearAttention** `layers.L.linear_attn.in_proj_qkv.weight` | `[2·Kh·Kd + Vh·Vd][Hidden]` | output order: all q, all k, all v |
| `…in_proj_z.weight` | `[Vh·Vd][Hidden]` | |
| `…in_proj_b.weight`, `…in_proj_a.weight` | `[Vh][Hidden]` | |
| `…conv1d.weight` | `[2·Kh·Kd + Vh·Vd][1][ConvKernel]` | depthwise |
| `…A_log`, `…dt_bias` | `[Vh]` | |
| `…norm.weight` | `[Vd]` | **plain** weight (NOT zero-centered) |
| `…out_proj.weight` | `[Hidden][Vh·Vd]` | |

(Kh = `LinearKeyHeads`, Kd = `LinearKeyDim`, Vh = `LinearValueHeads`, Vd = `LinearValueDim`.)

**Zero-centered RMSNorm.** Qwen3.5's `RMSNorm` computes `x · rsqrt(mean(x²) + eps) · (1 + w)`.
At load, store `gamma = 1 + w` for every zero-centered norm, and call `nn.RMSNorm(dst, src, gamma, dim, eps)`.
The DeltaNet gated norm (`linear_attn.norm`) uses `w` as is.

Linear projections use `nn.MatmulT(dst, x, W, 1, in, out)`, where `W` is the HF `[out][in]` tensor
(that is exactly `MatmulT`'s transposed-B layout).

## Stage 3 — the step (`state.go`, `step.go`, `attention.go`, `deltanet.go`, `ops.go`)

`State` holds, per layer, the KV cache (full attention: keys and values of every position so far,
`[pos][KVHeads·HeadDim]`), the convolution state (DeltaNet: the last `ConvKernel−1` inputs of each
of the `2·Kh·Kd + Vh·Vd` channels, initially zero), the recurrent state (DeltaNet:
`Vh × Kd × Vd`, initially zero), and `Pos` (tokens fed so far). The KV cache grows with `append`.
This is the only allowed growth.

`func (m *Model) Step(st *State, token int, logits []float32) error`:
`len(logits)` must be `Vocab`, and `0 <= token < Vocab`. Otherwise return an error.

```
x = embed[token]                                         // copy of one row, Hidden values
for each layer L:
    h = RMSNorm(x, gamma_input[L])
    x += FullAttention(h) or DeltaNet(h)                  // by Layers[L]
    h = RMSNorm(x, gamma_post[L])
    x += down · ( SiLU(gate · h) ⊙ (up · h) )
x = RMSNorm(x, gamma_final)
logits = embed · x                                        // MatmulT(logits, x, embed, 1, Hidden, Vocab)
st.Pos++
```

**FullAttention(h)** at position `p = st.Pos`:
1. `qg = q_proj · h`. For head `i`, `q_i = qg[i·2·HeadDim : i·2·HeadDim+HeadDim]` and
   `gate_i = qg[i·2·HeadDim+HeadDim : (i+1)·2·HeadDim]`.
2. `k = k_proj · h`, `v = v_proj · h`, `KVHeads` heads each.
3. `q_i = RMSNorm(q_i, gamma_qnorm)`, `k_j = RMSNorm(k_j, gamma_knorm)` (per head, over `HeadDim`).
4. **Partial RoPE** on the first `RotaryDim` values of every `q_i` and `k_j`, GPT-NeoX halves.
   For `d in [0, RotaryDim/2)`: `inv = RopeTheta^(−2d/RotaryDim)`, `θ = p·inv`, and with
   `a = x[d]`, `b = x[d + RotaryDim/2]`: `x[d] = a·cosθ − b·sinθ`, `x[d + RotaryDim/2] = b·cosθ + a·sinθ`.
   The remaining `HeadDim − RotaryDim` values are unchanged. Compute in float64, store float32.
5. Append `k`, `v` to this layer's cache.
6. For query head `i`, key/value head `j = i / (Heads/KVHeads)`: scores over positions `0..p`
   are `q_i · k_j[t] / sqrt(HeadDim)`, then `nn.Softmax`, and `o_i = Σ_t score_t · v_j[t]`.
7. `o = concat(o_0 … o_{Heads−1})`, then `o[c] *= sigmoid(gate[c])` element-wise (gate concatenated
   in the same head order). Return `o_proj · o`.

**DeltaNet(h)**:
1. `mixed = in_proj_qkv · h`, `z = in_proj_z · h`, `b = in_proj_b · h`, `a = in_proj_a · h`.
2. **Causal convolution**, per channel `c`: `window = [convState[c][0 … K−2], mixed[c]]` (oldest
   first, `K = ConvKernel`), then `y[c] = Σ_{j=0}^{K−1} conv[c][0][j] · window[j]`. Then `y[c] = SiLU(y[c])`.
   Then shift: `convState[c] = window[1 … K−1]`.
3. Split `y` into `q` (`Kh·Kd`), `k` (`Kh·Kd`), `v` (`Vh·Vd`), in that order.
4. For each value head `h`: `β_h = sigmoid(b[h])`, `g_h = −exp(A_log[h]) · softplus(a[h] + dt_bias[h])`
   with `softplus(x) = x` when `x > 20`, else `log(1 + exp(x))`.
5. If `Vh > Kh`, value head `h` uses key head `h / (Vh/Kh)` (each key head serves `Vh/Kh`
   consecutive value heads).
6. Per head: `q ← q · rsqrt(Σq² + 1e−6)`, `k ← k · rsqrt(Σk² + 1e−6)` (L2 normalize), then
   `q ← q / sqrt(Kd)`.
7. **Gated delta rule** on the state `S` (`Kd × Vd`) of head `h`:
   `S ← S · exp(g_h)`; `mem[v] = Σ_i S[i][v]·k[i]`; `δ[v] = (v[v] − mem[v]) · β_h`;
   `S[i][v] += k[i]·δ[v]`; `o[v] = Σ_i S[i][v]·q[i]`.
8. **Gated norm** per head over `Vd`: `o ← o · rsqrt(mean(o²) + Eps) · norm_w · SiLU(z_h)`
   (`norm_w` is the plain `linear_attn.norm.weight`, `z_h` is head `h`'s slice of `z`).
9. Return `out_proj · concat(o_0 … o_{Vh−1})`.

`ops.go` holds the small helpers: `sigmoid`, `softplus`, `l2Normalize`, `ropePartial`,
`causalConvStep`, `deltaRuleStep`. They are unexported, and each has a unit test with
hand-computed values (for example `softplus(0) = ln 2`, and a 2-channel convolution with known
weights).

## Stage 4 — verification against the reference (`step_test.go`)

The test builds the tiny `Config` by hand:

```go
cfg := decoder.Config{
	Vocab: 256, Hidden: 64, Intermediate: 128,
	Layers: []decoder.LayerKind{decoder.LinearAttention, decoder.LinearAttention, decoder.LinearAttention, decoder.FullAttention},
	Heads: 4, KVHeads: 2, HeadDim: 32, RotaryDim: 8, RopeTheta: 10000000,
	LinearKeyHeads: 4, LinearValueHeads: 4, LinearKeyDim: 16, LinearValueDim: 16, ConvKernel: 4,
	Eps: 1e-6,
}
```

It loads `testdata/tiny/model.wtypw` with `weights.Open` and `decoder.New(cfg, art, "model.language_model.")`.

| Test | Asserts |
|---|---|
| `TestStep_MatchesReferenceTokenByToken` | feeding `prompt` one token at a time, the logits after each token match `token_by_token_logits` with max absolute difference ≤ 1e-4 |
| `TestStep_MatchesPrefill` | the same logits also match `prefill_logits` (≤ 1e-4) |
| `TestStep_GreedyContinuation` | after the prompt, argmax of the last logits is `greedy_tokens[0]`; feeding each greedy token, logits match `greedy_step_logits` (≤ 1e-4) and argmax gives the next greedy token |
| `TestStep_StatesAreIndependent` | two `State`s from one `Model`, fed different tokens, do not affect each other (feed state A the prompt, state B token 5, then state A's next logits still match the reference) |
| `TestNew_MissingTensor` | an artifact without `norm.weight` → `decoder: missing tensor model.language_model.norm.weight` |
| `TestStep_RejectsBadInput` | `token = Vocab` and `len(logits) != Vocab` → errors |
| ops unit tests | as described in Stage 3 |

If a reference comparison fails, **do not loosen the tolerance**. Compare layer by layer with the
math in Stage 3. The fixture was checked to be self-consistent (prefill vs token by token agree to
2.7e-6).

Add `BenchmarkStep_Tiny` (one `Step` on the tiny model) so later speed plans have a baseline.

## Stage 5 — docs

`README.md`: remove the `STATUS` note, and add the "I want X → use Y" table (`New`, `NewState`,
`Step`) plus the three-line usage example from Stage 4. `docs/ARCHITECTURE.md` already describes
the design. Check that it matches, and correct it if something differs.

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `config.go`, `errors.go`, `go.mod` | `Validate` tests pass |
| 2 | `weights.go` | `TestNew_MissingTensor` passes |
| 3 | `state.go`, `step.go`, `attention.go`, `deltanet.go`, `ops.go` | ops tests pass; `grep -rn '"fmt"\|"strings"\|"errors"\|map\[' --include=*.go . \| grep -v _test.go` → empty |
| 4 | `step_test.go`, `ops_test.go` | all reference tests pass under `gotest` **and** `gotest -tinygo` |
| 5 | `README.md` | no `STATUS` line |
