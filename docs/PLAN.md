---
PLAN: "feat: matrices in Int4Block32 — the decoder runs 4-bit weights with nn.MatVecQ4Block32"
TAG: v0.6.0
EXECUTOR: jules
REVIEWER: none
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.


# Plan — `decoder` v0.6.0: 4-bit matrices

**Read [AGENTS.md](../AGENTS.md) first**: browser library, `gotest -tinygo` decides, forbidden
imports, `Step` allocates nothing. Master plan:
[AGENT_ECOSYSTEM_MASTER_PLAN.md](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md)
(D13: 4-bit blocks).

## Why

With int8 matrices the browser tab holds 2.9 GB (decider-0.8b + LFM2.5-350M); the clinic's PCs have
4 GB. `weights` v0.4.0 stores matrices in `Int4Block32` and `nn` v0.5.0 multiplies them. `decoder`
must load such a tensor **without converting it** (the bytes stay 4-bit in memory) and use the new
kernel. Every Qwen3.5 and LFM2 matrix the decoder loads goes through `loadMatrix` (`matrix.go`), so
this is one file plus tests.

The published pieces:

- `weights.Int4Block32` — layout: rows×cols, cols a multiple of 32; per row `cols/32` blocks of 16
  bytes; byte j of a block = column j in the low nibble, column j+16 in the high nibble; value =
  (nibble − 8) × scale; `Data` has rows×cols/2 bytes, `Scales` rows×cols/32.
- `weights.DequantInt4Block32(dst []float32, q []byte, scales []float32)` — unpacks whole blocks.
- `weights.QuantizeInt4Block32(row []float32) (q []byte, scales []float32, err error)`.
- `nn.MatVecQ4Block32(dst []float32, xq []int8, xs []float32, q []byte, scales []float32, rows, cols int) error`
  — same contract as `nn.MatVecQ8Block32`, input from `nn.QuantizeBlocks32`.

## Design gate

1. **Prior art.** llama.cpp keeps each tensor in its quantized type and dispatches the matching
   `vec_dot`; same here.
2. **Novice-name test.** No new exported API: `decoder.New` accepts one more tensor type.
3. **Complexity ledger.** +1 storage case in an unexported type.
4. **Where it belongs.** Here (`matrix.go`).
5. **What it deletes.** Nothing.

## Stage 1 — `go.mod`

`go get webtyp.com/weights@v0.4.0 webtyp.com/nn@v0.5.0`, `go mod tidy`.

## Stage 2 — `matrix.go`

- `matrix` gets `q4 bool // q holds Int4Block32 (rows×cols/2 bytes) instead of Int8Block32`.
- `loadMatrix`: new `case weights.Int4Block32:` — `cols%weights.BlockSize != 0` →
  `fmt.ErrType(makeErr(fmt.Sprintf("%s: %s", name, weights.ErrInt4Cols.Error())), weights.ErrInt4Cols)`;
  `len(t.Data) != rows*cols/2` → `WrongTensorSizeError(name, len(t.Data), rows*cols/2)`;
  `len(t.Scales) != rows*cols/32` → the existing scales-mismatch error; returns
  `matrix{rows, cols, q: t.Data, scales: t.Scales, q4: true}`.
- `mulVec`: first case `m.q4` → `nn.QuantizeBlocks32(...)` exactly as the int8 case, then
  `nn.MatVecQ4Block32(dst, qb.xq, qb.xs, m.q, m.scales, m.rows, m.cols)`. The int8 cases are
  unchanged (guard them with `!m.q4`).
- `row(dst, i)`: when `m.q4`, `weights.DequantInt4Block32(dst[:m.cols], m.q[i*m.cols/2:(i+1)*m.cols/2], m.scales[i*m.cols/32:(i+1)*m.cols/32])`.
  `rowsDot` already goes through `row`.

No other file changes: embeddings, the tied output projection and every layer use `matrix`.

## Stage 3 — tests (`tests/int4_test.go`, `package tests`, pattern of `tests/int8_test.go`)

Helper `int4Artifact(t, f32 *weights.Artifact) *weights.Artifact`: every `Float32` tensor with 2
dimensions whose second dimension is a multiple of 32 becomes `Int4Block32` (rows quantized one by
one with `weights.QuantizeInt4Block32`); every other tensor is copied as is; written with
`weights.WriteArtifact` and reopened with `weights.Open`. Source: `../testdata/tiny/model.wtypw`
(Qwen3.5 tiny) and `../testdata/tiny_lfm2/model.wtypw` (LFM2 tiny).

| Test | Proves |
|---|---|
| `TestInt4Model_LoadsWithoutConverting` | `decoder.New(tinyConfig(), int4Artifact(…), "model.language_model.")` succeeds; at least one tensor of the artifact is `Int4Block32` |
| `TestInt4Model_MatchesDequantizedWeights` | a Float32 artifact built by `DequantRow` of every `Int4Block32` tensor of the int4 artifact (like step 2 of `TestInt8Model`): the logits of both models, token by token over `reference.json`'s prompt, within 0.1 max abs (only the activation quantization differs, as measured for int8) |
| `TestInt4Model_LFM2` | the same two checks for the LFM2 tiny fixture: `tinyLFM2Config()` and prefix `"model."` (`tests/lfm2_test.go`) |
| `TestInt4Model_ZeroAllocStep` | `testing.AllocsPerRun` of `Step` with the int4 model → 0, as `TestInt8Model` step 4 |
| `TestInt4Model_LogsReferenceError` | logits vs `reference.json` (float model): **log** the max abs diff with `t.Logf` — no threshold (a tiny random model is not representative of 4-bit error; the real gate is decider-0.8b on the 36 questions, measured outside this plan) |
| `TestLoadMatrix_Int4WrongSize` | an artifact whose `embed_tokens.weight` is `Int4Block32` with one byte missing → `decoder.New` error naming the tensor |

## Stage 4 — docs

`README.md`: supported tensor types now include `Int4Block32` (with `nn.MatVecQ4Block32`).
`docs/` (if it lists types or kernels): the same.

## Acceptance

- `gotest` and `gotest -tinygo` green. Never run `gopush` or `codejob`.
- `grep -n "MatVecQ4Block32\|DequantInt4Block32" matrix.go` → both present.

| Stage | Files | Done when |
|---|---|---|
| 1 | `go.mod` | weights v0.4.0, nn v0.5.0 |
| 2 | `matrix.go` | int4 storage, kernel, row |
| 3 | `tests/int4_test.go` | table green |
| 4 | `README.md` | documented |
