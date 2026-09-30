---
PLAN: "feat!: read Int8Block32 weights — matrices stay int8 in memory (the real Qwen3.5-0.8B artifact loads)"
TAG: v0.2.0
EXECUTOR: jules
REVIEWER: none
STATUS: running
SESSION: 8146648134492594019
---

> This plan is dispatched via the CodeJob workflow. See skill: agents-workflow.

# Plan — `webtyp.com/decoder`: int8 weights

## 0. Context

`decoder` runs the causal decoder of Qwen3.5 in Go for the browser (TinyGo/WASM). The input is
one token id, and the output is the logits of the next token (`Model.Step`). `decoder.New(cfg,
artifact, prefix)` loads the weights from a `webtyp.com/weights` artifact. Today it reads only
**Float32** tensors (`getTensor` in `weights.go` calls `Tensor.Float32s()`), and every weight
matrix is a `[]float32`.

The real model's artifact (Qwen3.5-0.8B, 851 MB, made by `weightsc -quant int8-block32`) stores
every 2-D tensor as **`weights.Int8Block32`**: one int8 per value, row-major, plus one float32
scale per block of 32 consecutive values of a row (value = q × scale). The other tensors
(norms, `A_log`, `dt_bias`, the 3-D conv) stay Float32. So `New` rejects that artifact today,
and `webtyp.com/qwen` cannot run the model.

Converting the int8 weights to float32 on load would take ~2 GB, more than the 4 GB clinic PCs
can give a browser tab. So **matrices stay int8 in memory** and are multiplied with
`nn.MatVecInt8Block32` (new in `webtyp.com/nn` v0.2.0):

```go
// MatVecInt8Block32 computes dst[r] = Σ_c W[r,c]·x[c] for a rows×cols matrix W stored as int8
// values in blocks: q holds rows×cols bytes, row-major, each byte an int8 (two's complement), and
// scales holds rows×ceil(cols/Int8BlockSize) entries, row by row.
func MatVecInt8Block32(dst, x []float32, q []byte, scales []float32, rows, cols int) error
```

A `weights.Tensor` has `DType`, `Shape`, `Data []byte` (for Int8Block32, exactly those int8
bytes) and `Scales []float32` (rows × ceil(cols/32)). `weights.BlockSize` is 32.

## Development rules (inline)

- Primary runtime: browser, TinyGo/WASM. `GOOS=js GOARCH=wasm go build ./...` must pass. Never
  import `fmt`, `errors`, `strings`, `strconv` (use `webtyp.com/fmt`), `encoding/json`, `sort`
  or `map[K]V` in non-test files. Tests may use the standard library.
- `Step` keeps **0 allocations**. The existing test that measures it stays green.
- Update dependencies first: `go get webtyp.com/nn@v0.2.0 webtyp.com/weights@latest`.
- No `TODO`, no commented-out code, no fallback. A tensor with a dtype the decoder cannot read
  is an error naming the tensor and its dtype.
- `gotest` green at the end.

## Design gate (api-design — five answers)

1. **Prior art.**
   - llama.cpp (`ggml`) keeps Q8_0 blocks in memory and multiplies them directly.
   - `candle` has `QMatMul` over quantized tensors.
   - PyTorch's `torch.ao` keeps int8 weights with per-channel scales.

   All three keep weights quantized and give the matmul a quantized path. None dequantizes the
   whole model on load.
2. **Novice-name test.** No new public name. The public change is that `New` accepts an
   int8 artifact.
3. **Complexity ledger.**

   ```
   Concepts the developer must learn   −3 (FullAttnWeights, LinearAttnWeights, LayerWeights become unexported)
   Files they must touch to do X       +0
   Lines at the call site              +0 (decoder.New and Step are unchanged)
   Ways to do the same thing           +0
   ```

4. **Where it belongs.** The kernel is a stateless operation, so it lives in `nn` (done). Which
   storage a tensor has is the decoder's loading concern.
5. **What it deletes.** The exported weight structs (`FullAttnWeights`, `LinearAttnWeights`,
   `LayerWeights`) and the exported `Model.Embed`, `Model.Norm` and `Model.Layers` fields.
   Nothing outside this repository reads them (checked in `webtyp/qwen`). `Model` keeps only its
   exported `Config`.

## Stage 1 — a matrix that is float32 or int8 (`matrix.go`)

```go
// matrix is a rows×cols weight matrix in the storage its tensor had: float32, or int8 in
// blocks of 32 with one scale per block. It is never converted on load.
type matrix struct {
	rows, cols int
	f32        []float32 // Float32 storage, rows×cols
	q          []byte    // Int8Block32 storage, rows×cols int8 values
	scales     []float32 // Int8Block32 scales, rows×ceil(cols/32)
}

// mulVec computes dst = W·x (dst has rows entries, x has cols).
func (m matrix) mulVec(dst, x []float32)

// row writes row i of W into dst as float32 (the embedding lookup of a token).
func (m matrix) row(dst []float32, i int)
```

- `mulVec` uses `nn.MatmulT(dst, x, m.f32, 1, m.cols, m.rows)` for float32, and
  `nn.MatVecInt8Block32(dst, x, m.q, m.scales, m.rows, m.cols)` for int8.
- `row` copies for float32, and for int8 computes `float32(int8(q)) * scale` in a loop over the
  row's blocks, without allocating.
- `loadMatrix(a *weights.Artifact, name string, rows, cols int) (matrix, error)`:
  - the tensor must exist → the current missing-tensor error;
  - it must have `rows*cols` values → the current size error;
  - `Float32` → `f32` from `Float32s()`;
  - `Int8Block32` → `q = t.Data`, `scales = t.Scales`, and it checks
    `len(t.Scales) == rows*ceil(cols/32)` or returns `weights.ErrScalesMismatch` wrapped with the
    tensor name;
  - any other dtype → `decoder: tensor %s has dtype %s, which the decoder cannot read`
    (`ErrUnsupportedDType`, following how `errors.go` builds its other errors).

## Stage 2 — the weights use `matrix` (`weights.go`, `step.go`, `attention.go`, `deltanet.go`)

- Rename the weight structs to unexported ones (`fullAttnWeights`, `linearAttnWeights`,
  `layerWeights`), and the `Model` fields to `embed`, `norm` and `layers`. Every 2-D projection
  field (`QProj`, `KProj`, `VProj`, `OProj`, `InProjQKV`, `InProjZ`, `InProjB`, `InProjA`,
  `OutProj`, `GateProj`, `UpProj`, `DownProj`) and the embedding become `matrix`, loaded with
  `loadMatrix` and the shapes that `New` already checks. The 1-D and conv tensors stay
  `[]float32` from `getTensor`, and `getTensor` rejects a non-Float32 tensor with
  `ErrUnsupportedDType`.
- Every `nn.MatmulT(out, in, W, 1, k, n)` in `step.go`, `attention.go` and `deltanet.go` becomes
  `W.mulVec(out, in)`.
- Embedding: `m.embed.row(scr.X, token)`. Logits (tied embeddings): `m.embed.mulVec(logits, scr.X)`.

## Stage 3 — tests (`tests/`)

The ecosystem rule, and the owner's standing request: tests live in `tests/`, `package tests`,
exported API only.

- Move `step_test.go`, `weights_test.go`, `config_test.go` and any other `package decoder_test`
  file to `tests/` as `package tests`. Fixture paths become `../testdata/...`.
- `ops_test.go` and `export_test.go` test unexported math helpers. Keep those two files in the
  root, as `package decoder`, and add one line to `AGENTS.md`:
  `ops_test.go stays in the root: it tests unexported math helpers that no exported API exposes one by one.`
- New `tests/int8_test.go`, with the fixture `testdata/tiny/model_int8b32.wtypw`. It is already
  in the repository, made by `testdata/tiny_to_int8b32.go.txt` from the same tiny model as
  `model.wtypw`: 32 two-dimensional tensors in Int8Block32, the rest in Float32.
  1. `New` loads it with the tiny config (the same config the float32 tests use).
  2. **Kernel exactness.** Build a float32 artifact in the test from the int8 one:
     `weights.WriteArtifact` with every tensor converted to Float32 through
     `Tensor.DequantRow`. Then `weights.Open` it, feed the token sequence of
     `testdata/tiny/reference.json` to both models, and require every logit within `1e-4` of the
     other. This proves that the int8 path computes exactly what the dequantized weights
     compute.
  3. **Quantization error.** The int8 model's logits against `reference.json` (the float32
     PyTorch reference): the greedy next token equals the reference's at every position, and
     the maximum absolute logit difference is below `0.15`. Measured beforehand in PyTorch with the
     same block quantization: 0.0998 on logits ranging from −2.2 to 2.1, greedy tokens equal. Print
     the measured value with `t.Logf`.
  4. `testing.AllocsPerRun` of `Step` on the int8 model is 0.
- Existing float32 tests stay green, unchanged apart from the move.

## Stage 4 — docs

- `README.md`: one line under the usage example: the artifact may be Float32 or Int8Block32
  (`weightsc -quant int8-block32`), and int8 matrices stay int8 in memory.
- `docs/ARCHITECTURE.md`: a short section "Weights in memory" with the reason from §0 (2 GB vs
  4 GB machines).

## Stages

| Stage | Files | Acceptance |
|---|---|---|
| 1 | `matrix.go`, `errors.go` | builds |
| 2 | `weights.go`, `step.go`, `attention.go`, `deltanet.go` | `grep -rn "nn.MatmulT" step.go attention.go deltanet.go` → empty |
| 3 | `tests/`, `AGENTS.md` | `tests/int8_test.go` passes; only `ops_test.go`/`export_test.go` remain in the root |
| 4 | `README.md`, `docs/ARCHITECTURE.md` | both mention Int8Block32 |
| all | — | `gotest` green; `GOOS=js GOARCH=wasm go build ./...` |
