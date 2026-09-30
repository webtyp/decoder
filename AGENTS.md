# Agent Guide — `webtyp/decoder`

Constraints for agents working on this library. **Read this before any change.**
The current work order, when one exists, is [docs/PLAN.md](docs/PLAN.md). The ecosystem plan
is [`agent/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md`](https://github.com/webtyp/agent/blob/main/docs/AGENT_ECOSYSTEM_MASTER_PLAN.md).

## What this library is

The causal decoder graph (Gated DeltaNet + gated full attention) configurable per checkpoint,
with an immutable `Model` and one mutable `State` per sequence. It never contains a tokenizer, a
chat template, a sampler policy or model-specific configuration. Those belong to the model
adapter (`webtyp/qwen`).

**Correctness is defined by `testdata/tiny/reference.json`**, logits produced by the reference
implementation (Hugging Face `transformers`) for a tiny model of the same architecture. Never
loosen a tolerance to make a test pass. Find the step that diverges.

ops_test.go stays in the root: it tests unexported math helpers that no exported API exposes one by one.

## The builds that define "done"

```bash
go vet ./...
gotest
gotest -tinygo
GOOS=js GOARCH=wasm go build ./...
```

## Never import these (non-test files)

| Never | Use instead | Why |
|---|---|---|
| `fmt`, `errors`, `strings`, `strconv` | `webtyp.com/fmt` | isomorphic, small under TinyGo |
| `context` (stdlib) | `webtyp.com/context` | |
| `sort`, `encoding/json`, `map[K]V` | slices | size tax under TinyGo |
| `os`, `log`, `net/http` | nothing | weights arrive as a `*weights.Artifact` |

`math` is allowed. Tests may use `os` and `encoding/json` to read `testdata`.

## Common mistakes to avoid

- Re-implementing an operation `webtyp/nn` already has (matmul, RMSNorm, SiLU, softmax).
- Forgetting that Qwen3.5's RMSNorm multiplies by `(1 + w)` (zero-centered), except the
  DeltaNet gated norm, which multiplies by `w`.
- Rotating the whole head with RoPE. Only the first `RotaryDim` values rotate.
- Allocating inside `Step`. Buffers belong to `Model`/`State`.
- Speed work (SIMD, int8, block prefill) in a correctness plan. See `webtyp/nn/docs/SIMD.md`.
