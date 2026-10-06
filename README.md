# decoder
<img src="docs/img/badges.svg">

The causal decoder of webtyp's in-browser language models, in Go for TinyGo/WASM. Token ids go
in, and the logits of the next token come out, one token at a time, with the state the model
keeps between tokens. It is the counterpart of `webtyp/encoder`: the graph, configurable per
model, with the arithmetic in `webtyp/nn`.

## I want X → use Y

| I want to... | Use... |
|---|---|
| Load a Qwen3.5 model | `decoder.New(Config{Arch: decoder.Qwen35, ...}, artifact, prefix)` |
| Load an LFM2 model | `decoder.New(Config{Arch: decoder.LFM2, ...}, artifact, prefix)` |
| Allocate sequence state for conversation | `model.NewState()` |
| Step the model with a token | `model.Step(state, token, logits)` |
| Read a prompt token without its prediction (34 % faster for Qwen3.5) | `model.Step(state, token, nil)` |
| Logits of a few tokens only (a decision's option letters) | `model.LogitsFor(state, ids, out)` |
| Resume from a saved state (e.g. after a fixed prompt prefix) | `state.CopyFrom(saved)` |
| Keep a state across sessions (e.g. in the browser's OPFS) | `data, err := state.MarshalBinary()`; later `fresh.UnmarshalBinary(data)` on a `NewState()` of the same model |

## Usage Example

```go
cfg := decoder.Config{
	Arch: decoder.Qwen35,
	// ...
}
art, _ := weights.Open(data)
model, err := decoder.New(cfg, art, "model.language_model.")
if err != nil {
    // handle error
}

st := model.NewState()
logits := make([]float32, model.Config.Vocab)

// Step model token by token
err = model.Step(st, tokenID, logits)
```

The artifact may be Float32, Int8Block32 (`weightsc -quant int8-block32`), or Int4Block32 (`weightsc -quant int4-block32`), and quantized matrices stay quantized in memory.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
