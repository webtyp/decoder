# decoder

The causal decoder of webtyp's in-browser language models, in Go for TinyGo/WASM. Token ids go
in, and the logits of the next token come out, one token at a time, with the state the model
keeps between tokens. It is the counterpart of `webtyp/encoder`: the graph, configurable per
model, with the arithmetic in `webtyp/nn`.

## I want X → use Y

| I want to... | Use... |
|---|---|
| Load a decoder model | `decoder.New(cfg, artifact, prefix)` |
| Allocate sequence state for conversation | `model.NewState()` |
| Step the model with a token | `model.Step(state, token, logits)` |

## Usage Example

```go
cfg := decoder.Config{ ... }
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

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
