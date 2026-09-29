# decoder

The causal decoder of webtyp's in-browser language models, in Go for TinyGo/WASM. Token ids go
in, and the logits of the next token come out, one token at a time, with the state the model
keeps between tokens. It is the counterpart of `webtyp/encoder`: the graph, configurable per
model, with the arithmetic in `webtyp/nn`.

> **STATUS (remove this note when the first plan lands):** documentation only. See
> [Architecture](docs/ARCHITECTURE.md) for what it will contain.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
