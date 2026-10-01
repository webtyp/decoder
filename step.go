package decoder

import (
	"webtyp.com/nn"
)

// Step computes the next token logits for token given the current State.
func (m *Model) Step(st *State, token int, logits []float32) error {
	if token < 0 || token >= m.Config.Vocab {
		return ErrTokenOutOfBounds
	}
	if logits != nil && len(logits) != m.Config.Vocab {
		return ErrLogitsLenMismatch
	}

	scr := &st.Scratch

	// Embed token
	m.embed.row(scr.X, token)

	// Process layers
	for l, layer := range m.layers {
		stLayer := &st.Layers[l]

		// 1. Input RMSNorm
		_ = nn.RMSNorm(scr.H, scr.X, layer.InputLN, m.Config.Hidden, m.Config.Eps)

		// 2. Token mixer
		switch layer.Kind {
		case FullAttention:
			if m.Config.Arch == LFM2 {
				stepPlainAttention(m.Config, layer.plainAttn, stLayer, scr, st.Pos, scr.H)
			} else {
				stepFullAttention(m.Config, layer.FullAttn, stLayer, scr, st.Pos, scr.H)
			}
			for i := 0; i < m.Config.Hidden; i++ {
				scr.X[i] += scr.AttnProjOut[i]
			}
		case LinearAttention:
			stepDeltaNet(m.Config, layer.LinearAttn, stLayer, scr, scr.H)
			for i := 0; i < m.Config.Hidden; i++ {
				scr.X[i] += scr.LinearProjO[i]
			}
		case ShortConv:
			stepShortConv(m.Config, layer.shortConv, stLayer, scr, scr.H)
			for i := 0; i < m.Config.Hidden; i++ {
				scr.X[i] += scr.LinearProjO[i]
			}
		}

		// 3. Post-attention RMSNorm
		_ = nn.RMSNorm(scr.H, scr.X, layer.PostAttnLN, m.Config.Hidden, m.Config.Eps)

		// 4. MLP: down · ( SiLU(gate · h) ⊙ (up · h) )
		layer.GateProj.mulVec(scr.MLPGate, scr.H, &scr.Quant)
		layer.UpProj.mulVec(scr.MLPUp, scr.H, &scr.Quant)
		_ = nn.SiLU(scr.MLPGate)
		for i := 0; i < m.Config.Intermediate; i++ {
			scr.GateUp[i] = scr.MLPGate[i] * scr.MLPUp[i]
		}
		layer.DownProj.mulVec(scr.H, scr.GateUp, &scr.Quant)

		for i := 0; i < m.Config.Hidden; i++ {
			scr.X[i] += scr.H[i]
		}
	}

	// Final RMSNorm
	_ = nn.RMSNorm(scr.X, scr.X, m.norm, m.Config.Hidden, m.Config.Eps)

	// Output projection (tied with embed_tokens)
	// A prompt token's prediction is not needed: logits == nil skips the output projection, the
	// largest matrix of the model (34 % of a Qwen3.5-0.8B step).
	if logits != nil {
		m.embed.mulVec(logits, scr.X, &scr.Quant)
	}

	st.Pos++
	return nil
}

// LogitsFor writes into out[k] the logit of token ids[k] for the position Step last read, without
// computing the rest of the vocabulary. A decision needs a few option letters, not every token:
// read the prompt with Step(st, tok, nil), then call LogitsFor once.
func (m *Model) LogitsFor(st *State, ids []int, out []float32) error {
	if len(out) < len(ids) {
		return ErrLogitsLenMismatch
	}
	for _, id := range ids {
		if id < 0 || id >= m.Config.Vocab {
			return ErrTokenOutOfBounds
		}
	}
	m.embed.rowsDot(out, ids, st.Scratch.X, st.Scratch.RowTmp)
	return nil
}
