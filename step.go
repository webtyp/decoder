package decoder

import (
	"webtyp.com/nn"
)

// Step computes the next token logits for token given the current State.
func (m *Model) Step(st *State, token int, logits []float32) error {
	if token < 0 || token >= m.Config.Vocab {
		return ErrTokenOutOfBounds
	}
	if len(logits) != m.Config.Vocab {
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
		layer.GateProj.mulVec(scr.MLPGate, scr.H)
		layer.UpProj.mulVec(scr.MLPUp, scr.H)
		_ = nn.SiLU(scr.MLPGate)
		for i := 0; i < m.Config.Intermediate; i++ {
			scr.GateUp[i] = scr.MLPGate[i] * scr.MLPUp[i]
		}
		layer.DownProj.mulVec(scr.H, scr.GateUp)

		for i := 0; i < m.Config.Hidden; i++ {
			scr.X[i] += scr.H[i]
		}
	}

	// Final RMSNorm
	_ = nn.RMSNorm(scr.X, scr.X, m.norm, m.Config.Hidden, m.Config.Eps)

	// Output projection (tied with embed_tokens)
	m.embed.mulVec(logits, scr.X)

	st.Pos++
	return nil
}
