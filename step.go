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
	embedRow := m.Embed[token*m.Config.Hidden : (token+1)*m.Config.Hidden]
	copy(scr.X, embedRow)

	// Process layers
	for l, layer := range m.Layers {
		stLayer := &st.Layers[l]

		// 1. Input RMSNorm
		_ = nn.RMSNorm(scr.H, scr.X, layer.InputLN, m.Config.Hidden, m.Config.Eps)

		// 2. Token mixer
		if layer.Kind == FullAttention {
			stepFullAttention(m.Config, layer.FullAttn, stLayer, scr, st.Pos, scr.H)
			for i := 0; i < m.Config.Hidden; i++ {
				scr.X[i] += scr.AttnProjOut[i]
			}
		} else {
			stepDeltaNet(m.Config, layer.LinearAttn, stLayer, scr, scr.H)
			for i := 0; i < m.Config.Hidden; i++ {
				scr.X[i] += scr.LinearProjO[i]
			}
		}

		// 3. Post-attention RMSNorm
		_ = nn.RMSNorm(scr.H, scr.X, layer.PostAttnLN, m.Config.Hidden, m.Config.Eps)

		// 4. MLP: down · ( SiLU(gate · h) ⊙ (up · h) )
		_ = nn.MatmulT(scr.MLPGate, scr.H, layer.GateProj, 1, m.Config.Hidden, m.Config.Intermediate)
		_ = nn.MatmulT(scr.MLPUp, scr.H, layer.UpProj, 1, m.Config.Hidden, m.Config.Intermediate)
		_ = nn.SiLU(scr.MLPGate)
		for i := 0; i < m.Config.Intermediate; i++ {
			scr.GateUp[i] = scr.MLPGate[i] * scr.MLPUp[i]
		}
		_ = nn.MatmulT(scr.H, scr.GateUp, layer.DownProj, 1, m.Config.Intermediate, m.Config.Hidden)

		for i := 0; i < m.Config.Hidden; i++ {
			scr.X[i] += scr.H[i]
		}
	}

	// Final RMSNorm
	_ = nn.RMSNorm(scr.X, scr.X, m.Norm, m.Config.Hidden, m.Config.Eps)

	// Output projection (tied with embed_tokens)
	_ = nn.MatmulT(logits, scr.X, m.Embed, 1, m.Config.Hidden, m.Config.Vocab)

	st.Pos++
	return nil
}
