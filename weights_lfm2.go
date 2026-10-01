package decoder

import (
	"webtyp.com/fmt"
	"webtyp.com/weights"
)

type shortConvWeights struct {
	inProj  matrix    // [3*Hidden][Hidden]
	conv    []float32 // [Hidden*ConvKernel]: channel c uses conv[c*K : c*K+K], index 0 the oldest input
	outProj matrix    // [Hidden][Hidden]
}

type plainAttnWeights struct {
	q, k, v, o   matrix    // q [Heads*HeadDim][Hidden], k,v [KVHeads*HeadDim][Hidden], o [Hidden][Heads*HeadDim]
	qNorm, kNorm []float32 // [HeadDim], used as is
}

func newLFM2(cfg Config, a *weights.Artifact, prefix string) (*Model, error) {
	embed, err := loadMatrix(a, prefix+"embed_tokens.weight", cfg.Vocab, cfg.Hidden)
	if err != nil {
		return nil, err
	}

	norm, err := getTensor(a, prefix+"embedding_norm.weight", cfg.Hidden)
	if err != nil {
		return nil, err
	}

	layers := make([]layerWeights, len(cfg.Layers))

	for i, kind := range cfg.Layers {
		idxStr := fmt.Sprintf("%d", i)
		lPrefix := prefix + "layers." + idxStr + "."

		opNorm, err := getTensor(a, lPrefix+"operator_norm.weight", cfg.Hidden)
		if err != nil {
			return nil, err
		}

		ffnNorm, err := getTensor(a, lPrefix+"ffn_norm.weight", cfg.Hidden)
		if err != nil {
			return nil, err
		}

		gate, err := loadMatrix(a, lPrefix+"feed_forward.w1.weight", cfg.Intermediate, cfg.Hidden)
		if err != nil {
			return nil, err
		}

		up, err := loadMatrix(a, lPrefix+"feed_forward.w3.weight", cfg.Intermediate, cfg.Hidden)
		if err != nil {
			return nil, err
		}

		down, err := loadMatrix(a, lPrefix+"feed_forward.w2.weight", cfg.Hidden, cfg.Intermediate)
		if err != nil {
			return nil, err
		}

		lw := layerWeights{
			Kind:       kind,
			InputLN:    opNorm,
			PostAttnLN: ffnNorm,
			GateProj:   gate,
			UpProj:     up,
			DownProj:   down,
		}

		switch kind {
		case ShortConv:
			inProj, err := loadMatrix(a, lPrefix+"conv.in_proj.weight", 3*cfg.Hidden, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			conv, err := getTensor(a, lPrefix+"conv.conv.weight", cfg.Hidden*cfg.ConvKernel)
			if err != nil {
				return nil, err
			}
			outProj, err := loadMatrix(a, lPrefix+"conv.out_proj.weight", cfg.Hidden, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			lw.shortConv = &shortConvWeights{
				inProj:  inProj,
				conv:    conv,
				outProj: outProj,
			}

		case FullAttention:
			qProj, err := loadMatrix(a, lPrefix+"self_attn.q_proj.weight", cfg.Heads*cfg.HeadDim, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			kProj, err := loadMatrix(a, lPrefix+"self_attn.k_proj.weight", cfg.KVHeads*cfg.HeadDim, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			vProj, err := loadMatrix(a, lPrefix+"self_attn.v_proj.weight", cfg.KVHeads*cfg.HeadDim, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			oProj, err := loadMatrix(a, lPrefix+"self_attn.out_proj.weight", cfg.Hidden, cfg.Heads*cfg.HeadDim)
			if err != nil {
				return nil, err
			}
			qNorm, err := getTensor(a, lPrefix+"self_attn.q_layernorm.weight", cfg.HeadDim)
			if err != nil {
				return nil, err
			}
			kNorm, err := getTensor(a, lPrefix+"self_attn.k_layernorm.weight", cfg.HeadDim)
			if err != nil {
				return nil, err
			}

			lw.plainAttn = &plainAttnWeights{
				q:     qProj,
				k:     kProj,
				v:     vProj,
				o:     oProj,
				qNorm: qNorm,
				kNorm: kNorm,
			}
		}

		layers[i] = lw
	}

	return &Model{
		Config: cfg,
		embed:  embed,
		norm:   norm,
		layers: layers,
	}, nil
}
