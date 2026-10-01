package decoder

import (
	"webtyp.com/fmt"
	"webtyp.com/weights"
)

// fullAttnWeights contains weights for a full attention layer.
type fullAttnWeights struct {
	QProj matrix    // [Heads*2*HeadDim][Hidden]
	KProj matrix    // [KVHeads*HeadDim][Hidden]
	VProj matrix    // [KVHeads*HeadDim][Hidden]
	OProj matrix    // [Hidden][Heads*HeadDim]
	QNorm []float32 // [HeadDim] zero-centered (1 + w)
	KNorm []float32 // [HeadDim] zero-centered (1 + w)
}

// linearAttnWeights contains weights for a Gated DeltaNet layer.
type linearAttnWeights struct {
	InProjQKV matrix    // [2*Kh*Kd + Vh*Vd][Hidden]
	InProjZ   matrix    // [Vh*Vd][Hidden]
	InProjB   matrix    // [Vh][Hidden]
	InProjA   matrix    // [Vh][Hidden]
	Conv1D    []float32 // [2*Kh*Kd + Vh*Vd][ConvKernel]
	ALog      []float32 // [Vh]
	DtBias    []float32 // [Vh]
	Norm      []float32 // [Vd] plain weight (NOT zero-centered)
	OutProj   matrix    // [Hidden][Vh*Vd]
}

// layerWeights contains weights for a single decoder layer.
type layerWeights struct {
	Kind       LayerKind
	InputLN    []float32 // [Hidden] zero-centered (1 + w)
	PostAttnLN []float32 // [Hidden] zero-centered (1 + w)
	GateProj   matrix    // [Intermediate][Hidden]
	UpProj     matrix    // [Intermediate][Hidden]
	DownProj   matrix    // [Hidden][Intermediate]

	FullAttn   *fullAttnWeights
	LinearAttn *linearAttnWeights
	shortConv  *shortConvWeights
	plainAttn  *plainAttnWeights
}

// Model represents an immutable loaded decoder model.
type Model struct {
	Config Config
	embed  matrix
	norm   []float32 // [Hidden] zero-centered (1 + w)
	layers []layerWeights
}

func getTensor(a *weights.Artifact, name string, expectedLen int) ([]float32, error) {
	t, ok := a.Tensor(name)
	if !ok {
		return nil, MissingTensorError(name)
	}
	if t.DType != weights.Float32 {
		return nil, UnsupportedDTypeError(name, string(t.DType))
	}
	data, err := t.Float32s()
	if err != nil {
		return nil, err
	}
	if len(data) != expectedLen {
		return nil, WrongTensorSizeError(name, len(data), expectedLen)
	}
	return data, nil
}

func makeZeroCentered(w []float32) []float32 {
	gamma := make([]float32, len(w))
	for i, v := range w {
		gamma[i] = 1.0 + v
	}
	return gamma
}

// New validates cfg and loads all model weights from the artifact.
func New(cfg Config, a *weights.Artifact, prefix string) (*Model, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	if cfg.Arch == LFM2 {
		return newLFM2(cfg, a, prefix)
	}

	embed, err := loadMatrix(a, prefix+"embed_tokens.weight", cfg.Vocab, cfg.Hidden)
	if err != nil {
		return nil, err
	}

	normRaw, err := getTensor(a, prefix+"norm.weight", cfg.Hidden)
	if err != nil {
		return nil, err
	}
	norm := makeZeroCentered(normRaw)

	layers := make([]layerWeights, len(cfg.Layers))

	Kh := cfg.LinearKeyHeads
	Kd := cfg.LinearKeyDim
	Vh := cfg.LinearValueHeads
	Vd := cfg.LinearValueDim
	qkvChannels := 2*Kh*Kd + Vh*Vd

	for i, kind := range cfg.Layers {
		idxStr := fmt.Sprintf("%d", i)
		lPrefix := prefix + "layers." + idxStr + "."

		inLNRaw, err := getTensor(a, lPrefix+"input_layernorm.weight", cfg.Hidden)
		if err != nil {
			return nil, err
		}

		postLNRaw, err := getTensor(a, lPrefix+"post_attention_layernorm.weight", cfg.Hidden)
		if err != nil {
			return nil, err
		}

		gate, err := loadMatrix(a, lPrefix+"mlp.gate_proj.weight", cfg.Intermediate, cfg.Hidden)
		if err != nil {
			return nil, err
		}

		up, err := loadMatrix(a, lPrefix+"mlp.up_proj.weight", cfg.Intermediate, cfg.Hidden)
		if err != nil {
			return nil, err
		}

		down, err := loadMatrix(a, lPrefix+"mlp.down_proj.weight", cfg.Hidden, cfg.Intermediate)
		if err != nil {
			return nil, err
		}

		lw := layerWeights{
			Kind:       kind,
			InputLN:    makeZeroCentered(inLNRaw),
			PostAttnLN: makeZeroCentered(postLNRaw),
			GateProj:   gate,
			UpProj:     up,
			DownProj:   down,
		}

		switch kind {
		case FullAttention:
			qProj, err := loadMatrix(a, lPrefix+"self_attn.q_proj.weight", cfg.Heads*2*cfg.HeadDim, cfg.Hidden)
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
			oProj, err := loadMatrix(a, lPrefix+"self_attn.o_proj.weight", cfg.Hidden, cfg.Heads*cfg.HeadDim)
			if err != nil {
				return nil, err
			}
			qNormRaw, err := getTensor(a, lPrefix+"self_attn.q_norm.weight", cfg.HeadDim)
			if err != nil {
				return nil, err
			}
			kNormRaw, err := getTensor(a, lPrefix+"self_attn.k_norm.weight", cfg.HeadDim)
			if err != nil {
				return nil, err
			}

			lw.FullAttn = &fullAttnWeights{
				QProj: qProj,
				KProj: kProj,
				VProj: vProj,
				OProj: oProj,
				QNorm: makeZeroCentered(qNormRaw),
				KNorm: makeZeroCentered(kNormRaw),
			}

		case LinearAttention:
			inQKV, err := loadMatrix(a, lPrefix+"linear_attn.in_proj_qkv.weight", qkvChannels, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			inZ, err := loadMatrix(a, lPrefix+"linear_attn.in_proj_z.weight", Vh*Vd, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			inB, err := loadMatrix(a, lPrefix+"linear_attn.in_proj_b.weight", Vh, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			inA, err := loadMatrix(a, lPrefix+"linear_attn.in_proj_a.weight", Vh, cfg.Hidden)
			if err != nil {
				return nil, err
			}
			conv, err := getTensor(a, lPrefix+"linear_attn.conv1d.weight", qkvChannels*cfg.ConvKernel)
			if err != nil {
				return nil, err
			}
			aLog, err := getTensor(a, lPrefix+"linear_attn.A_log", Vh)
			if err != nil {
				return nil, err
			}
			dtBias, err := getTensor(a, lPrefix+"linear_attn.dt_bias", Vh)
			if err != nil {
				return nil, err
			}
			norm, err := getTensor(a, lPrefix+"linear_attn.norm.weight", Vd)
			if err != nil {
				return nil, err
			}
			outProj, err := loadMatrix(a, lPrefix+"linear_attn.out_proj.weight", cfg.Hidden, Vh*Vd)
			if err != nil {
				return nil, err
			}

			lw.LinearAttn = &linearAttnWeights{
				InProjQKV: inQKV,
				InProjZ:   inZ,
				InProjB:   inB,
				InProjA:   inA,
				Conv1D:    conv,
				ALog:      aLog,
				DtBias:    dtBias,
				Norm:      norm, // plain weight
				OutProj:   outProj,
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
