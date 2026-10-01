package decoder

// Arch is the model family: which tensors a checkpoint has and how a layer is computed.
type Arch uint8

const (
	Qwen35 Arch = iota + 1 // Qwen3.5: Gated DeltaNet + gated attention, (1+w) RMSNorm
	LFM2                   // LFM2: short convolutions + plain attention, w·x RMSNorm
)

// LayerKind is the token mixer of one layer.
type LayerKind uint8

const (
	LinearAttention LayerKind = iota // Gated DeltaNet: fixed-size recurrent state
	FullAttention                    // gated, causal, grouped-query attention: growing KV cache
	ShortConv                        // LFM2: gated depthwise causal convolution, fixed-size state
)

// Config is the shape of one decoder checkpoint.
type Config struct {
	Arch         Arch        // required: Qwen35 or LFM2
	Vocab        int         // vocabulary size (rows of the embedding table)
	Hidden       int         // model width
	Intermediate int         // MLP width
	Layers       []LayerKind // one entry per layer, in order

	Heads     int     // full attention: query heads
	KVHeads   int     // full attention: key/value heads (Heads % KVHeads == 0)
	HeadDim   int     // full attention: size of one head
	RotaryDim int     // full attention: leading dims of each head that RoPE rotates (HeadDim × partial_rotary_factor)
	RopeTheta float64 // full attention: RoPE base

	LinearKeyHeads   int // DeltaNet: key heads
	LinearValueHeads int // DeltaNet: value heads (multiple of LinearKeyHeads)
	LinearKeyDim     int // DeltaNet: size of one key head
	LinearValueDim   int // DeltaNet: size of one value head
	ConvKernel       int // DeltaNet: causal convolution width

	Eps float32 // RMSNorm epsilon
}

// Validate checks that all fields in Config are valid.
func (c Config) Validate() error {
	if c.Arch != Qwen35 && c.Arch != LFM2 {
		return ErrInvalidArch
	}
	if c.Vocab <= 0 {
		return ErrInvalidVocab
	}
	if c.Hidden <= 0 {
		return ErrInvalidHidden
	}
	if c.Intermediate <= 0 {
		return ErrInvalidIntermediate
	}
	if len(c.Layers) == 0 {
		return ErrEmptyLayers
	}
	for _, l := range c.Layers {
		if c.Arch == Qwen35 {
			if l != LinearAttention && l != FullAttention {
				return ErrLayerKindForArch
			}
		} else if c.Arch == LFM2 {
			if l != ShortConv && l != FullAttention {
				return ErrLayerKindForArch
			}
		}
	}
	if c.Heads <= 0 {
		return ErrInvalidHeads
	}
	if c.KVHeads <= 0 || c.Heads%c.KVHeads != 0 {
		return ErrInvalidKVHeads
	}
	if c.HeadDim <= 0 {
		return ErrInvalidHeadDim
	}
	if c.RotaryDim <= 0 || c.RotaryDim%2 != 0 || c.RotaryDim > c.HeadDim {
		return ErrInvalidRotaryDim
	}
	if c.RopeTheta <= 0 {
		return ErrInvalidRopeTheta
	}
	if c.Arch == Qwen35 {
		if c.LinearKeyHeads <= 0 {
			return ErrInvalidLinearKeyHeads
		}
		if c.LinearValueHeads <= 0 || c.LinearValueHeads%c.LinearKeyHeads != 0 {
			return ErrInvalidLinearValueHeads
		}
		if c.LinearKeyDim <= 0 {
			return ErrInvalidLinearKeyDim
		}
		if c.LinearValueDim <= 0 {
			return ErrInvalidLinearValueDim
		}
	}
	if c.ConvKernel <= 0 {
		return ErrInvalidConvKernel
	}
	if c.Eps <= 0 {
		return ErrInvalidEps
	}
	return nil
}
