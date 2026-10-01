package decoder

// LayerState holds the mutable state for a single layer.
type LayerState struct {
	// FullAttention KV cache
	KCache []float32 // [pos][KVHeads * HeadDim] concatenated
	VCache []float32 // [pos][KVHeads * HeadDim] concatenated

	// DeltaNet states
	ConvState []float32 // [qkvChannels * (ConvKernel - 1)]
	RecState  []float32 // [LinearValueHeads * LinearKeyDim * LinearValueDim]
}

// Scratch holds pre-allocated working buffers for a State to avoid allocations in Step.
type Scratch struct {
	X       []float32 // [Hidden]
	H       []float32 // [Hidden]
	GateUp  []float32 // [Intermediate]
	MLPGate []float32 // [Intermediate]
	MLPUp   []float32 // [Intermediate]

	// FullAttention scratch
	QG          []float32 // [Heads * 2 * HeadDim]
	K           []float32 // [KVHeads * HeadDim]
	V           []float32 // [KVHeads * HeadDim]
	AttnOut     []float32 // [Heads * HeadDim]
	AttnGate    []float32 // [Heads * HeadDim]
	AttnProjOut []float32 // [Hidden]
	AttnScores  []float32 // [maxPos] dynamic

	// DeltaNet scratch
	QKV         []float32 // [qkvChannels]
	ConvOut     []float32 // [qkvChannels]
	Z           []float32 // [Vh * Vd]
	B           []float32 // [Vh]
	A           []float32 // [Vh]
	DeltaMem    []float32 // [Vd]
	Delta       []float32 // [Vd]
	DeltaO      []float32 // [Vd]
	LinearAttnO []float32 // [Vh * Vd]
	LinearProjO []float32 // [Hidden]

	// LFM2 scratch
	BCX   []float32 // [3 * Hidden]
	ConvY []float32 // [Hidden]

	// Quant holds the current input vector quantized to int8 blocks (matrix.mulVec), and
	// RowTmp one dequantized weight row (LogitsFor). Both are sized for the widest matrix.
	Quant  quantBuf
	RowTmp []float32
}

// State holds the mutable sequence state across steps.
type State struct {
	Pos     int
	Layers  []LayerState
	Scratch Scratch
}

// NewState creates a new State initialized for the given Model.
func (m *Model) NewState() *State {
	st := &State{
		Pos:     0,
		Layers:  make([]LayerState, len(m.Config.Layers)),
		Scratch: newScratch(m.Config),
	}

	Kh := m.Config.LinearKeyHeads
	Kd := m.Config.LinearKeyDim
	Vh := m.Config.LinearValueHeads
	Vd := m.Config.LinearValueDim
	qkvChannels := 2*Kh*Kd + Vh*Vd
	kMinus1 := m.Config.ConvKernel - 1

	for i, kind := range m.Config.Layers {
		ls := LayerState{}
		if kind == LinearAttention {
			ls.ConvState = make([]float32, qkvChannels*kMinus1)
			ls.RecState = make([]float32, Vh*Kd*Vd)
		} else if kind == ShortConv {
			ls.ConvState = make([]float32, m.Config.Hidden*kMinus1)
		}
		st.Layers[i] = ls
	}

	return st
}

func newScratch(cfg Config) Scratch {
	widest := cfg.Hidden
	for _, w := range []int{cfg.Intermediate, cfg.Heads * cfg.HeadDim, cfg.LinearValueHeads * cfg.LinearValueDim} {
		if w > widest {
			widest = w
		}
	}
	scr := Scratch{
		Quant:   quantBuf{xq: make([]int8, widest), xs: make([]float32, (widest+31)/32)},
		RowTmp:  make([]float32, cfg.Hidden),
		X:       make([]float32, cfg.Hidden),
		H:       make([]float32, cfg.Hidden),
		GateUp:  make([]float32, cfg.Intermediate),
		MLPGate: make([]float32, cfg.Intermediate),
		MLPUp:   make([]float32, cfg.Intermediate),

		QG:          make([]float32, cfg.Heads*2*cfg.HeadDim),
		K:           make([]float32, cfg.KVHeads*cfg.HeadDim),
		V:           make([]float32, cfg.KVHeads*cfg.HeadDim),
		AttnOut:     make([]float32, cfg.Heads*cfg.HeadDim),
		AttnGate:    make([]float32, cfg.Heads*cfg.HeadDim),
		AttnProjOut: make([]float32, cfg.Hidden),
		AttnScores:  make([]float32, 0, 1024),
	}

	if cfg.Arch == Qwen35 {
		Kh := cfg.LinearKeyHeads
		Kd := cfg.LinearKeyDim
		Vh := cfg.LinearValueHeads
		Vd := cfg.LinearValueDim
		qkvChannels := 2*Kh*Kd + Vh*Vd

		scr.QKV = make([]float32, qkvChannels)
		scr.ConvOut = make([]float32, qkvChannels)
		scr.Z = make([]float32, Vh*Vd)
		scr.B = make([]float32, Vh)
		scr.A = make([]float32, Vh)
		scr.DeltaMem = make([]float32, Vd)
		scr.Delta = make([]float32, Vd)
		scr.DeltaO = make([]float32, Vd)
		scr.LinearAttnO = make([]float32, Vh*Vd)
		scr.LinearProjO = make([]float32, cfg.Hidden)
	} else if cfg.Arch == LFM2 {
		scr.BCX = make([]float32, 3*cfg.Hidden)
		scr.ConvY = make([]float32, cfg.Hidden)
		scr.LinearProjO = make([]float32, cfg.Hidden)
	}

	return scr
}

// CopyFrom makes s the same sequence state as src: position, KV caches and recurrent states.
// The two states share no memory afterwards, and s reuses the capacity it already has. It is
// how a runtime keeps the state after a prompt's fixed prefix (identity and tools) and
// resumes from it on the next turn instead of reading that prefix again. Both states must
// come from the same Model.
func (s *State) CopyFrom(src *State) error {
	if len(s.Layers) != len(src.Layers) {
		return ErrStateMismatch
	}
	s.Pos = src.Pos
	for i := range src.Layers {
		d, o := &s.Layers[i], &src.Layers[i]
		if len(d.ConvState) != len(o.ConvState) || len(d.RecState) != len(o.RecState) {
			return ErrStateMismatch
		}
		d.KCache = append(d.KCache[:0], o.KCache...)
		d.VCache = append(d.VCache[:0], o.VCache...)
		copy(d.ConvState, o.ConvState)
		copy(d.RecState, o.RecState)
	}
	return nil
}
