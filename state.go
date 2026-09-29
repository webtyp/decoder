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
	X        []float32 // [Hidden]
	H        []float32 // [Hidden]
	GateUp   []float32 // [Intermediate]
	MLPGate  []float32 // [Intermediate]
	MLPUp    []float32 // [Intermediate]

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
		}
		st.Layers[i] = ls
	}

	return st
}

func newScratch(cfg Config) Scratch {
	Kh := cfg.LinearKeyHeads
	Kd := cfg.LinearKeyDim
	Vh := cfg.LinearValueHeads
	Vd := cfg.LinearValueDim
	qkvChannels := 2*Kh*Kd + Vh*Vd

	return Scratch{
		X:           make([]float32, cfg.Hidden),
		H:           make([]float32, cfg.Hidden),
		GateUp:      make([]float32, cfg.Intermediate),
		MLPGate:     make([]float32, cfg.Intermediate),
		MLPUp:       make([]float32, cfg.Intermediate),

		QG:          make([]float32, cfg.Heads*2*cfg.HeadDim),
		K:           make([]float32, cfg.KVHeads*cfg.HeadDim),
		V:           make([]float32, cfg.KVHeads*cfg.HeadDim),
		AttnOut:     make([]float32, cfg.Heads*cfg.HeadDim),
		AttnGate:    make([]float32, cfg.Heads*cfg.HeadDim),
		AttnProjOut: make([]float32, cfg.Hidden),
		AttnScores:  make([]float32, 0, 1024),

		QKV:         make([]float32, qkvChannels),
		ConvOut:     make([]float32, qkvChannels),
		Z:           make([]float32, Vh*Vd),
		B:           make([]float32, Vh),
		A:           make([]float32, Vh),
		DeltaMem:    make([]float32, Vd),
		Delta:       make([]float32, Vd),
		DeltaO:      make([]float32, Vd),
		LinearAttnO: make([]float32, Vh*Vd),
		LinearProjO: make([]float32, cfg.Hidden),
	}
}
