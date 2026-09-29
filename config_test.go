package decoder_test

import (
	"testing"

	"webtyp.com/decoder"
)

func validConfig() decoder.Config {
	return decoder.Config{
		Vocab:            256,
		Hidden:           64,
		Intermediate:     128,
		Layers:           []decoder.LayerKind{decoder.LinearAttention, decoder.FullAttention},
		Heads:            4,
		KVHeads:          2,
		HeadDim:          32,
		RotaryDim:        8,
		RopeTheta:        10000,
		LinearKeyHeads:   4,
		LinearValueHeads: 4,
		LinearKeyDim:     16,
		LinearValueDim:   16,
		ConvKernel:       4,
		Eps:              1e-6,
	}
}

func TestConfigValidate(t *testing.T) {
	cfg := validConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config, got %v", err)
	}

	invalidCases := []struct {
		name   string
		modify func(*decoder.Config)
	}{
		{"Vocab zero", func(c *decoder.Config) { c.Vocab = 0 }},
		{"Hidden zero", func(c *decoder.Config) { c.Hidden = 0 }},
		{"Intermediate zero", func(c *decoder.Config) { c.Intermediate = 0 }},
		{"Layers empty", func(c *decoder.Config) { c.Layers = nil }},
		{"Heads zero", func(c *decoder.Config) { c.Heads = 0 }},
		{"KVHeads not dividing Heads", func(c *decoder.Config) { c.KVHeads = 3 }},
		{"HeadDim zero", func(c *decoder.Config) { c.HeadDim = 0 }},
		{"RotaryDim odd", func(c *decoder.Config) { c.RotaryDim = 7 }},
		{"RotaryDim > HeadDim", func(c *decoder.Config) { c.RotaryDim = 64 }},
		{"RopeTheta zero", func(c *decoder.Config) { c.RopeTheta = 0 }},
		{"LinearKeyHeads zero", func(c *decoder.Config) { c.LinearKeyHeads = 0 }},
		{"LinearValueHeads not multiple", func(c *decoder.Config) { c.LinearKeyHeads = 3; c.LinearValueHeads = 4 }},
		{"LinearKeyDim zero", func(c *decoder.Config) { c.LinearKeyDim = 0 }},
		{"LinearValueDim zero", func(c *decoder.Config) { c.LinearValueDim = 0 }},
		{"ConvKernel zero", func(c *decoder.Config) { c.ConvKernel = 0 }},
		{"Eps zero", func(c *decoder.Config) { c.Eps = 0 }},
	}

	for _, tc := range invalidCases {
		t.Run(tc.name, func(t *testing.T) {
			c := validConfig()
			tc.modify(&c)
			if err := c.Validate(); err == nil {
				t.Fatalf("expected error for case %s, got nil", tc.name)
			}
		})
	}
}
