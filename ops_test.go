package decoder_test

import (
	"math"
	"testing"

	"webtyp.com/decoder"
)

func TestOpsHelpers(t *testing.T) {
	// Sigmoid
	s0 := decoder.DecoderExportSigmoid(0)
	if math.Abs(float64(s0-0.5)) > 1e-6 {
		t.Errorf("sigmoid(0) = %f, want 0.5", s0)
	}

	// Softplus
	sp0 := decoder.DecoderExportSoftplus(0)
	expectedSp0 := float32(math.Log(2.0))
	if math.Abs(float64(sp0-expectedSp0)) > 1e-6 {
		t.Errorf("softplus(0) = %f, want %f", sp0, expectedSp0)
	}
	spLarge := decoder.DecoderExportSoftplus(25.0)
	if spLarge != 25.0 {
		t.Errorf("softplus(25) = %f, want 25.0", spLarge)
	}

	// L2Normalize
	x := []float32{3.0, 4.0}
	decoder.DecoderExportL2Normalize(x)
	// norm of [3, 4] is sqrt(25 + 1e-6) ~ 5.0
	if math.Abs(float64(x[0]-0.6)) > 1e-3 || math.Abs(float64(x[1]-0.8)) > 1e-3 {
		t.Errorf("l2Normalize([3,4]) = %v, want [~0.6, ~0.8]", x)
	}
}
