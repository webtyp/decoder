package decoder

// stepShortConv computes one LFM2 short convolution layer step.
func stepShortConv(cfg Config, w *shortConvWeights, st *LayerState, scr *Scratch, h []float32) {
	H := cfg.Hidden
	K := cfg.ConvKernel
	kMinus1 := K - 1

	w.inProj.mulVec(scr.BCX, h, &scr.Quant)

	B := scr.BCX[:H]
	C := scr.BCX[H : 2*H]
	X := scr.BCX[2*H : 3*H]

	for c := 0; c < H; c++ {
		bx := B[c] * X[c]
		stateOffset := c * kMinus1
		convOffset := c * K

		var y float32
		for j := 0; j < kMinus1; j++ {
			y += st.ConvState[stateOffset+j] * w.conv[convOffset+j]
		}
		y += bx * w.conv[convOffset+kMinus1]

		if kMinus1 > 1 {
			copy(st.ConvState[stateOffset:stateOffset+kMinus1-1], st.ConvState[stateOffset+1:stateOffset+kMinus1])
		}
		if kMinus1 > 0 {
			st.ConvState[stateOffset+kMinus1-1] = bx
		}

		scr.ConvY[c] = C[c] * y
	}

	w.outProj.mulVec(scr.LinearProjO, scr.ConvY, &scr.Quant)
}
