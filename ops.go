package decoder

import (
	"math"

	"webtyp.com/nn"
)

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func softplus(x float32) float32 {
	if x > 20.0 {
		return x
	}
	return float32(math.Log(1.0 + math.Exp(float64(x))))
}

func l2Normalize(x []float32) {
	var sumSq float64
	for _, v := range x {
		sumSq += float64(v) * float64(v)
	}
	rsqrt := float32(1.0 / math.Sqrt(sumSq+1e-6))
	for i := range x {
		x[i] *= rsqrt
	}
}

func ropeSlice(x []float32, pos int, rotaryDim int, theta float64) {
	half := rotaryDim / 2
	p := float64(pos)
	for d := 0; d < half; d++ {
		inv := math.Pow(theta, -float64(2*d)/float64(rotaryDim))
		th := p * inv
		cosTh := math.Cos(th)
		sinTh := math.Sin(th)

		a := float64(x[d])
		b := float64(x[d+half])

		x[d] = float32(a*cosTh - b*sinTh)
		x[d+half] = float32(b*cosTh + a*sinTh)
	}
}

// causalConvStep processes one step of depthwise causal 1d convolution.
// convState is channels * (convKernel - 1)
// input is channels
// weight is channels * convKernel (weight for channel c is weight[c*convKernel : (c+1)*convKernel])
// out receives channels values after SiLU activation.
func causalConvStep(convState []float32, input []float32, weight []float32, channels, convKernel int, out []float32) {
	kMinus1 := convKernel - 1
	for c := 0; c < channels; c++ {
		stateOffset := c * kMinus1
		weightOffset := c * convKernel

		var acc float64
		// Oldest in convState to newest
		for j := 0; j < kMinus1; j++ {
			acc += float64(convState[stateOffset+j]) * float64(weight[weightOffset+j])
		}
		// Current input
		inVal := input[c]
		acc += float64(inVal) * float64(weight[weightOffset+kMinus1])

		// Compute SiLU in-place on out[c]
		out[c] = float32(acc)
		nn.SiLU(out[c : c+1])

		// Shift state
		if kMinus1 > 1 {
			copy(convState[stateOffset:stateOffset+kMinus1-1], convState[stateOffset+1:stateOffset+kMinus1])
		}
		if kMinus1 > 0 {
			convState[stateOffset+kMinus1-1] = inVal
		}
	}
}

// deltaRuleStep updates the Kd x Vd state matrix S for one value head h and computes output o.
// S is Kd * Vd
// q, k are Kd
// v, mem, delta, o are Vd
func deltaRuleStep(S []float32, q, k, v []float32, aVal, bVal, aLogVal, dtBiasVal float32, Kd, Vd int, mem, delta, o []float32) {
	beta := sigmoid(bVal)
	sp := softplus(aVal + dtBiasVal)
	g := -float32(math.Exp(float64(aLogVal))) * sp
	expG := float32(math.Exp(float64(g)))

	// 1. Decay state S *= exp(g)
	for i := 0; i < len(S); i++ {
		S[i] *= expG
	}

	// 2. mem[v] = sum_i S[i][v] * k[i]
	for j := 0; j < Vd; j++ {
		var sum float64
		for i := 0; i < Kd; i++ {
			sum += float64(S[i*Vd+j]) * float64(k[i])
		}
		mem[j] = float32(sum)
	}

	// 3. delta[v] = (v[v] - mem[v]) * beta
	for j := 0; j < Vd; j++ {
		delta[j] = (v[j] - mem[j]) * beta
	}

	// 4. S[i][v] += k[i] * delta[v]
	for i := 0; i < Kd; i++ {
		ki := k[i]
		rowOffset := i * Vd
		for j := 0; j < Vd; j++ {
			S[rowOffset+j] += ki * delta[j]
		}
	}

	// 5. o[v] = sum_i S[i][v] * q[i]
	for j := 0; j < Vd; j++ {
		var sum float64
		for i := 0; i < Kd; i++ {
			sum += float64(S[i*Vd+j]) * float64(q[i])
		}
		o[j] = float32(sum)
	}
}
