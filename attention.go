package decoder

import (
	"math"

	"webtyp.com/nn"
)

// stepFullAttention computes the full attention block for position p.
func stepFullAttention(cfg Config, w *fullAttnWeights, st *LayerState, scr *Scratch, p int, h []float32) {
	// 1. qg = q_proj · h (Heads * 2 * HeadDim)
	w.QProj.mulVec(scr.QG, h)

	// 2. k = k_proj · h, v = v_proj · h
	kvDim := cfg.KVHeads * cfg.HeadDim
	w.KProj.mulVec(scr.K, h)
	w.VProj.mulVec(scr.V, h)

	// Split QG into per-head q and gate, apply RMSNorm & RoPE
	groupSize := cfg.Heads / cfg.KVHeads
	for i := 0; i < cfg.Heads; i++ {
		qHead := scr.QG[i*2*cfg.HeadDim : i*2*cfg.HeadDim+cfg.HeadDim]
		gHead := scr.QG[i*2*cfg.HeadDim+cfg.HeadDim : (i+1)*2*cfg.HeadDim]

		_ = nn.RMSNorm(qHead, qHead, w.QNorm, cfg.HeadDim, cfg.Eps)
		ropeSlice(qHead, p, cfg.RotaryDim, cfg.RopeTheta)

		// Copy gate to AttnGate
		copy(scr.AttnGate[i*cfg.HeadDim:(i+1)*cfg.HeadDim], gHead)
	}

	// Apply RMSNorm & RoPE to K heads
	for j := 0; j < cfg.KVHeads; j++ {
		kHead := scr.K[j*cfg.HeadDim : (j+1)*cfg.HeadDim]
		_ = nn.RMSNorm(kHead, kHead, w.KNorm, cfg.HeadDim, cfg.Eps)
		ropeSlice(kHead, p, cfg.RotaryDim, cfg.RopeTheta)
	}

	// 5. Append k, v to layer cache
	st.KCache = append(st.KCache, scr.K...)
	st.VCache = append(st.VCache, scr.V...)

	// Ensure AttnScores buffer is large enough for positions 0..p
	seqLen := p + 1
	if cap(scr.AttnScores) < seqLen {
		scr.AttnScores = make([]float32, seqLen)
	} else {
		scr.AttnScores = scr.AttnScores[:seqLen]
	}

	invSqrtHeadDim := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))

	// 6. Compute scores and weighted sum of V for each query head
	for i := 0; i < cfg.Heads; i++ {
		j := i / groupSize // corresponding KV head
		qHead := scr.QG[i*2*cfg.HeadDim : i*2*cfg.HeadDim+cfg.HeadDim]

		// Compute dot products with cached keys for head j across positions t=0..p
		for t := 0; t <= p; t++ {
			kCached := st.KCache[t*kvDim+j*cfg.HeadDim : t*kvDim+(j+1)*cfg.HeadDim]
			var dot float64
			for d := 0; d < cfg.HeadDim; d++ {
				dot += float64(qHead[d]) * float64(kCached[d])
			}
			scr.AttnScores[t] = float32(dot) * invSqrtHeadDim
		}

		_ = nn.Softmax(scr.AttnScores)

		// Weighted sum over VCache
		oHead := scr.AttnOut[i*cfg.HeadDim : (i+1)*cfg.HeadDim]
		for d := 0; d < cfg.HeadDim; d++ {
			var sum float64
			for t := 0; t <= p; t++ {
				vCached := st.VCache[t*kvDim+j*cfg.HeadDim+d]
				sum += float64(scr.AttnScores[t]) * float64(vCached)
			}
			oHead[d] = float32(sum)
		}
	}

	// 7. Gate application: o[c] *= sigmoid(gate[c])
	for c := 0; c < cfg.Heads*cfg.HeadDim; c++ {
		scr.AttnOut[c] *= sigmoid(scr.AttnGate[c])
	}

	// Out projection
	w.OProj.mulVec(scr.AttnProjOut, scr.AttnOut)
}
