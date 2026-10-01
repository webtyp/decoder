package decoder

import (
	"math"

	"webtyp.com/nn"
)

// stepPlainAttention computes the plain attention block for position p in LFM2.
func stepPlainAttention(cfg Config, w *plainAttnWeights, st *LayerState, scr *Scratch, p int, h []float32) {
	// 1. q = w.q · h (Heads * HeadDim)
	qLen := cfg.Heads * cfg.HeadDim
	q := scr.QG[:qLen]
	w.q.mulVec(q, h)

	// 2. k = w.k · h, v = w.v · h
	kvDim := cfg.KVHeads * cfg.HeadDim
	w.k.mulVec(scr.K, h)
	w.v.mulVec(scr.V, h)

	// Apply RMSNorm & RoPE to Q heads
	groupSize := cfg.Heads / cfg.KVHeads
	for i := 0; i < cfg.Heads; i++ {
		qHead := q[i*cfg.HeadDim : (i+1)*cfg.HeadDim]
		_ = nn.RMSNorm(qHead, qHead, w.qNorm, cfg.HeadDim, cfg.Eps)
		ropeSlice(qHead, p, cfg.RotaryDim, cfg.RopeTheta)
	}

	// Apply RMSNorm & RoPE to K heads
	for j := 0; j < cfg.KVHeads; j++ {
		kHead := scr.K[j*cfg.HeadDim : (j+1)*cfg.HeadDim]
		_ = nn.RMSNorm(kHead, kHead, w.kNorm, cfg.HeadDim, cfg.Eps)
		ropeSlice(kHead, p, cfg.RotaryDim, cfg.RopeTheta)
	}

	// 3. Append k, v to layer cache
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

	// 4. Compute scores and weighted sum of V for each query head
	for i := 0; i < cfg.Heads; i++ {
		j := i / groupSize // corresponding KV head
		qHead := q[i*cfg.HeadDim : (i+1)*cfg.HeadDim]

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

	// Out projection
	w.o.mulVec(scr.AttnProjOut, scr.AttnOut)
}
