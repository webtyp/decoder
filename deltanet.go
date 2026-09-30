package decoder

import (
	"math"

	"webtyp.com/nn"
)

// stepDeltaNet computes the Gated DeltaNet linear attention block.
func stepDeltaNet(cfg Config, w *linearAttnWeights, st *LayerState, scr *Scratch, h []float32) {
	Kh := cfg.LinearKeyHeads
	Kd := cfg.LinearKeyDim
	Vh := cfg.LinearValueHeads
	Vd := cfg.LinearValueDim
	qkvChannels := 2*Kh*Kd + Vh*Vd

	// 1. In projections
	w.InProjQKV.mulVec(scr.QKV, h)
	w.InProjZ.mulVec(scr.Z, h)
	w.InProjB.mulVec(scr.B, h)
	w.InProjA.mulVec(scr.A, h)

	// 2. Causal conv1d on QKV
	causalConvStep(st.ConvState, scr.QKV, w.Conv1D, qkvChannels, cfg.ConvKernel, scr.ConvOut)

	// 3. Split ConvOut into q, k, v
	qAll := scr.ConvOut[0 : Kh*Kd]
	kAll := scr.ConvOut[Kh*Kd : 2*Kh*Kd]
	vAll := scr.ConvOut[2*Kh*Kd : 2*Kh*Kd+Vh*Vd]

	// 6. L2 normalize q & k heads and scale q by 1/sqrt(Kd)
	invSqrtKd := float32(1.0 / math.Sqrt(float64(Kd)))
	for i := 0; i < Kh; i++ {
		qHead := qAll[i*Kd : (i+1)*Kd]
		kHead := kAll[i*Kd : (i+1)*Kd]
		l2Normalize(qHead)
		l2Normalize(kHead)
		for d := 0; d < Kd; d++ {
			qHead[d] *= invSqrtKd
		}
	}

	// 7. Gated delta rule per value head h
	vPerK := Vh / Kh
	for headIdx := 0; headIdx < Vh; headIdx++ {
		keyHeadIdx := headIdx / vPerK
		qHead := qAll[keyHeadIdx*Kd : (keyHeadIdx+1)*Kd]
		kHead := kAll[keyHeadIdx*Kd : (keyHeadIdx+1)*Kd]
		vHead := vAll[headIdx*Vd : (headIdx+1)*Vd]

		S := st.RecState[headIdx*Kd*Vd : (headIdx+1)*Kd*Vd]
		oHead := scr.LinearAttnO[headIdx*Vd : (headIdx+1)*Vd]

		deltaRuleStep(
			S, qHead, kHead, vHead,
			scr.A[headIdx], scr.B[headIdx], w.ALog[headIdx], w.DtBias[headIdx],
			Kd, Vd,
			scr.DeltaMem, scr.Delta, oHead,
		)

		// 8. Gated norm per head over Vd: o <- RMSNorm(o, w.Norm) * SiLU(z_h)
		_ = nn.RMSNorm(oHead, oHead, w.Norm, Vd, cfg.Eps)

		zHead := scr.Z[headIdx*Vd : (headIdx+1)*Vd]
		for d := 0; d < Vd; d++ {
			// Compute SiLU(z)
			zVal := zHead[d]
			_ = nn.SiLU(zHead[d : d+1])
			oHead[d] *= zHead[d]
			zHead[d] = zVal // restore if needed (though scratch Z is overwritten next step)
		}
	}

	// 9. Out projection
	w.OutProj.mulVec(scr.LinearProjO, scr.LinearAttnO)
}
