package decoder

func DecoderExportSigmoid(x float32) float32 {
	return sigmoid(x)
}

func DecoderExportSoftplus(x float32) float32 {
	return softplus(x)
}

func DecoderExportL2Normalize(x []float32) {
	l2Normalize(x)
}
