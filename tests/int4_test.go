package tests

import (
	"math"
	"os"
	"strings"
	"testing"

	"webtyp.com/decoder"
	"webtyp.com/weights"
)

func int4Artifact(t *testing.T, f32 *weights.Artifact) *weights.Artifact {
	inputs := make([]weights.TensorInput, len(f32.Tensors))
	for i, tensor := range f32.Tensors {
		if tensor.DType == weights.Float32 && len(tensor.Shape) == 2 && tensor.Shape[1]%32 == 0 {
			rows := tensor.Shape[0]
			cols := tensor.Shape[1]
			data, err := tensor.Float32s()
			if err != nil {
				t.Fatalf("failed to read float32 data for %s: %v", tensor.Name, err)
			}

			qData := make([]byte, 0, rows*cols/2)
			scales := make([]float32, 0, rows*cols/32)

			for r := 0; r < rows; r++ {
				row := data[r*cols : (r+1)*cols]
				q, s, err := weights.QuantizeInt4Block32(row)
				if err != nil {
					t.Fatalf("QuantizeInt4Block32 failed for %s row %d: %v", tensor.Name, r, err)
				}
				qData = append(qData, q...)
				scales = append(scales, s...)
			}

			inputs[i] = weights.TensorInput{
				Name:  tensor.Name,
				DType: weights.Int4Block32,
				Shape: tensor.Shape,
				Data:  qData,
				Scales: scales,
			}
		} else {
			inputs[i] = weights.TensorInput{
				Name:  tensor.Name,
				DType: tensor.DType,
				Shape: tensor.Shape,
				Data:  tensor.Data,
			}
		}
	}

	bytes, err := weights.WriteArtifact("tiny-int4", 1, weights.TokenizerConfig{}, inputs)
	if err != nil {
		t.Fatalf("failed to write int4 artifact: %v", err)
	}

	art, err := weights.Open(bytes)
	if err != nil {
		t.Fatalf("failed to open int4 artifact: %v", err)
	}

	return art
}

func TestInt4Model_LoadsWithoutConverting(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read model file: %v", err)
	}
	refArt, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}
	artInt4 := int4Artifact(t, refArt)

	hasInt4 := false
	for _, tensor := range artInt4.Tensors {
		if tensor.DType == weights.Int4Block32 {
			hasInt4 = true
			break
		}
	}
	if !hasInt4 {
		t.Fatalf("int4 artifact has no Int4Block32 tensors")
	}

	cfg := tinyConfig()
	_, err = decoder.New(cfg, artInt4, "model.language_model.")
	if err != nil {
		t.Fatalf("failed to load model: %v", err)
	}
}

func TestInt4Model_MatchesDequantizedWeights(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read model file: %v", err)
	}
	refArt, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}
	artInt4 := int4Artifact(t, refArt)

	// Create dequantized artifact
	inputs := make([]weights.TensorInput, len(artInt4.Tensors))
	for i, tensor := range artInt4.Tensors {
		if tensor.DType == weights.Int4Block32 {
			rows := tensor.Shape[0]
			cols := tensor.Shape[1]
			dequant := make([]float32, rows*cols)
			for r := 0; r < rows; r++ {
				weights.DequantInt4Block32(dequant[r*cols:(r+1)*cols], tensor.Data[r*cols/2:(r+1)*cols/2], tensor.Scales[r*cols/32:(r+1)*cols/32])
			}
			inputs[i] = weights.TensorInput{
				Name:  tensor.Name,
				DType: weights.Float32,
				Shape: tensor.Shape,
				Data:  float32sToBytes(dequant),
			}
		} else {
			inputs[i] = weights.TensorInput{
				Name:  tensor.Name,
				DType: tensor.DType,
				Shape: tensor.Shape,
				Data:  tensor.Data,
			}
		}
	}

	f32Bytes, err := weights.WriteArtifact("tiny-f32-from-int4", 1, weights.TokenizerConfig{}, inputs)
	if err != nil {
		t.Fatalf("failed to write dequantized artifact: %v", err)
	}

	artF32, err := weights.Open(f32Bytes)
	if err != nil {
		t.Fatalf("failed to open dequantized artifact: %v", err)
	}

	cfg := tinyConfig()
	prefix := "model.language_model."

	modelInt4, err := decoder.New(cfg, artInt4, prefix)
	if err != nil {
		t.Fatalf("failed to create int4 model: %v", err)
	}

	modelF32, err := decoder.New(cfg, artF32, prefix)
	if err != nil {
		t.Fatalf("failed to create dequantized f32 model: %v", err)
	}

	ref := loadReference(t)

	stInt4 := modelInt4.NewState()
	stF32 := modelF32.NewState()

	logitsInt4 := make([]float32, cfg.Vocab)
	logitsF32 := make([]float32, cfg.Vocab)

	for i, tok := range ref.Prompt {
		if err := modelInt4.Step(stInt4, tok, logitsInt4); err != nil {
			t.Fatalf("int4 model step %d failed: %v", i, err)
		}
		if err := modelF32.Step(stF32, tok, logitsF32); err != nil {
			t.Fatalf("f32 model step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "activation_quantization", logitsInt4, logitsF32, 0.1)
	}
}

func TestInt4Model_LFM2(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny_lfm2/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read LFM2 model file: %v", err)
	}
	refArt, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open LFM2 artifact: %v", err)
	}
	artInt4 := int4Artifact(t, refArt)

	cfg := tinyLFM2Config()
	prefix := "model."

	// Load successfully
	modelInt4, err := decoder.New(cfg, artInt4, prefix)
	if err != nil {
		t.Fatalf("failed to create int4 model: %v", err)
	}

	// Matches dequantized weights
	inputs := make([]weights.TensorInput, len(artInt4.Tensors))
	for i, tensor := range artInt4.Tensors {
		if tensor.DType == weights.Int4Block32 {
			rows := tensor.Shape[0]
			cols := tensor.Shape[1]
			dequant := make([]float32, rows*cols)
			for r := 0; r < rows; r++ {
				weights.DequantInt4Block32(dequant[r*cols:(r+1)*cols], tensor.Data[r*cols/2:(r+1)*cols/2], tensor.Scales[r*cols/32:(r+1)*cols/32])
			}
			inputs[i] = weights.TensorInput{
				Name:  tensor.Name,
				DType: weights.Float32,
				Shape: tensor.Shape,
				Data:  float32sToBytes(dequant),
			}
		} else {
			inputs[i] = weights.TensorInput{
				Name:  tensor.Name,
				DType: tensor.DType,
				Shape: tensor.Shape,
				Data:  tensor.Data,
			}
		}
	}

	f32Bytes, err := weights.WriteArtifact("tiny-lfm2-f32", 1, weights.TokenizerConfig{}, inputs)
	if err != nil {
		t.Fatalf("failed to write dequantized artifact: %v", err)
	}

	artF32, err := weights.Open(f32Bytes)
	if err != nil {
		t.Fatalf("failed to open dequantized artifact: %v", err)
	}

	modelF32, err := decoder.New(cfg, artF32, prefix)
	if err != nil {
		t.Fatalf("failed to create dequantized f32 model: %v", err)
	}

	ref := loadLFM2Reference(t)

	stInt4 := modelInt4.NewState()
	stF32 := modelF32.NewState()

	logitsInt4 := make([]float32, cfg.Vocab)
	logitsF32 := make([]float32, cfg.Vocab)

	for i, tok := range ref.Prompt {
		if err := modelInt4.Step(stInt4, tok, logitsInt4); err != nil {
			t.Fatalf("int4 model step %d failed: %v", i, err)
		}
		if err := modelF32.Step(stF32, tok, logitsF32); err != nil {
			t.Fatalf("f32 model step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "activation_quantization", logitsInt4, logitsF32, 0.1)
	}
}

func TestInt4Model_ZeroAllocStep(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read model file: %v", err)
	}
	refArt, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}
	artInt4 := int4Artifact(t, refArt)
	cfg := tinyConfig()
	modelInt4, err := decoder.New(cfg, artInt4, "model.language_model.")
	if err != nil {
		t.Fatalf("failed to create int4 model: %v", err)
	}

	st := modelInt4.NewState()
	logits := make([]float32, cfg.Vocab)

	allocs := testing.AllocsPerRun(10, func() {
		_ = modelInt4.Step(st, 1, logits)
	})

	if allocs > 0 {
		t.Fatalf("Step on int4 model allocated %g objects, want 0", allocs)
	}
}

func TestInt4Model_LogsReferenceError(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read model file: %v", err)
	}
	refArt, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}
	artInt4 := int4Artifact(t, refArt)
	cfg := tinyConfig()
	modelInt4, err := decoder.New(cfg, artInt4, "model.language_model.")
	if err != nil {
		t.Fatalf("failed to create int4 model: %v", err)
	}

	ref := loadReference(t)
	st := modelInt4.NewState()
	logits := make([]float32, cfg.Vocab)

	var maxAbsDiff float64
	for i, tok := range ref.Prompt {
		if err := modelInt4.Step(st, tok, logits); err != nil {
			t.Fatalf("int4 model step %d failed: %v", i, err)
		}
		refLogits := ref.TokenByTokenLogits[i]
		for j, l := range logits {
			diff := math.Abs(float64(l - refLogits[j]))
			if diff > maxAbsDiff {
				maxAbsDiff = diff
			}
		}
	}
	t.Logf("Measured max abs logit diff against reference: %g", maxAbsDiff)
}

func TestLoadMatrix_Int4WrongSize(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read model file: %v", err)
	}
	refArt, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}
	artInt4 := int4Artifact(t, refArt)

	// Corrupt a tensor in place
	for i, tensor := range artInt4.Tensors {
		if tensor.Name == "model.language_model.embed_tokens.weight" && tensor.DType == weights.Int4Block32 {
			artInt4.Tensors[i].Data = tensor.Data[:len(tensor.Data)-1] // One byte missing
			break
		}
	}

	cfg := tinyConfig()
	_, err = decoder.New(cfg, artInt4, "model.language_model.")
	if err == nil {
		t.Fatalf("expected error loading with bad tensor size, got nil")
	}
	if !strings.Contains(err.Error(), "embed_tokens.weight") {
		t.Fatalf("expected error to name the tensor, got: %v", err)
	}
}
