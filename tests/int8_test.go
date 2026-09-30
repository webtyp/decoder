package tests

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"webtyp.com/decoder"
	"webtyp.com/weights"
)

func float32sToBytes(floats []float32) []byte {
	buf := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], math.Float32bits(f))
	}
	return buf
}

func TestInt8Model(t *testing.T) {
	int8Data, err := os.ReadFile("../testdata/tiny/model_int8b32.wtypw")
	if err != nil {
		t.Fatalf("failed to read int8 model file: %v", err)
	}

	artInt8, err := weights.Open(int8Data)
	if err != nil {
		t.Fatalf("failed to open int8 artifact: %v", err)
	}

	cfg := tinyConfig()
	prefix := "model.language_model."

	// 1. New loads int8 artifact successfully
	modelInt8, err := decoder.New(cfg, artInt8, prefix)
	if err != nil {
		t.Fatalf("failed to create int8 model: %v", err)
	}

	// 2. Kernel exactness: construct float32 artifact from int8 artifact by dequantizing rows
	inputs := make([]weights.TensorInput, len(artInt8.Tensors))
	for i, tensor := range artInt8.Tensors {
		if tensor.DType == weights.Int8Block32 {
			rows := tensor.Shape[0]
			cols := tensor.Shape[1]
			dequant := make([]float32, rows*cols)
			for r := 0; r < rows; r++ {
				if err := tensor.DequantRow(dequant[r*cols:(r+1)*cols], r); err != nil {
					t.Fatalf("DequantRow failed for tensor %s row %d: %v", tensor.Name, r, err)
				}
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

	f32Bytes, err := weights.WriteArtifact("tiny-f32", 1, weights.TokenizerConfig{}, inputs)
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

	ref := loadReference(t)

	// Compare outputs token-by-token between int8 model and dequantized float32 model
	stInt8 := modelInt8.NewState()
	stF32 := modelF32.NewState()

	logitsInt8 := make([]float32, cfg.Vocab)
	logitsF32 := make([]float32, cfg.Vocab)

	for i, tok := range ref.Prompt {
		if err := modelInt8.Step(stInt8, tok, logitsInt8); err != nil {
			t.Fatalf("int8 model step %d failed: %v", i, err)
		}
		if err := modelF32.Step(stF32, tok, logitsF32); err != nil {
			t.Fatalf("f32 model step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "kernel_exactness", logitsInt8, logitsF32, 1e-4)
	}

	// 3. Quantization error against reference.json
	st := modelInt8.NewState()
	logits := make([]float32, cfg.Vocab)

	var maxAbsDiff float64
	for i, tok := range ref.Prompt {
		if err := modelInt8.Step(st, tok, logits); err != nil {
			t.Fatalf("int8 model step %d failed: %v", i, err)
		}

		refLogits := ref.TokenByTokenLogits[i]
		for j, l := range logits {
			diff := math.Abs(float64(l - refLogits[j]))
			if diff > maxAbsDiff {
				maxAbsDiff = diff
			}
		}

		greedyGot := argmax(logits)
		greedyWant := argmax(refLogits)
		if greedyGot != greedyWant {
			t.Fatalf("step %d: greedy token mismatch: got %d, want %d", i, greedyGot, greedyWant)
		}
	}

	t.Logf("Measured max abs logit diff against reference: %g", maxAbsDiff)
	if maxAbsDiff >= 0.15 {
		t.Fatalf("max abs logit diff %g exceeds threshold 0.15", maxAbsDiff)
	}

	// 4. Zero allocations during Step
	stAlloc := modelInt8.NewState()
	logitsAlloc := make([]float32, cfg.Vocab)

	allocs := testing.AllocsPerRun(10, func() {
		_ = modelInt8.Step(stAlloc, 1, logitsAlloc)
	})

	if allocs > 0 {
		t.Fatalf("Step on int8 model allocated %g objects, want 0", allocs)
	}
}
