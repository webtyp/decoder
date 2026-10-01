package tests

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"

	"webtyp.com/decoder"
	"webtyp.com/weights"
)

func tinyLFM2Config() decoder.Config {
	return decoder.Config{
		Arch:         decoder.LFM2,
		Vocab:        256,
		Hidden:       64,
		Intermediate: 96,
		Layers: []decoder.LayerKind{
			decoder.ShortConv,
			decoder.ShortConv,
			decoder.FullAttention,
			decoder.ShortConv,
		},
		Heads:      4,
		KVHeads:    2,
		HeadDim:    16,
		RotaryDim:  16,
		RopeTheta:  1e6,
		ConvKernel: 3,
		Eps:        1e-5,
	}
}

func loadTinyLFM2Model(t *testing.T, filename string) *decoder.Model {
	t.Helper()
	data, err := os.ReadFile("../testdata/tiny_lfm2/" + filename)
	if err != nil {
		t.Fatalf("failed to read model file %s: %v", filename, err)
	}
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact %s: %v", filename, err)
	}
	cfg := tinyLFM2Config()
	model, err := decoder.New(cfg, art, "model.")
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	return model
}

func loadLFM2Reference(t *testing.T) referenceData {
	t.Helper()
	data, err := os.ReadFile("../testdata/tiny_lfm2/reference.json")
	if err != nil {
		t.Fatalf("failed to read reference file: %v", err)
	}
	var ref referenceData
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatalf("failed to unmarshal reference data: %v", err)
	}
	return ref
}

func TestLFM2_StepMatchesTokenByToken(t *testing.T) {
	model := loadTinyLFM2Model(t, "model.wtypw")
	ref := loadLFM2Reference(t)

	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	for i, tok := range ref.Prompt {
		if err := model.Step(st, tok, logits); err != nil {
			t.Fatalf("step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "lfm2 token_by_token_logits", logits, ref.TokenByTokenLogits[i], 1e-4)
	}
}

func TestLFM2_GreedyContinuation(t *testing.T) {
	model := loadTinyLFM2Model(t, "model.wtypw")
	ref := loadLFM2Reference(t)

	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	for _, tok := range ref.Prompt {
		if err := model.Step(st, tok, logits); err != nil {
			t.Fatalf("prompt step failed: %v", err)
		}
	}

	for i, expectedNextTok := range ref.GreedyTokens {
		nextTok := argmax(logits)
		if nextTok != expectedNextTok {
			t.Fatalf("greedy token %d: got %d want %d", i, nextTok, expectedNextTok)
		}

		if err := model.Step(st, nextTok, logits); err != nil {
			t.Fatalf("greedy step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "lfm2 greedy_step_logits", logits, ref.GreedyStepLogits[i], 1e-4)
	}
}

func TestLFM2_Int8Block32(t *testing.T) {
	model := loadTinyLFM2Model(t, "model_int8b32.wtypw")
	ref := loadLFM2Reference(t)

	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	var maxDiff float64
	for i, tok := range ref.Prompt {
		if err := model.Step(st, tok, logits); err != nil {
			t.Fatalf("step %d failed: %v", i, err)
		}

		gotNext := argmax(logits)
		wantNext := argmax(ref.PrefillLogits[i])
		if gotNext != wantNext {
			t.Fatalf("prompt position %d: greedy next token got %d want %d", i, gotNext, wantNext)
		}

		wantLogits := ref.PrefillLogits[i]
		for k := range logits {
			diff := math.Abs(float64(logits[k] - wantLogits[k]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}

	t.Logf("LFM2 Int8Block32 max |logit - prefill_logits| = %g", maxDiff)
	if maxDiff >= 0.1 {
		t.Fatalf("max logit diff %g exceeds limit 0.1", maxDiff)
	}
}

func TestLFM2_ZeroAllocationsInStep(t *testing.T) {
	model := loadTinyLFM2Model(t, "model.wtypw")
	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	// Warmup
	_ = model.Step(st, 1, logits)

	allocs := testing.AllocsPerRun(10, func() {
		_ = model.Step(st, 1, logits)
	})
	if allocs != 0 {
		t.Fatalf("LFM2 Step allocated %g times per run", allocs)
	}
}

func TestLFM2_CopyFromResumesIdentically(t *testing.T) {
	model := loadTinyLFM2Model(t, "model.wtypw")
	ref := loadLFM2Reference(t)
	prompt := ref.Prompt

	logits := make([]float32, model.Config.Vocab)
	original := model.NewState()
	for _, tok := range prompt[:len(prompt)-1] {
		if err := model.Step(original, tok, logits); err != nil {
			t.Fatal(err)
		}
	}

	resumed := model.NewState()
	if err := resumed.CopyFrom(original); err != nil {
		t.Fatal(err)
	}

	last := prompt[len(prompt)-1]
	fromCopy := make([]float32, model.Config.Vocab)
	if err := model.Step(resumed, last, fromCopy); err != nil {
		t.Fatal(err)
	}

	fromOriginal := make([]float32, model.Config.Vocab)
	if err := model.Step(original, last, fromOriginal); err != nil {
		t.Fatal(err)
	}

	checkLogitsMatch(t, "lfm2 copy vs original", fromCopy, fromOriginal, 0)
}

func TestLFM2_ConfigValidateErrors(t *testing.T) {
	cfg := tinyLFM2Config()
	cfg.Arch = 0
	if !errors.Is(cfg.Validate(), decoder.ErrInvalidArch) {
		t.Fatalf("got %v, want ErrInvalidArch", cfg.Validate())
	}

	cfg = tinyLFM2Config()
	cfg.Layers = append(cfg.Layers, decoder.LinearAttention)
	if !errors.Is(cfg.Validate(), decoder.ErrLayerKindForArch) {
		t.Fatalf("got %v, want ErrLayerKindForArch", cfg.Validate())
	}
}
