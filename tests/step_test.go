package tests

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"webtyp.com/decoder"
	"webtyp.com/weights"
)

type referenceData struct {
	Prompt              []int       `json:"prompt"`
	TokenByTokenLogits  [][]float32 `json:"token_by_token_logits"`
	PrefillLogits       [][]float32 `json:"prefill_logits"`
	GreedyTokens        []int       `json:"greedy_tokens"`
	GreedyStepLogits    [][]float32 `json:"greedy_step_logits"`
}

func tinyConfig() decoder.Config {
	return decoder.Config{
		Vocab:            256,
		Hidden:           64,
		Intermediate:     128,
		Layers:           []decoder.LayerKind{decoder.LinearAttention, decoder.LinearAttention, decoder.LinearAttention, decoder.FullAttention},
		Heads:            4,
		KVHeads:          2,
		HeadDim:          32,
		RotaryDim:        8,
		RopeTheta:        10000000,
		LinearKeyHeads:   4,
		LinearValueHeads: 4,
		LinearKeyDim:     16,
		LinearValueDim:   16,
		ConvKernel:       4,
		Eps:              1e-6,
	}
}

func loadTinyModel(t *testing.T) *decoder.Model {
	t.Helper()
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read model file: %v", err)
	}
	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}
	cfg := tinyConfig()
	model, err := decoder.New(cfg, art, "model.language_model.")
	if err != nil {
		t.Fatalf("failed to create model: %v", err)
	}
	return model
}

func loadReference(t *testing.T) referenceData {
	t.Helper()
	data, err := os.ReadFile("../testdata/tiny/reference.json")
	if err != nil {
		t.Fatalf("failed to read reference file: %v", err)
	}
	var ref referenceData
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatalf("failed to unmarshal reference data: %v", err)
	}
	return ref
}

func checkLogitsMatch(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", name, len(got), len(want))
	}
	var maxDiff float64
	for i := range got {
		diff := math.Abs(float64(got[i] - want[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff > tol {
		t.Fatalf("%s: max logit diff %g exceeds tolerance %g", name, maxDiff, tol)
	}
}

func argmax(logits []float32) int {
	bestIdx := 0
	bestVal := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > bestVal {
			bestVal = logits[i]
			bestIdx = i
		}
	}
	return bestIdx
}

func TestStep_MatchesReferenceTokenByToken(t *testing.T) {
	model := loadTinyModel(t)
	ref := loadReference(t)

	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	for i, tok := range ref.Prompt {
		if err := model.Step(st, tok, logits); err != nil {
			t.Fatalf("step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "token_by_token_logits", logits, ref.TokenByTokenLogits[i], 1e-4)
	}
}

func TestStep_MatchesPrefill(t *testing.T) {
	model := loadTinyModel(t)
	ref := loadReference(t)

	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	for i, tok := range ref.Prompt {
		if err := model.Step(st, tok, logits); err != nil {
			t.Fatalf("prefill step %d failed: %v", i, err)
		}
		checkLogitsMatch(t, "prefill_logits", logits, ref.PrefillLogits[i], 1e-4)
	}
}

func TestStep_GreedyContinuation(t *testing.T) {
	model := loadTinyModel(t)
	ref := loadReference(t)

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
		checkLogitsMatch(t, "greedy_step_logits", logits, ref.GreedyStepLogits[i], 1e-4)
	}
}

func TestStep_StatesAreIndependent(t *testing.T) {
	model := loadTinyModel(t)
	ref := loadReference(t)

	stA := model.NewState()
	stB := model.NewState()

	logitsA := make([]float32, model.Config.Vocab)
	logitsB := make([]float32, model.Config.Vocab)

	// Step stA with prompt[0] and stB with token 5
	_ = model.Step(stA, ref.Prompt[0], logitsA)
	_ = model.Step(stB, 5, logitsB)

	// Feed stA the rest of prompt
	for i := 1; i < len(ref.Prompt); i++ {
		_ = model.Step(stA, ref.Prompt[i], logitsA)
	}

	checkLogitsMatch(t, "independent_state_logits", logitsA, ref.TokenByTokenLogits[len(ref.Prompt)-1], 1e-4)
}

func TestStep_RejectsBadInput(t *testing.T) {
	model := loadTinyModel(t)
	st := model.NewState()

	logits := make([]float32, model.Config.Vocab)

	if err := model.Step(st, -1, logits); err == nil {
		t.Errorf("expected error for negative token, got nil")
	}
	if err := model.Step(st, model.Config.Vocab, logits); err == nil {
		t.Errorf("expected error for token >= Vocab, got nil")
	}
	if err := model.Step(st, 0, logits[:10]); err == nil {
		t.Errorf("expected error for logits size mismatch, got nil")
	}
}

func BenchmarkStep_Tiny(b *testing.B) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		b.Fatalf("failed to read model file: %v", err)
	}
	art, err := weights.Open(data)
	if err != nil {
		b.Fatalf("failed to open artifact: %v", err)
	}
	cfg := tinyConfig()
	model, err := decoder.New(cfg, art, "model.language_model.")
	if err != nil {
		b.Fatalf("failed to create model: %v", err)
	}

	st := model.NewState()
	logits := make([]float32, model.Config.Vocab)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = model.Step(st, i%model.Config.Vocab, logits)
	}
}
