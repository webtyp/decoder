package tests

import (
	"errors"
	"testing"

	"webtyp.com/decoder"
)

// Reading the prompt with nil logits and asking LogitsFor at the end gives the same logits as
// computing every logit at every step: prompt tokens only need the state, and a decision only a
// few tokens' logits.
func TestStep_NilLogitsThenLogitsForMatchesFullLogits(t *testing.T) {
	m := loadTinyModel(t)
	prompt := loadReference(t).Prompt

	full := m.NewState()
	logits := make([]float32, m.Config.Vocab)
	for _, tok := range prompt {
		if err := m.Step(full, tok, logits); err != nil {
			t.Fatal(err)
		}
	}

	lean := m.NewState()
	for _, tok := range prompt {
		if err := m.Step(lean, tok, nil); err != nil {
			t.Fatal(err)
		}
	}
	ids := []int{0, 7, 42, m.Config.Vocab - 1}
	got := make([]float32, len(ids))
	if err := m.LogitsFor(lean, ids, got); err != nil {
		t.Fatal(err)
	}
	for k, id := range ids {
		d := got[k] - logits[id]
		if d > 1e-4 || d < -1e-4 {
			t.Errorf("token %d: LogitsFor %v, full logits %v", id, got[k], logits[id])
		}
	}
}

func TestLogitsFor_RejectsBadIDs(t *testing.T) {
	m := loadTinyModel(t)
	st := m.NewState()
	if err := m.Step(st, 1, nil); err != nil {
		t.Fatal(err)
	}
	if err := m.LogitsFor(st, []int{m.Config.Vocab}, make([]float32, 1)); !errors.Is(err, decoder.ErrTokenOutOfBounds) {
		t.Fatalf("got %v, want ErrTokenOutOfBounds", err)
	}
	if err := m.LogitsFor(st, []int{1, 2}, make([]float32, 1)); !errors.Is(err, decoder.ErrLogitsLenMismatch) {
		t.Fatalf("got %v, want ErrLogitsLenMismatch", err)
	}
}
