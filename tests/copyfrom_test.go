package tests

import (
	"errors"
	"testing"

	"webtyp.com/decoder"
)

// A copied state continues exactly like the original, and the two share no memory: stepping
// the copy never changes what the original computes next.
func TestState_CopyFromResumesTheSameSequence(t *testing.T) {
	m := loadTinyModel(t)
	ref := loadReference(t)
	prompt := ref.Prompt
	if len(prompt) < 3 {
		t.Fatal("the fixture prompt needs at least 3 tokens")
	}
	logits := make([]float32, m.Config.Vocab)

	original := m.NewState()
	for _, tok := range prompt[:len(prompt)-1] {
		if err := m.Step(original, tok, logits); err != nil {
			t.Fatal(err)
		}
	}

	resumed := m.NewState()
	if err := resumed.CopyFrom(original); err != nil {
		t.Fatal(err)
	}
	if resumed.Pos != original.Pos {
		t.Fatalf("Pos %d, want %d", resumed.Pos, original.Pos)
	}

	last := prompt[len(prompt)-1]
	fromCopy := make([]float32, m.Config.Vocab)
	if err := m.Step(resumed, last, fromCopy); err != nil {
		t.Fatal(err)
	}
	// Step the copy further; the original must not notice.
	if err := m.Step(resumed, last, make([]float32, m.Config.Vocab)); err != nil {
		t.Fatal(err)
	}
	fromOriginal := make([]float32, m.Config.Vocab)
	if err := m.Step(original, last, fromOriginal); err != nil {
		t.Fatal(err)
	}
	checkLogitsMatch(t, "copy vs original", fromCopy, fromOriginal, 0)
}

func TestState_CopyFromRejectsAnotherModelsState(t *testing.T) {
	m := loadTinyModel(t)
	other := &decoder.State{}
	if err := m.NewState().CopyFrom(other); !errors.Is(err, decoder.ErrStateMismatch) {
		t.Fatalf("got %v, want ErrStateMismatch", err)
	}
}

func TestState_CopyFromDoesNotAllocateOnceSized(t *testing.T) {
	m := loadTinyModel(t)
	src := m.NewState()
	logits := make([]float32, m.Config.Vocab)
	for _, tok := range loadReference(t).Prompt {
		if err := m.Step(src, tok, logits); err != nil {
			t.Fatal(err)
		}
	}
	dst := m.NewState()
	_ = dst.CopyFrom(src)
	if allocs := testing.AllocsPerRun(10, func() { _ = dst.CopyFrom(src) }); allocs != 0 {
		t.Fatalf("CopyFrom allocated %v times per run", allocs)
	}
}
