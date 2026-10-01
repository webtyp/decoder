package tests

import (
	"errors"
	"testing"

	"webtyp.com/decoder"
)

// A state saved after a prompt prefix and restored into a fresh state of the same model continues
// exactly like the original: this is how a runtime keeps the tool list read across sessions.
func TestState_SavedAndRestoredResumesTheSameSequence(t *testing.T) {
	for _, tc := range []struct {
		name  string
		model *decoder.Model
	}{
		{"qwen35", loadTinyModel(t)},
		{"lfm2", loadTinyLFM2Model(t, "model.wtypw")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model
			prompt := []int{11, 200, 7, 42, 99, 3}
			logits := make([]float32, m.Config.Vocab)
			original := m.NewState()
			for _, tok := range prompt[:len(prompt)-1] {
				if err := m.Step(original, tok, logits); err != nil {
					t.Fatal(err)
				}
			}
			saved, err := original.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			restored := m.NewState()
			if err := restored.UnmarshalBinary(saved); err != nil {
				t.Fatal(err)
			}
			if restored.Pos != original.Pos {
				t.Fatalf("Pos %d, want %d", restored.Pos, original.Pos)
			}
			last := prompt[len(prompt)-1]
			fromRestored := make([]float32, m.Config.Vocab)
			fromOriginal := make([]float32, m.Config.Vocab)
			if err := m.Step(restored, last, fromRestored); err != nil {
				t.Fatal(err)
			}
			if err := m.Step(original, last, fromOriginal); err != nil {
				t.Fatal(err)
			}
			checkLogitsMatch(t, "restored vs original", fromRestored, fromOriginal, 0)
		})
	}
}

// Bytes that are not a saved state, or are cut short, are refused and leave the state unchanged.
func TestState_UnmarshalRejectsBadBytes(t *testing.T) {
	m := loadTinyModel(t)
	st := m.NewState()
	if err := m.Step(st, 5, nil); err != nil {
		t.Fatal(err)
	}
	saved, err := st.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string][]byte{
		"empty":         nil,
		"not a state":   []byte("hello world, not a state"),
		"cut short":     saved[:len(saved)-3],
		"trailing data": append(append([]byte{}, saved...), 0),
	} {
		target := m.NewState()
		if err := target.UnmarshalBinary(data); !errors.Is(err, decoder.ErrStateFormat) {
			t.Errorf("%s: got %v, want ErrStateFormat", name, err)
		}
		if target.Pos != 0 {
			t.Errorf("%s: the state changed (Pos %d)", name, target.Pos)
		}
	}
}

// A state saved by one model is refused by a model of another shape.
func TestState_UnmarshalRejectsAnotherModelsState(t *testing.T) {
	qwen, lfm := loadTinyModel(t), loadTinyLFM2Model(t, "model.wtypw")
	saved, err := qwen.NewState().MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if err := lfm.NewState().UnmarshalBinary(saved); !errors.Is(err, decoder.ErrStateMismatch) {
		t.Fatalf("got %v, want ErrStateMismatch", err)
	}
}
