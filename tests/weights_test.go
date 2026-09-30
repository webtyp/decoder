package tests

import (
	"os"
	"testing"

	"webtyp.com/decoder"
	"webtyp.com/weights"
)

func TestNew_MissingAndSizeError(t *testing.T) {
	data, err := os.ReadFile("../testdata/tiny/model.wtypw")
	if err != nil {
		t.Fatalf("failed to read test artifact: %v", err)
	}

	art, err := weights.Open(data)
	if err != nil {
		t.Fatalf("failed to open artifact: %v", err)
	}

	cfg := validConfig()
	// Model in testdata has 4 layers
	cfg.Layers = []decoder.LayerKind{
		decoder.LinearAttention,
		decoder.LinearAttention,
		decoder.LinearAttention,
		decoder.FullAttention,
	}

	_, err = decoder.New(cfg, art, "model.language_model.")
	if err != nil {
		t.Fatalf("expected successful model creation, got %v", err)
	}

	// Missing tensor check
	_, err = decoder.New(cfg, art, "nonexistent_prefix.")
	if err == nil {
		t.Fatalf("expected error for missing tensor, got nil")
	}

	// Wrong config/size check
	badCfg := cfg
	badCfg.Hidden = 128
	_, err = decoder.New(badCfg, art, "model.language_model.")
	if err == nil {
		t.Fatalf("expected error for wrong tensor size, got nil")
	}
}
