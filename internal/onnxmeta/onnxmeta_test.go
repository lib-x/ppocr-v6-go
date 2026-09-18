package onnxmeta

import (
	"path/filepath"
	"testing"
)

func TestParseFixture(t *testing.T) {
	m, err := ParseFile(filepath.Join("..", "..", "testdata", "fixture.onnx"))
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}
	if len(m.Inputs) != 1 || len(m.Outputs) != 1 {
		t.Fatalf("expected 1 input and 1 output, got %d/%d", len(m.Inputs), len(m.Outputs))
	}
	in := m.Inputs[0]
	if in.Name != "x" {
		t.Errorf("input name = %q, want x", in.Name)
	}
	if in.ElemType != ElemTypeFloat32 {
		t.Errorf("input elem type = %d, want %d", in.ElemType, ElemTypeFloat32)
	}
	wantShape := []Dim{{Value: 1}, {Value: 3}, {Value: 48}, {Value: -1, Param: "W"}}
	if len(in.Shape) != len(wantShape) {
		t.Fatalf("input shape = %v, want %v", in.Shape, wantShape)
	}
	for i, d := range wantShape {
		if in.Shape[i] != d {
			t.Errorf("input dim[%d] = %+v, want %+v", i, in.Shape[i], d)
		}
	}
	out := m.Outputs[0]
	if out.Name != "fetch_name_0" {
		t.Errorf("output name = %q, want fetch_name_0", out.Name)
	}
	if len(out.Shape) != 3 || out.Shape[2].Value != 10 {
		t.Errorf("output shape = %v, want [1 T 10]", out.Shape)
	}
	if !out.Shape[1].IsDynamic() {
		t.Errorf("output dim[1] should be dynamic, got %+v", out.Shape[1])
	}
}

func TestParseRealModel(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "inference.onnx")
	m, err := ParseFile(path)
	if err != nil {
		t.Skipf("real model not available: %v", err)
	}
	in := m.Inputs[0]
	if in.Name != "x" || in.ElemType != ElemTypeFloat32 {
		t.Fatalf("unexpected input contract: %+v", in)
	}
	// [batch, 3, 48, width] with dynamic batch and width.
	if len(in.Shape) != 4 || in.Shape[1].Value != 3 || in.Shape[2].Value != 48 {
		t.Fatalf("unexpected input shape: %v", in.Shape)
	}
	if !in.Shape[0].IsDynamic() || !in.Shape[3].IsDynamic() {
		t.Errorf("expected dynamic batch and width dims, got %v", in.Shape)
	}
	out := m.Outputs[0]
	if len(out.Shape) != 3 || out.Shape[2].Value != 18710 {
		t.Fatalf("unexpected output shape: %v", out.Shape)
	}
}
