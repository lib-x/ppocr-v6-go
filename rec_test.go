package ppocrv6

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
)

func skipIfNoRuntime(t *testing.T) {
	if os.Getenv("ONNXRUNTIME_LIB_PATH") == "" {
		t.Skip("ONNXRUNTIME_LIB_PATH not set")
	}
	path := filepath.Join("testdata", "inference.onnx")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s not found: %v", path, err)
	}
}

func TestRecognizeGolden(t *testing.T) {
	skipIfNoRuntime(t)

	f, err := os.Open(filepath.Join("testdata", "test3.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}

	rec, err := New(Config{
		ModelPath: filepath.Join("testdata", "inference.onnx"),
		DictPath:  filepath.Join("testdata", "ppocrv6_dict.txt"),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Close()

	res, err := rec.Recognize(img)
	if err != nil {
		t.Fatal(err)
	}
	const golden = "codex 还在跑，但诊断结果已经出来了——管线是通的："
	if res.Text != golden {
		t.Errorf("golden mismatch:\n got  %q\n want %q", res.Text, golden)
	}
	if res.Text == "" {
		t.Error("empty recognition result")
	}
	if res.Score <= 0 {
		t.Error("score must be positive")
	}
	t.Logf("recognized: %q (score %.4f)", res.Text, res.Score)
}

func TestRecognizeConsecutive(t *testing.T) {
	// Guards the environment refcount lifecycle: a Recognizer must survive
	// multiple consecutive calls without leaking or crashing.
	skipIfNoRuntime(t)
	rec, err := New(Config{
		ModelPath: filepath.Join("testdata", "inference.onnx"),
		DictPath:  filepath.Join("testdata", "ppocrv6_dict.txt"),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer rec.Close()

	f, _ := os.Open(filepath.Join("testdata", "test3.png"))
	defer f.Close()
	img, _, _ := image.Decode(f)

	for i := range 3 {
		_, err := rec.Recognize(img)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
	}
}
