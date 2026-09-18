package ppocrv6

import (
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func skipPipelineIfNoRuntime(t *testing.T) {
	if os.Getenv("ONNXRUNTIME_LIB_PATH") == "" {
		t.Skip("ONNXRUNTIME_LIB_PATH not set")
	}
	for _, f := range []string{"inference.onnx", "det-v6.onnx", "menu.png"} {
		if _, err := os.Stat(filepath.Join("testdata", f)); err != nil {
			t.Skipf("testdata/%s not found: %v", f, err)
		}
	}
}

// TestPipelineGolden runs the full detection + recognition pipeline on a
// menu image and checks the recognized lines.
func TestPipelineGolden(t *testing.T) {
	skipPipelineIfNoRuntime(t)

	f, err := os.Open(filepath.Join("testdata", "menu.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Fatal(err)
	}

	p, err := NewPipeline(
		Config{ModelPath: filepath.Join("testdata", "inference.onnx")},
		DetConfig{ModelPath: filepath.Join("testdata", "det-v6.onnx")},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	results, err := p.Recognize(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) < 30 {
		t.Errorf("detected %d lines, want at least 30", len(results))
	}

	// Every line must decode to non-empty text with a reasonable score.
	for i, r := range results {
		if r.Text == "" {
			t.Errorf("line %d (%+v): empty text", i, r.Box)
		}
		if r.Score < 0.5 {
			t.Errorf("line %d %q: score %.3f below 0.5", i, r.Text, r.Score)
		}
	}

	// Known content must appear in the recognized lines.
	joined := ""
	for _, r := range results {
		joined += r.Text + "\n"
	}
	for _, want := range []string{"剁椒鱼头", "川香水煮鱼", "歌乐山辣子鸡", "富贵毛血旺", "香椿拌黄花鱼", "58元", "48元"} {
		if !strings.Contains(joined, want) {
			t.Errorf("recognized text does not contain %q", want)
		}
	}
	t.Logf("recognized %d lines", len(results))
}
