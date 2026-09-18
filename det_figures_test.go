package ppocrv6

import (
	"image"
	"image/color"
	"image/draw"
	"image/png"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"

	ort "github.com/amikos-tech/pure-onnx/ort"
)

// TestGenerateDocsFigures regenerates docs/det-overlay.png and
// docs/det-probmap.png from testdata/menu.png. It is opt-in because it writes
// into the working tree:
//
//	PPOCR_UPDATE_FIGURES=1 ONNXRUNTIME_LIB_PATH=... go test -run TestGenerateDocsFigures .
func TestGenerateDocsFigures(t *testing.T) {
	if os.Getenv("PPOCR_UPDATE_FIGURES") == "" {
		t.Skip("set PPOCR_UPDATE_FIGURES=1 to regenerate docs figures")
	}
	if os.Getenv("ONNXRUNTIME_LIB_PATH") == "" {
		t.Skip("no runtime")
	}
	f, err := os.Open(filepath.Join("testdata", "menu.png"))
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}

	d, err := NewDetector(DetConfig{ModelPath: filepath.Join("testdata", "det-v6.onnx")})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	boxes, err := d.Detect(img)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("detected %d boxes", len(boxes))

	// 1. Overlay: original image with red boxes.
	overlay := image.NewRGBA(img.Bounds())
	draw.Draw(overlay, overlay.Bounds(), img, img.Bounds().Min, draw.Src)
	red := color.RGBA{R: 255, A: 255}
	for _, b := range boxes {
		drawRect(overlay, b.MinX, b.MinY, b.MaxX, b.MaxY, red)
	}
	if err := writePNG(filepath.Join("docs", "det-overlay.png"), overlay); err != nil {
		t.Fatal(err)
	}

	// 2. Probability map: run the detection model once and save the raw map.
	resized, rw, rh, _, _ := resizeForDet(img, d.cfg.MaxSideLen)
	inData := normalizeDet(toRGBA(resized), rw, rh)
	inTensor, err := ort.NewTensor(ort.Shape{1, 3, int64(rh), int64(rw)}, inData)
	if err != nil {
		t.Fatal(err)
	}
	defer inTensor.Destroy()
	outH, outW, err := d.probeOutputSize(inTensor, rw, rh)
	if err != nil {
		t.Fatal(err)
	}
	outTensor, err := ort.NewEmptyTensor[float32](ort.Shape{1, 1, int64(outH), int64(outW)})
	if err != nil {
		t.Fatal(err)
	}
	defer outTensor.Destroy()
	if err := d.sess.RunWithValues([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		t.Fatal(err)
	}
	pm := outTensor.GetData()
	gray := image.NewRGBA(image.Rect(0, 0, outW, outH))
	for y := 0; y < outH; y++ {
		for x := 0; x < outW; x++ {
			v := uint8(pm[y*outW+x] * 255)
			gray.SetRGBA(x, y, color.RGBA{R: v, G: v, B: v, A: 255})
		}
	}
	if err := writePNG(filepath.Join("docs", "det-probmap.png"), gray); err != nil {
		t.Fatal(err)
	}
	t.Logf("prob map %dx%d written", outW, outH)
}

func drawRect(img *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	b := img.Bounds()
	set := func(x, y int) {
		if x >= b.Min.X && x < b.Max.X && y >= b.Min.Y && y < b.Max.Y {
			img.SetRGBA(x, y, c)
		}
	}
	for x := x0; x < x1; x++ {
		for _, y := range []int{y0, y0 + 1, y1 - 1, y1} {
			set(x, y)
		}
	}
	for y := y0; y < y1; y++ {
		for _, x := range []int{x0, x0 + 1, x1 - 1, x1} {
			set(x, y)
		}
	}
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
