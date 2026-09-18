package ppocrv6

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestPreprocessNormalization(t *testing.T) {
	// A 2x2 image: all-white pixels. After (255/255 - 0.5)/0.5 = 1.0
	// every channel of every pixel must be exactly 1.0.
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := range 2 * 2 {
		img.Pix[i*4] = 255
		img.Pix[i*4+1] = 255
		img.Pix[i*4+2] = 255
		img.Pix[i*4+3] = 255
	}
	cfg := &Config{Height: 48}
	cfg.normalize()

	data, w, err := Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	if w != 48 { // ceil(48*2/2) = 48
		t.Fatalf("width = %d, want 48", w)
	}
	if len(data) != 3*48*48 {
		t.Fatalf("data length = %d, want %d", len(data), 3*48*48)
	}
	for i, v := range data {
		if math.Abs(float64(v)-1.0) > 1e-6 {
			t.Fatalf("data[%d] = %v, want 1.0 (white pixel)", i, v)
		}
	}
}

func TestPreprocessBlackAndChannelOrder(t *testing.T) {
	// Black pixel: (0 - 0.5)/0.5 = -1.0. A pure-red pixel must put 1.0 in
	// channel 0 (R) with the default RGB order and in channel 2 (B position)
	// with BGR.
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	cfg := &Config{Height: 48}
	cfg.normalize()
	data, _, err := Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	// NCHW: channel c starts at c*h*w. Pixel (0,0) is at +0.
	if math.Abs(float64(data[0])-1.0) > 1e-6 {
		t.Errorf("R channel = %v, want 1.0", data[0])
	}
	if math.Abs(float64(data[48*48])+1.0) > 1e-6 {
		t.Errorf("G channel = %v, want -1.0", data[48*48])
	}
	if math.Abs(float64(data[2*48*48])+1.0) > 1e-6 {
		t.Errorf("B channel = %v, want -1.0", data[2*48*48])
	}

	cfg.BGR = true
	data, _, err = Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("Preprocess BGR: %v", err)
	}
	// BGR order: R value lands in the last channel.
	if math.Abs(float64(data[2*48*48])-1.0) > 1e-6 {
		t.Errorf("BGR: B channel = %v, want 1.0 (red pixel in B slot)", data[2*48*48])
	}
}

func TestPreprocessAspectAndMaxWidth(t *testing.T) {
	// 96x48 source -> ceil(48*96/48) = 96 wide.
	img := image.NewRGBA(image.Rect(0, 0, 96, 48))
	cfg := &Config{Height: 48, MaxWidth: 64}
	cfg.normalize()
	_, w, err := Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	if w != 64 {
		t.Fatalf("width = %d, want capped 64", w)
	}
	cfg.MaxWidth = 0
	_, w, err = Preprocess(img, cfg)
	if err != nil {
		t.Fatalf("Preprocess: %v", err)
	}
	if w != 96 {
		t.Fatalf("width = %d, want 96", w)
	}
}
