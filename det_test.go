package ppocrv6

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestResizeForDetPadsToMultipleOf32(t *testing.T) {
	// A 970x1417 page: the long side becomes 960, the short side 657, then
	// both are padded up to multiples of 32.
	img := image.NewRGBA(image.Rect(0, 0, 970, 1417))
	_, rw, rh, sx, sy := resizeForDet(img, 960)
	if rw != 672 || rh != 960 {
		t.Fatalf("resized = %dx%d, want 672x960", rw, rh)
	}
	if rw%32 != 0 || rh%32 != 0 {
		t.Fatalf("resized %dx%d is not a multiple of 32", rw, rh)
	}
	// sx is original pixels per resized pixel, measured against the
	// unpadded width (657).
	if math.Abs(sx-970.0/657.0) > 1e-9 {
		t.Errorf("sx = %v, want %v", sx, 970.0/657.0)
	}
	if math.Abs(sy-1417.0/960.0) > 1e-9 {
		t.Errorf("sy = %v, want %v", sy, 1417.0/960.0)
	}
}

func TestScaleBoxesToOriginal(t *testing.T) {
	// Regression: the box must be multiplied by original-per-resized pixels
	// (1.476 for a 970-wide page), not divided by it. Dividing shrank every
	// box toward the origin and cut the last characters off each line.
	boxes := []Box{{MinX: 84, MinY: 214, MaxX: 301, MaxY: 226}}
	sx, sy := 970.0/657.0, 1417.0/960.0
	scaleBoxesToOriginal(boxes, sx, sy, 970, 1417)

	want := Box{MinX: 124, MinY: 316, MaxX: 444, MaxY: 334}
	if boxes[0] != want {
		t.Errorf("scaled box = %+v, want %+v", boxes[0], want)
	}
	// The inverse factor is what the bug did; assert it is not in use.
	buggy := Box{
		MinX: int(float64(84) / sx),
		MaxX: int(float64(301) / sx),
	}
	if boxes[0].MinX == buggy.MinX && boxes[0].MaxX == buggy.MaxX {
		t.Errorf("boxes still scaled by the inverse factor")
	}
}

func TestScaleBoxesToOriginalClamps(t *testing.T) {
	boxes := []Box{{MinX: -20, MinY: -5, MaxX: 1200, MaxY: 1500}}
	scaleBoxesToOriginal(boxes, 1.0, 1.0, 970, 1417)
	want := Box{MinX: 0, MinY: 0, MaxX: 970, MaxY: 1417}
	if boxes[0] != want {
		t.Errorf("clamped box = %+v, want %+v", boxes[0], want)
	}
}

func TestNormalizeDetImageNet(t *testing.T) {
	// White pixel: (1 - mean) / std per channel.
	img := image.NewRGBA(image.Rect(0, 0, 1, 1))
	img.SetRGBA(0, 0, color.RGBA{R: 255, G: 255, B: 255, A: 255})
	data := normalizeDet(img, 1, 1)
	for c := range 3 {
		want := float32((1.0 - detMean[c]) / detStd[c])
		if math.Abs(float64(data[c]-want)) > 1e-5 {
			t.Errorf("channel %d = %v, want %v", c, data[c], want)
		}
	}
}

func TestExtractBoxesThresholdAndMinArea(t *testing.T) {
	const w, h = 40, 12
	prob := make([]float32, w*h)

	// A 6x4 blob at (10, 4) above the threshold.
	for y := 4; y < 8; y++ {
		for x := 10; x < 16; x++ {
			prob[y*w+x] = 0.9
		}
	}
	// A 2x2 blob below the threshold must be ignored.
	for y := 1; y < 3; y++ {
		for x := 30; x < 32; x++ {
			prob[y*w+x] = 0.2
		}
	}

	boxes := extractBoxes(prob, h, w, 0.3, 10)
	if len(boxes) != 1 {
		t.Fatalf("boxes = %d, want 1 (%+v)", len(boxes), boxes)
	}
	want := Box{MinX: 10, MinY: 4, MaxX: 16, MaxY: 8}
	if boxes[0] != want {
		t.Errorf("box = %+v, want %+v", boxes[0], want)
	}

	// The same blob with a min area above its size is dropped.
	if got := extractBoxes(prob, h, w, 0.3, 6*4+1); len(got) != 0 {
		t.Errorf("min area filter returned %d boxes, want 0", len(got))
	}
}

func TestMergeLineBoxesJoinsNeighbours(t *testing.T) {
	// Two boxes on the same line, close together, must merge into one.
	boxes := []Box{
		{MinX: 10, MinY: 10, MaxX: 40, MaxY: 30},
		{MinX: 45, MinY: 11, MaxX: 80, MaxY: 31},
	}
	merged := mergeLineBoxes(boxes)
	if len(merged) != 1 {
		t.Fatalf("merged = %d boxes, want 1 (%+v)", len(merged), merged)
	}
	want := Box{MinX: 10, MinY: 10, MaxX: 80, MaxY: 31}
	if merged[0] != want {
		t.Errorf("merged box = %+v, want %+v", merged[0], want)
	}
}

func TestCCLCompactsLabels(t *testing.T) {
	// Three separate blobs: every returned label must have pixels, and the
	// component count must equal the number of blobs. Returning the largest
	// root label left gaps that extractBoxes turned into inverted boxes.
	const w, h = 20, 6
	bin := make([]uint8, w*h)
	for _, p := range [][2]int{{1, 1}, {5, 1}, {10, 3}} {
		bin[p[1]*w+p[0]] = 1
	}
	labels, n := ccl(bin, w, h)
	if n != 3 {
		t.Fatalf("components = %d, want 3", n)
	}
	seen := map[int32]int{}
	for _, l := range labels {
		if l > 0 {
			seen[l]++
		}
	}
	if len(seen) != 3 {
		t.Fatalf("distinct labels = %d, want 3 (%v)", len(seen), seen)
	}
	for l := int32(1); l <= int32(n); l++ {
		if seen[l] == 0 {
			t.Errorf("label %d has no pixels", l)
		}
	}

	boxes := extractBoxesFromBinary(bin, h, w, 0)
	for _, b := range boxes {
		if b.Width() <= 0 || b.Height() <= 0 {
			t.Errorf("degenerate box %+v", b)
		}
	}
}

// extractBoxesFromBinary runs the component-to-box stage on a binary map
// (threshold 0 so every set pixel counts).
func extractBoxesFromBinary(bin []uint8, h, w int, minArea int) []Box {
	prob := make([]float32, len(bin))
	for i, v := range bin {
		if v == 1 {
			prob[i] = 1
		}
	}
	return extractBoxes(prob, h, w, 0.5, minArea)
}
