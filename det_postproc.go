package ppocrv6

import (
	"image"
	"image/color"
)

// resizeForDet resizes img keeping the aspect ratio so the longer side is at
// most maxSide, then pads to multiples of 32 (the detection model stride).
// Returns the resized image, its dimensions, and the scale factors from the
// original to the resized (net of padding).
func resizeForDet(img image.Image, maxSide int) (resized image.Image, rw, rh int, sx, sy float64) {
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()

	ratio := float64(srcW) / float64(srcH)
	if srcW > srcH {
		rw = maxSide
		rh = int(float64(rw)/ratio + 0.5)
	} else {
		rh = maxSide
		rw = int(float64(rh)*ratio + 0.5)
	}
	// Pad to multiple of 32.
	pw := (32 - rw%32) % 32
	ph := (32 - rh%32) % 32
	rw += pw
	rh += ph

	rgba := toRGBA(img)
	rows, nw := resizeRowsNoPad(rgba, srcW, srcH, rw-pw, rh-ph)
	// Pad right/bottom with white (255).
	dst := image.NewRGBA(image.Rect(0, 0, rw, rh))
	for y := 0; y < rh-ph; y++ {
		copy(dst.Pix[y*dst.Stride:], rows[y][:nw])
	}
	// Fill padding rows with white.
	if ph > 0 {
		for y := rh - ph; y < rh; y++ {
			for x := 0; x < rw; x++ {
				dst.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}
	// Fill padding columns.
	if pw > 0 {
		for y := 0; y < rh; y++ {
			for x := rw - pw; x < rw; x++ {
				dst.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			}
		}
	}
	sx = float64(srcW) / float64(rw-pw)
	sy = float64(srcH) / float64(rh-ph)
	return dst, rw, rh, sx, sy
}

// resizeRowsNoPad resizes RGBA to (w, h) with bilinear, returning rows of
// RGBA bytes (not structs). No padding is applied.
func resizeRowsNoPad(src *image.RGBA, srcW, srcH, w, h int) ([][]byte, int) {
	rows := make([][]byte, h)
	sx := float64(srcW) / float64(w)
	sy := float64(srcH) / float64(h)
	clampX := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= srcW {
			return srcW - 1
		}
		return v
	}
	clampY := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= srcH {
			return srcH - 1
		}
		return v
	}
	for dy := 0; dy < h; dy++ {
		y := float64(dy)*sy + 0.5*sy - 0.5
		y0 := clampY(int(y))
		y1 := clampY(y0 + 1)
		fy := float32(y - float64(int(y)))
		if fy < 0 {
			fy = 0
		}

		row := make([]byte, w*4)
		top := src.PixOffset(0, y0)
		bot := src.PixOffset(0, y1)
		for dx := 0; dx < w; dx++ {
			x := float64(dx)*sx + 0.5*sx - 0.5
			x0 := clampX(int(x))
			x1 := clampX(x0 + 1)
			fx := float32(x - float64(int(x)))
			if fx < 0 {
				fx = 0
			}

			w00 := (1 - fx) * (1 - fy)
			w10 := fx * (1 - fy)
			w01 := (1 - fx) * fy
			w11 := fx * fy
			off := dx * 4
			for c := 0; c < 4; c++ {
				v := float32(src.Pix[top+x0*4+c])*w00 +
					float32(src.Pix[top+x1*4+c])*w10 +
					float32(src.Pix[bot+x0*4+c])*w01 +
					float32(src.Pix[bot+x1*4+c])*w11
				row[off+c] = byte(v + 0.5)
			}
		}
		rows[dy] = row
	}
	return rows, w * 4
}

// extractBoxes thresholds a probability map and finds axis-aligned bounding
// boxes via connected-component labeling. Coordinates are in the resized
// (model output) space.
func extractBoxes(prob []float32, h, w int, thresh float32, minArea int) []Box {
	// Threshold → binary.
	bin := make([]uint8, h*w)
	for i, v := range prob {
		if v > thresh {
			bin[i] = 1
		}
	}
	// Connected-component labeling (two-pass with union-find).
	labels, n := ccl(bin, w, h)

	// Compute bounding box per connected component.
	minX := make([]int, n+1)
	minY := make([]int, n+1)
	maxX := make([]int, n+1)
	maxY := make([]int, n+1)
	for i := 1; i <= n; i++ {
		minX[i] = w
		minY[i] = h
	}
	for i, lb := range labels {
		if lb == 0 {
			continue
		}
		x := i % w
		y := i / w
		if x < minX[lb] {
			minX[lb] = x
		}
		if y < minY[lb] {
			minY[lb] = y
		}
		if x > maxX[lb] {
			maxX[lb] = x
		}
		if y > maxY[lb] {
			maxY[lb] = y
		}
	}
	out := make([]Box, 0, n)
	for i := 1; i <= n; i++ {
		b := Box{minX[i], minY[i], maxX[i] + 1, maxY[i] + 1}
		if b.Width()*b.Height() >= minArea {
			out = append(out, b)
		}
	}
	return mergeLineBoxes(out)
}

// mergeLineBoxes merges boxes that belong to the same visual text line:
// they overlap vertically by more than half of the shorter box and the
// horizontal gap between them is small relative to their height. The DB
// probability map often splits one line into several character clusters
// (for example dish names and prices), and recognition expects whole lines.
func mergeLineBoxes(boxes []Box) []Box {
	if len(boxes) < 2 {
		return boxes
	}
	changed := true
	for changed {
		changed = false
		for i := 0; i < len(boxes); i++ {
			for j := i + 1; j < len(boxes); j++ {
				if sameLine(boxes[i], boxes[j]) {
					boxes[i] = unionBox(boxes[i], boxes[j])
					boxes = append(boxes[:j], boxes[j+1:]...)
					changed = true
					j--
				}
			}
		}
	}
	return boxes
}

// sameLine reports whether two boxes belong to the same text line.
func sameLine(a, b Box) bool {
	// Vertical overlap must cover more than half of the shorter box.
	overlapY := min(a.MaxY, b.MaxY) - max(a.MinY, b.MinY)
	shorter := min(a.Height(), b.Height())
	if overlapY*2 < shorter {
		return false
	}
	// Horizontal gap must be small compared to the line height, so that
	// separate columns (dish list vs. price list) do not merge.
	gap := 0
	switch {
	case b.MinX > a.MaxX:
		gap = b.MinX - a.MaxX
	case a.MinX > b.MaxX:
		gap = a.MinX - b.MaxX
	}
	maxGap := min(a.Height(), b.Height())
	return gap <= maxGap
}

func unionBox(a, b Box) Box {
	return Box{
		MinX: min(a.MinX, b.MinX),
		MinY: min(a.MinY, b.MinY),
		MaxX: max(a.MaxX, b.MaxX),
		MaxY: max(a.MaxY, b.MaxY),
	}
}

// ccl labels 8-connected components on a binary (0/1) grid. Returns a
// label map (0 = background) and the number of components found.
func ccl(bin []uint8, w, h int) ([]int32, int) {
	labels := make([]int32, len(bin))
	parent := []int32{0} // 1-indexed: parent[label]
	nextLabel := int32(1)

	neighbors := func(x, y int) []int32 {
		var n []int32
		// 4 prior neighbors (8-connected): left, top-left, top, top-right.
		offs := [][2]int{{-1, 0}, {-1, -1}, {0, -1}, {1, -1}}
		for _, o := range offs {
			nx, ny := x+o[0], y+o[1]
			if nx >= 0 && nx < w && ny >= 0 {
				if l := labels[ny*w+nx]; l > 0 {
					n = append(n, l)
				}
			}
		}
		return n
	}

	find := func(l int32) int32 {
		for parent[l] != l {
			parent[l] = parent[parent[l]]
			l = parent[l]
		}
		return l
	}

	union := func(a, b int32) {
		ra, rb := find(a), find(b)
		if ra != rb {
			if ra < rb {
				parent[rb] = ra
			} else {
				parent[ra] = rb
			}
		}
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if bin[y*w+x] == 0 {
				continue
			}
			ns := neighbors(x, y)
			if len(ns) == 0 {
				labels[y*w+x] = nextLabel
				parent = append(parent, nextLabel)
				nextLabel++
			} else {
				minL := ns[0]
				for _, l := range ns[1:] {
					if l < minL {
						minL = l
					}
				}
				labels[y*w+x] = minL
				for _, l := range ns {
					union(minL, l)
				}
			}
		}
	}

	// Flatten: reassign each label to its root.
	count := 0
	for i, l := range labels {
		if l > 0 {
			r := find(l)
			labels[i] = r
			if r > int32(count) {
				count = int(r)
			}
		}
	}
	return labels, count
}
