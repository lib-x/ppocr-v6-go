package ppocrv6

import (
	"cmp"
	"fmt"
	"image"
	"slices"
)

// LineResult is one recognized text line.
type LineResult struct {
	// Box is the detected text region in the original image coordinates.
	Box Box
	// Text is the recognized text of the line.
	Text string
	// Score is the recognition confidence.
	Score float32
}

// Pipeline runs the full OCR flow: text detection, line cropping and
// recognition. Use it for whole-image OCR; use [Recognizer] directly when
// the input is already a cropped text line.
type Pipeline struct {
	det *Detector
	rec *Recognizer
}

// NewPipeline loads both models. The recognition dictionary defaults to the
// embedded ppocrv6_dict.txt when Config.DictPath is empty.
func NewPipeline(recCfg Config, detCfg DetConfig) (*Pipeline, error) {
	rec, err := New(recCfg)
	if err != nil {
		return nil, err
	}
	det, err := NewDetector(detCfg)
	if err != nil {
		rec.Close()
		return nil, err
	}
	return &Pipeline{det: det, rec: rec}, nil
}

// Recognize detects text lines in the full image and recognizes each of
// them. Results are ordered top-to-bottom, then left-to-right.
func (p *Pipeline) Recognize(img image.Image) ([]LineResult, error) {
	boxes, err := p.det.Detect(img)
	if err != nil {
		return nil, err
	}
	sortBoxes(boxes)

	results := make([]LineResult, 0, len(boxes))
	for _, b := range boxes {
		crop, err := cropImage(img, b)
		if err != nil {
			continue
		}
		res, err := p.rec.Recognize(crop)
		if err != nil {
			return nil, fmt.Errorf("ppocrv6: recognize box %+v: %w", b, err)
		}
		if res.Text == "" {
			continue
		}
		results = append(results, LineResult{Box: b, Text: res.Text, Score: res.Score})
	}
	return results, nil
}

// Close releases both sessions.
func (p *Pipeline) Close() error {
	var first error
	if p.rec != nil {
		if err := p.rec.Close(); err != nil && first == nil {
			first = err
		}
		p.rec = nil
	}
	if p.det != nil {
		if err := p.det.Close(); err != nil && first == nil {
			first = err
		}
		p.det = nil
	}
	return first
}

// sortBoxes orders boxes by reading order: primarily by vertical position,
// then horizontally. Boxes whose vertical centers are close (within half the
// average height) are treated as the same visual line.
func sortBoxes(boxes []Box) {
	if len(boxes) < 2 {
		return
	}
	var totalH int
	for _, b := range boxes {
		totalH += b.Height()
	}
	lineTolerance := totalH / len(boxes) / 2
	if lineTolerance < 1 {
		lineTolerance = 1
	}
	slices.SortStableFunc(boxes, func(a, b Box) int {
		ca := (a.MinY + a.MaxY) / 2
		cb := (b.MinY + b.MaxY) / 2
		if diff := ca - cb; diff > lineTolerance || diff < -lineTolerance {
			return cmp.Compare(ca, cb)
		}
		return cmp.Compare(a.MinX, b.MinX)
	})
}

// cropImage returns the sub-image of img covered by box. It works with any
// image.Image implementation by copying pixels into a fresh RGBA image.
func cropImage(img image.Image, b Box) (image.Image, error) {
	bounds := img.Bounds()
	minX, minY := b.MinX+bounds.Min.X, b.MinY+bounds.Min.Y
	maxX, maxY := b.MaxX+bounds.Min.X, b.MaxY+bounds.Min.Y
	if minX < bounds.Min.X {
		minX = bounds.Min.X
	}
	if minY < bounds.Min.Y {
		minY = bounds.Min.Y
	}
	if maxX > bounds.Max.X {
		maxX = bounds.Max.X
	}
	if maxY > bounds.Max.Y {
		maxY = bounds.Max.Y
	}
	w, h := maxX-minX, maxY-minY
	if w < 1 || h < 1 {
		return nil, fmt.Errorf("ppocrv6: empty crop for box %+v", b)
	}
	// Fast path for RGBA images.
	if rgba, ok := img.(*image.RGBA); ok {
		dst := image.NewRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			copy(dst.Pix[y*dst.Stride:y*dst.Stride+w*4], rgba.Pix[(minY+y-rgba.Rect.Min.Y)*rgba.Stride+(minX-rgba.Rect.Min.X)*4:])
		}
		return dst, nil
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.Set(x, y, img.At(minX+x, minY+y))
		}
	}
	return dst, nil
}
