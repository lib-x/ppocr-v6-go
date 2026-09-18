package ppocrv6

import (
	"errors"
	"fmt"
	"image"
	"regexp"
	"sync"

	"github.com/amikos-tech/pure-onnx/ort"

	"github.com/lib-x/ppocr-v6-go/internal/onnxmeta"
)

// DetConfig configures the text detection stage.
type DetConfig struct {
	// ModelPath is the path to the detection ONNX file (e.g., PP-OCRv6_medium_det_onnx).
	ModelPath string
	// MaxSideLen constrains the longer side of the image during preprocessing
	// (default 960). Images are resized keeping the aspect ratio, then padded
	// to a multiple of 32.
	MaxSideLen int
	// Threshold is the minimum probability for a pixel to be considered part
	// of a text region (default 0.3).
	Threshold float32
	// MinBoxArea is the minimum pixel area of a detected text box (in the
	// resized coordinate space) below which a box is discarded (default 10).
	MinBoxArea int
	// UnclipRatio expands each detected box to compensate the DB model's
	// shrink ratio (default 1.5). The offset follows the DB formula
	// d = area * ratio / perimeter.
	UnclipRatio float32
}

// normalize fills defaults.
func (c *DetConfig) normalize() error {
	if c.ModelPath == "" {
		return fmt.Errorf("ppocrv6: DetConfig.ModelPath is required")
	}
	if c.MaxSideLen <= 0 {
		c.MaxSideLen = 960
	}
	if c.Threshold <= 0 {
		c.Threshold = 0.3
	}
	if c.MinBoxArea <= 0 {
		c.MinBoxArea = 10
	}
	if c.UnclipRatio <= 0 {
		c.UnclipRatio = 1.5
	}
	return nil
}

// Box is an axis-aligned box in image coordinates.
type Box struct {
	MinX, MinY int
	MaxX, MaxY int
}

func (b Box) Width() int  { return b.MaxX - b.MinX }
func (b Box) Height() int { return b.MaxY - b.MinY }

// Detector runs text detection via the PP-OCRv6 detection ONNX model and
// DB (Differentiable Binarization) post-processing.
type Detector struct {
	cfg     DetConfig
	sess    *ort.AdvancedSession
	meta    *onnxmeta.Model
	inName  string
	outName string

	shapeMu sync.Mutex
	outSize map[[2]int][2]int // (resized w, h) -> probed output (h, w)
}

// Detection preprocessing constants (PaddleOCR det pipeline): ImageNet
// normalization with the longer side capped and padding to multiples of 32.
var detMean = [3]float32{0.485, 0.456, 0.406}
var detStd = [3]float32{0.229, 0.224, 0.225}

// NewDetector loads the detection ONNX model and prepares a session.
func NewDetector(cfg DetConfig) (*Detector, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}
	meta, err := onnxmeta.ParseFile(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("ppocrv6: inspect det model: %w", err)
	}
	if len(meta.Inputs) == 0 || len(meta.Outputs) == 0 {
		return nil, fmt.Errorf("ppocrv6: det model has no inputs or outputs")
	}
	inName := meta.Inputs[0].Name
	outName := meta.Outputs[0].Name

	if err := initEnvironment(""); err != nil {
		return nil, err
	}
	envMu.Lock()
	recCount++
	envMu.Unlock()

	// placeholder tensors
	phIn, _ := ort.NewTensor(ort.Shape{1, 3, 32, 32}, make([]float32, 3*32*32))
	phOut, _ := ort.NewEmptyTensor[float32](ort.Shape{1, 1, 8, 8})
	sess, err := ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{inName}, []string{outName},
		[]ort.Value{phIn}, []ort.Value{phOut},
		nil,
	)
	if err != nil {
		phIn.Destroy()
		phOut.Destroy()
		return nil, fmt.Errorf("ppocrv6: create det session: %w", err)
	}
	return &Detector{
		cfg: cfg, sess: sess, meta: meta,
		inName: inName, outName: outName,
		outSize: make(map[[2]int][2]int),
	}, nil
}

// Close releases the detection session.
func (d *Detector) Close() error {
	envMu.Lock()
	recCount--
	destroyEnv := recCount == 0
	envMu.Unlock()
	var errs []error
	if d.sess != nil {
		if err := d.sess.Destroy(); err != nil {
			errs = append(errs, err)
		}
		d.sess = nil
	}
	if destroyEnv {
		if err := ort.DestroyEnvironment(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Detect finds text regions in img. Box coordinates are in the original
// image coordinate space.
func (d *Detector) Detect(img image.Image) ([]Box, error) {
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()

	// resize + pad to multiple of 32
	resized, rw, rh, rxScale, ryScale := resizeForDet(img, d.cfg.MaxSideLen)

	// Detection normalization differs from recognition: PaddleOCR uses
	// ImageNet mean/std for the detection stage.
	inData := normalizeDet(toRGBA(resized), rw, rh)

	inTensor, err := ort.NewTensor(ort.Shape{1, 3, int64(rh), int64(rw)}, inData)
	if err != nil {
		return nil, fmt.Errorf("ppocrv6: det input tensor: %w", err)
	}
	defer inTensor.Destroy()

	// The detection output shape is dynamic ([1, 1, h', w']); probe for the
	// exact dims once per input size, as ONNX Runtime requires an exact match.
	outH, outW, err := d.outputSize(inTensor, rw, rh)
	if err != nil {
		return nil, err
	}
	outTensor, err := ort.NewEmptyTensor[float32](ort.Shape{1, 1, int64(outH), int64(outW)})
	if err != nil {
		return nil, fmt.Errorf("ppocrv6: det output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := d.sess.RunWithValues([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return nil, fmt.Errorf("ppocrv6: det inference: %w", err)
	}

	probMap := outTensor.GetData()
	boxes := extractBoxes(probMap, outH, outW, d.cfg.Threshold, d.cfg.MinBoxArea)

	// Unclip (DB shrink compensation) in resized space, then map the boxes
	// back to original image coordinates.
	for i := range boxes {
		boxes[i] = unclipBox(boxes[i], d.cfg.UnclipRatio)
	}
	scaleBoxesToOriginal(boxes, rxScale, ryScale, srcW, srcH)

	return boxes, nil
}

// unclipBox expands a box by the DB unclip offset
// d = area * ratio / perimeter, applied to all four sides.
func unclipBox(b Box, ratio float32) Box {
	w, h := b.Width(), b.Height()
	perimeter := 2 * (w + h)
	if perimeter == 0 {
		return b
	}
	d := int(float32(w*h)*ratio/float32(perimeter) + 0.5)
	if d < 1 {
		d = 1
	}
	return Box{b.MinX - d, b.MinY - d, b.MaxX + d, b.MaxY + d}
}

// normalizeDet converts a resized RGBA image to NCHW float32 with the
// ImageNet normalization used by the detection model.
func normalizeDet(rgba *image.RGBA, w, h int) []float32 {
	out := make([]float32, 3*h*w)
	const scale = 1.0 / 255.0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*rgba.Stride + x*4
			for c := 0; c < 3; c++ {
				ch := float32(rgba.Pix[i+c]) * scale
				out[c*h*w+y*w+x] = (ch - detMean[c]) / detStd[c]
			}
		}
	}
	return out
}

var detShapeRe = regexp.MustCompile(`Requested shape:\{[^,]+,\s*[^,]+,\s*(\d+),\s*(\d+)\}`)

// outputSize returns the detection output dims for a resized input size,
// probing the model once per distinct (w, h) and caching the result.
func (d *Detector) outputSize(in *ort.Tensor[float32], rw, rh int) (int, int, error) {
	key := [2]int{rw, rh}
	d.shapeMu.Lock()
	cached, ok := d.outSize[key]
	d.shapeMu.Unlock()
	if ok {
		return cached[0], cached[1], nil
	}
	h, w, err := d.probeOutputSize(in, rw, rh)
	if err != nil {
		return 0, 0, err
	}
	d.shapeMu.Lock()
	d.outSize[key] = [2]int{h, w}
	d.shapeMu.Unlock()
	return h, w, nil
}

// scaleBoxesToOriginal maps boxes from the resized (and padded) detection
// space back to original image coordinates and clamps them to the image
// bounds. sx and sy are original pixels per resized pixel, so they are
// multiplied in, never divided.
func scaleBoxesToOriginal(boxes []Box, sx, sy float64, srcW, srcH int) {
	for i := range boxes {
		boxes[i].MinX = int(float64(boxes[i].MinX)*sx + 0.5)
		boxes[i].MinY = int(float64(boxes[i].MinY)*sy + 0.5)
		boxes[i].MaxX = int(float64(boxes[i].MaxX)*sx + 0.5)
		boxes[i].MaxY = int(float64(boxes[i].MaxY)*sy + 0.5)
		boxes[i].MinX = max(boxes[i].MinX, 0)
		boxes[i].MinY = max(boxes[i].MinY, 0)
		boxes[i].MaxX = min(boxes[i].MaxX, srcW)
		boxes[i].MaxY = min(boxes[i].MaxY, srcH)
	}
}

// probeOutputSize discovers the detection model's output spatial dims.
//
// Unlike the recognition model (whose final Softmax node validates the
// output shape), the detection graph silently accepts an oversized output
// tensor and writes only the real data into the buffer prefix. A
// deliberately too-small tensor therefore forces ONNX Runtime to report the
// exact computed shape in its error message.
func (d *Detector) probeOutputSize(inTensor *ort.Tensor[float32], rw, rh int) (int, int, error) {
	probe, err := ort.NewEmptyTensor[float32](ort.Shape{1, 1, 1, 1})
	if err != nil {
		return 0, 0, fmt.Errorf("ppocrv6: det probe tensor: %w", err)
	}
	defer probe.Destroy()

	err = d.sess.RunWithValues([]ort.Value{inTensor}, []ort.Value{probe})
	if err == nil {
		return 0, 0, fmt.Errorf("ppocrv6: det shape probe unexpectedly accepted a 1x1 output tensor")
	}
	m := detShapeRe.FindStringSubmatch(err.Error())
	if m == nil {
		return 0, 0, fmt.Errorf("ppocrv6: det shape probe failed: %w", err)
	}
	h := atoiSafe(m[1])
	w := atoiSafe(m[2])
	if h <= 0 || w <= 0 {
		return 0, 0, fmt.Errorf("ppocrv6: det shape probe returned [%s %s]: %w", m[1], m[2], err)
	}
	return h, w, nil
}
