package ppocrv6

import "fmt"

// Defaults verified against the exported PP-OCRv6_small_rec ONNX model
// (ModelScope: PaddlePaddle/PP-OCRv6_small_rec_onnx) and the PaddleOCR
// source configuration (configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml).
const (
	// DefaultHeight is the fixed input height of the recognition model.
	DefaultHeight = 48

	// DefaultMean and DefaultStd are the normalization parameters:
	// (pixel/255 - mean) / std.
	DefaultMean = 0.5
	DefaultStd  = 0.5

	// DefaultScale normalizes 8-bit pixels to [0, 1].
	DefaultScale = 1.0 / 255.0

	// DefaultDictSize is the number of characters in ppocrv6_dict.txt.
	// The model outputs dictSize+2 classes: CTC blank at index 0, the
	// dictionary at indices 1..dictSize, and the space at index dictSize+1
	// (use_space_char=true in the PaddleOCR config).
	DefaultDictSize = 18708

	// DefaultModelInputName / DefaultModelOutputName are the tensor names of
	// the exported ONNX graph.
	DefaultModelInputName  = "x"
	DefaultModelOutputName = "fetch_name_0"
)

// Config controls the recognition pipeline. Zero values fall back to the
// model-verified defaults via [Config.normalize].
type Config struct {
	// ModelPath is the path to the PP-OCRv6_small_rec inference.onnx file.
	ModelPath string

	// DictPath is the path to ppocrv6_dict.txt.
	DictPath string

	// Height is the fixed resize height (default 48).
	Height int

	// MaxWidth caps the proportional resize width; 0 disables the cap.
	// Long lines are resized to this width, distorting the aspect ratio.
	MaxWidth int

	// BGR feeds the model channels in BGR order. The default (false) is RGB,
	// which is the order PP-OCRv6 was trained and exported in (the PaddleX
	// inference pipeline reads images with ReadImage(format="RGB")). Set it
	// only when the input images are already BGR-ordered.
	BGR bool

	// Mean and Std are the per-channel normalization parameters applied as
	// (pixel*Scale - Mean) / Std.
	Mean float32
	Std  float32

	// Scale maps 8-bit pixels into the model input range.
	Scale float32

	// BlankID is the CTC blank class index (default 0).
	BlankID int

	// InputName / OutputName override the ONNX graph tensor names.
	InputName  string
	OutputName string

	// LibraryPath optionally points to a prebuilt libonnxruntime.so. It is
	// used only when the runtime is not yet initialized; otherwise the
	// ONNXRUNTIME_LIB_PATH environment variable or bootstrap download
	// applies.
	LibraryPath string

	// DictSize overrides the dictionary size when a non-standard
	// dictionary file is used. 0 means derive from the dict file itself.
	DictSize int
}

// applyDefaults fills zero-value fields with the model-verified defaults.
// It is safe to call on a partial Config that only needs preprocessing
// parameters (for example the detection stage reusing the normalization).
func (c *Config) applyDefaults() {
	if c.Height == 0 {
		c.Height = DefaultHeight
	}
	if c.Mean == 0 {
		c.Mean = DefaultMean
	}
	if c.Std == 0 {
		c.Std = DefaultStd
	}
	if c.Scale == 0 {
		c.Scale = DefaultScale
	}
	if c.InputName == "" {
		c.InputName = DefaultModelInputName
	}
	if c.OutputName == "" {
		c.OutputName = DefaultModelOutputName
	}
}

// normalize fills defaults and validates the required paths and values.
func (c *Config) normalize() error {
	c.applyDefaults()
	if c.ModelPath == "" {
		return fmt.Errorf("ppocrv6: ModelPath is required")
	}
	if c.Height <= 0 {
		return fmt.Errorf("ppocrv6: invalid Height %d", c.Height)
	}
	if c.MaxWidth < 0 {
		return fmt.Errorf("ppocrv6: invalid MaxWidth %d", c.MaxWidth)
	}
	if c.BlankID < 0 {
		return fmt.Errorf("ppocrv6: invalid BlankID %d", c.BlankID)
	}
	return nil
}
