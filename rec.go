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

// Recognizer runs the PP-OCRv6 recognition model on text-line images.
//
// The model contract (input/output names, input height, output class count)
// is read from the ONNX file itself and validated against the dictionary, so
// the same code works with any PP-OCRv6 variant (small/medium/tiny) export.
//
// A Recognizer is not safe for concurrent use; create one per goroutine or
// guard calls with a mutex.
type Recognizer struct {
	cfg  Config
	dict *Dict
	sess *ort.AdvancedSession
	// placeholder tensors bound at session construction (see New).
	placeholderIn  *ort.Tensor[float32]
	placeholderOut *ort.Tensor[float32]
	classes        int
	closed         bool

	shapeMu      sync.Mutex
	outputLenMap map[int]int // resized width -> CTC sequence length (probed)
}

var (
	envInitSet bool // true once the library has initialized the runtime
	envMu      sync.Mutex
	recCount   int // open Recognizer count, gates DestroyEnvironment
)

// New loads the dictionary and the ONNX model and prepares an inference
// session.
//
// The model contract is introspected from the ONNX graph unless
// Config.InputName / Config.OutputName are explicitly set:
//   - input name/shape (batch, 3, Height, width) — Height must match
//     Config.Height (default: the model's static height, 48);
//   - output class count — the dictionary is validated against it.
//
// The ONNX Runtime shared library is resolved through
// ort.InitializeEnvironmentWithBootstrap: set ONNXRUNTIME_LIB_PATH to a
// prebuilt libonnxruntime.so, or let bootstrap download the default build.
// Config.LibraryPath, when set and the runtime is not yet initialized, is
// used instead.
func New(cfg Config) (*Recognizer, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}

	meta, err := onnxmeta.ParseFile(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("ppocrv6: inspect model: %w", err)
	}
	if err := validateContract(meta, &cfg); err != nil {
		return nil, err
	}
	classes := int(meta.Outputs[0].Shape[2].Value)
	if classes <= 0 {
		return nil, fmt.Errorf("ppocrv6: model output class count %d is invalid", classes)
	}

	// The dictionary may be embedded: when DictPath is empty the built-in
	// ppocrv6_dict.txt is used.
	var dict *Dict
	if cfg.DictPath == "" {
		dict = DefaultDict()
	} else {
		dict, err = LoadDict(cfg.DictPath)
		if err != nil {
			return nil, err
		}
	}
	if err := dict.Validate(classes); err != nil {
		return nil, err
	}

	if err := initEnvironment(cfg.LibraryPath); err != nil {
		return nil, err
	}
	envMu.Lock()
	recCount++
	envMu.Unlock()

	// Placeholder tensors for session construction; real per-image tensors
	// are passed through RunWithValues on every Recognize call because the
	// width changes per image.
	inShape := ort.Shape{1, 3, int64(cfg.Height), 1}
	outShape := ort.Shape{1, 1, int64(classes)}
	placeholderIn, err := ort.NewTensor(inShape, make([]float32, 3*cfg.Height))
	if err != nil {
		return nil, fmt.Errorf("ppocrv6: create input placeholder: %w", err)
	}
	placeholderOut, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		placeholderIn.Destroy()
		return nil, fmt.Errorf("ppocrv6: create output placeholder: %w", err)
	}

	sess, err := ort.NewAdvancedSession(
		cfg.ModelPath,
		[]string{cfg.InputName},
		[]string{cfg.OutputName},
		[]ort.Value{placeholderIn},
		[]ort.Value{placeholderOut},
		nil,
	)
	if err != nil {
		placeholderIn.Destroy()
		placeholderOut.Destroy()
		return nil, fmt.Errorf("ppocrv6: create session: %w", err)
	}

	return &Recognizer{
		cfg:            cfg,
		dict:           dict,
		sess:           sess,
		placeholderIn:  placeholderIn,
		placeholderOut: placeholderOut,
		classes:        classes,
		outputLenMap:   make(map[int]int),
	}, nil
}

// validateContract checks the ONNX graph contract against the config and
// fills the input height from the model when Config.Height was left zero.
func validateContract(meta *onnxmeta.Model, cfg *Config) error {
	if len(meta.Inputs) != 1 || len(meta.Outputs) != 1 {
		return fmt.Errorf("ppocrv6: model has %d inputs and %d outputs, want 1/1",
			len(meta.Inputs), len(meta.Outputs))
	}
	in := meta.Inputs[0]
	if in.ElemType != onnxmeta.ElemTypeFloat32 {
		return fmt.Errorf("ppocrv6: input element type %d, want float32", in.ElemType)
	}
	if len(in.Shape) != 4 {
		return fmt.Errorf("ppocrv6: input shape %v, want [batch 3 height width]", in.Shape)
	}
	if in.Shape[1].Value != 3 {
		return fmt.Errorf("ppocrv6: input channels %d, want 3", in.Shape[1].Value)
	}
	if cfg.Height == DefaultHeight && !in.Shape[2].IsDynamic() && int(in.Shape[2].Value) != cfg.Height {
		return fmt.Errorf("ppocrv6: model height %d does not match config height %d", in.Shape[2].Value, cfg.Height)
	}
	if !in.Shape[2].IsDynamic() && cfg.Height != int(in.Shape[2].Value) {
		return fmt.Errorf("ppocrv6: model height %d does not match config height %d", in.Shape[2].Value, cfg.Height)
	}
	out := meta.Outputs[0]
	if len(out.Shape) != 3 || !out.Shape[1].IsDynamic() {
		return fmt.Errorf("ppocrv6: output shape %v, want [batch seq classes]", out.Shape)
	}
	return nil
}

// initEnvironment ensures the ONNX Runtime environment is initialized.
// pure-onnx reference-counts initialization internally, so every call pairs
// with the DestroyEnvironment in Close. Config.LibraryPath is honored only
// on the first initialization; afterwards the ORT logging level is pinned to
// fatal because the shape probe intentionally triggers ORT error logging.
func initEnvironment(libraryPath string) error {
	envMu.Lock()
	defer envMu.Unlock()
	if envInitSet && libraryPath != "" {
		return fmt.Errorf("ppocrv6: LibraryPath cannot change after the runtime is initialized")
	}
	envInitSet = true
	if libraryPath != "" {
		if err := ort.SetSharedLibraryPath(libraryPath); err != nil {
			return fmt.Errorf("ppocrv6: set library path: %w", err)
		}
		_ = ort.SetLogLevel(ort.LoggingLevelFatal)
		return ort.InitializeEnvironment()
	}
	return ort.InitializeEnvironmentWithBootstrap()
}

// Recognize recognizes the text line in img and returns the decoded text
// with an average confidence score.
func (r *Recognizer) Recognize(img image.Image) (Result, error) {
	input, width, err := Preprocess(img, &r.cfg)
	if err != nil {
		return Result{}, err
	}

	seqLen, err := r.outputLen(width)
	if err != nil {
		return Result{}, err
	}

	inShape := ort.Shape{1, 3, int64(r.cfg.Height), int64(width)}
	inTensor, err := ort.NewTensor(inShape, input)
	if err != nil {
		return Result{}, fmt.Errorf("ppocrv6: create input tensor: %w", err)
	}
	defer inTensor.Destroy()

	outShape := ort.Shape{1, int64(seqLen), int64(r.classes)}
	outTensor, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return Result{}, fmt.Errorf("ppocrv6: create output tensor: %w", err)
	}
	defer outTensor.Destroy()

	if err := r.sess.RunWithValues([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return Result{}, fmt.Errorf("ppocrv6: run inference: %w", err)
	}

	return Decode(outTensor.GetData(), seqLen, r.classes, r.dict), nil
}

// outputLen returns the CTC sequence length for a resized input width. The
// model output dim is dynamic, and ONNX Runtime requires the output tensor
// shape to match the computed shape exactly. The length is therefore probed
// once per width: a run with an intentionally oversized output fails with a
// shape-verification error that reports the exact computed shape, which is
// parsed and cached. This is model-agnostic and costs one failed inference
// per new width (a few milliseconds).
func (r *Recognizer) outputLen(width int) (int, error) {
	r.shapeMu.Lock()
	defer r.shapeMu.Unlock()
	if t, ok := r.outputLenMap[width]; ok {
		return t, nil
	}
	t, err := r.probeOutputLen(width)
	if err != nil {
		return 0, err
	}
	r.outputLenMap[width] = t
	return t, nil
}

var requestShapeRe = regexp.MustCompile(`Requested shape:\{[^,]+,(\d+),(\d+)\}`)

// probeOutputLen runs a zero input through the model with an oversized
// output tensor and extracts the computed output shape from the resulting
// shape-verification error.
func (r *Recognizer) probeOutputLen(width int) (int, error) {
	inShape := ort.Shape{1, 3, int64(r.cfg.Height), int64(width)}
	inTensor, err := ort.NewTensor(inShape, make([]float32, 3*r.cfg.Height*width))
	if err != nil {
		return 0, fmt.Errorf("ppocrv6: probe input tensor: %w", err)
	}
	defer inTensor.Destroy()

	outShape := ort.Shape{1, int64(width), int64(r.classes)}
	outTensor, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return 0, fmt.Errorf("ppocrv6: probe output tensor: %w", err)
	}
	defer outTensor.Destroy()

	err = r.sess.RunWithValues([]ort.Value{inTensor}, []ort.Value{outTensor})
	if err == nil {
		// The model accepted the oversized shape: the sequence length is
		// bounded by the width and trailing rows are blank.
		return width, nil
	}
	m := requestShapeRe.FindStringSubmatch(err.Error())
	if m == nil {
		return 0, fmt.Errorf("ppocrv6: shape probe failed, cannot determine output length: %w", err)
	}
	seqLen := atoiSafe(m[1])
	classes := atoiSafe(m[2])
	if seqLen <= 0 || classes != r.classes {
		return 0, fmt.Errorf("ppocrv6: shape probe returned unexpected shape [1 %s %s]: %w", m[1], m[2], err)
	}
	return seqLen, nil
}

func atoiSafe(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// Close releases the inference session and the runtime environment
// reference. The Recognizer must not be used afterwards. Close is idempotent.
func (r *Recognizer) Close() error {
	envMu.Lock()
	if r.closed {
		envMu.Unlock()
		return nil
	}
	r.closed = true
	recCount--
	destroyEnv := recCount == 0
	envMu.Unlock()

	var errs []error
	if r.placeholderIn != nil {
		if err := r.placeholderIn.Destroy(); err != nil {
			errs = append(errs, err)
		}
		r.placeholderIn = nil
	}
	if r.placeholderOut != nil {
		if err := r.placeholderOut.Destroy(); err != nil {
			errs = append(errs, err)
		}
		r.placeholderOut = nil
	}
	if r.sess != nil {
		if err := r.sess.Destroy(); err != nil {
			errs = append(errs, err)
		}
		r.sess = nil
	}
	if destroyEnv {
		if err := ort.DestroyEnvironment(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
