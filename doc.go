// Package ppocrv6 wraps the PP-OCRv6 small recognition model (ONNX) for
// text-line recognition.
//
// The package performs the complete recognition pipeline in Go:
//
//  1. Preprocess: decode the input image, proportionally resize it to the
//     model's fixed height (48px), and normalize pixel values with the
//     model-verified parameters.
//  2. Inference: run the PP-OCRv6 CTC head through pure-onnx
//     (github.com/amikos-tech/pure-onnx), a purego binding of ONNX Runtime
//     that requires no CGo toolchain.
//  3. Postprocess: greedy CTC decoding against the PP-OCRv6 character
//     dictionary, producing the recognized text and an average confidence.
//
// Model contract (verified against inference.onnx from
// PaddlePaddle/PP-OCRv6_small_rec_onnx on ModelScope):
//
//	input  x            : float32 [batch, 3, 48, width], NCHW, dynamic width
//	output fetch_name_0 : float32 [batch, seq_len, 18710]
//
// The 18710 output classes are the 18708-entry dictionary plus the CTC blank
// (index 0) and the space character (index 18709).
package ppocrv6
