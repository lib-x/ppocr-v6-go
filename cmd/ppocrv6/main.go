// Command ppocrv6 recognizes text in an image with the PP-OCRv6 models.
//
// Single-line recognition:
//
//	ppocrv6 -model inference.onnx line.png
//
// Full-page OCR (detection + recognition):
//
//	ppocrv6 -model inference.onnx -det det.onnx page.png
//
// The ONNX Runtime shared library is resolved via ONNXRUNTIME_LIB_PATH or
// downloaded by pure-onnx bootstrap. The recognition dictionary is embedded;
// pass -dict to override it.
package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"

	"github.com/lib-x/ppocr-v6-go"
)

func main() {
	modelPath := flag.String("model", "", "path to the PP-OCRv6 recognition inference.onnx")
	detPath := flag.String("det", "", "path to the PP-OCRv6 detection inference.onnx (enables full-page OCR)")
	dictPath := flag.String("dict", "", "path to ppocrv6_dict.txt (default: embedded dictionary)")
	flag.Parse()
	if *modelPath == "" || flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatalf("open image: %v", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatalf("decode image: %v", err)
	}

	if *detPath != "" {
		p, err := ppocrv6.NewPipeline(
			ppocrv6.Config{ModelPath: *modelPath, DictPath: *dictPath},
			ppocrv6.DetConfig{ModelPath: *detPath},
		)
		if err != nil {
			log.Fatalf("new pipeline: %v", err)
		}
		defer p.Close()

		results, err := p.Recognize(img)
		if err != nil {
			log.Fatalf("recognize: %v", err)
		}
		for _, r := range results {
			fmt.Printf("%s\t%.4f\n", r.Text, r.Score)
		}
		return
	}

	rec, err := ppocrv6.New(ppocrv6.Config{ModelPath: *modelPath, DictPath: *dictPath})
	if err != nil {
		log.Fatalf("new recognizer: %v", err)
	}
	defer rec.Close()

	res, err := rec.Recognize(img)
	if err != nil {
		log.Fatalf("recognize: %v", err)
	}
	fmt.Printf("%s\t%.4f\n", res.Text, res.Score)
}
