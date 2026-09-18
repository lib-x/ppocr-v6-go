package ppocrv6

import (
	"bufio"
	_ "embed"
	"fmt"
	"os"
	"slices"
	"strings"
)

// DefaultDictBytes is the embedded PP-OCRv6 character dictionary
// (ppocrv6_dict.txt, Apache 2.0, from PaddlePaddle/PaddleOCR). It is used
// automatically when Config.DictPath is empty, so callers only need the ONNX
// model file.
//
//go:embed testdata/ppocrv6_dict.txt
var DefaultDictBytes []byte

// DefaultDict returns a Dict loaded from the embedded dictionary.
func DefaultDict() *Dict {
	d, err := LoadDictBytes(DefaultDictBytes)
	if err != nil {
		// The embedded dictionary is validated by tests; a failure here means
		// the binary was built with a corrupted embed.
		panic(fmt.Sprintf("ppocrv6: embedded dict: %v", err))
	}
	return d
}

// LoadDictBytes parses a dictionary from raw bytes (one character per line).
func LoadDictBytes(data []byte) (*Dict, error) {
	var chars []rune
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		chars = append(chars, []rune(line)[0])
	}
	if len(chars) == 0 {
		return nil, fmt.Errorf("ppocrv6: dict is empty")
	}
	return &Dict{chars: chars}, nil
}

// Dict is the PP-OCRv6 character dictionary with the PaddleOCR CTC class
// layout: class 0 is the CTC blank, classes 1..Len() map to
// chars[0..Len()-1], and the final class maps to a space character
// (use_space_char=true).
type Dict struct {
	chars []rune
}

// LoadDict reads a dictionary file with one character per line. Blank lines
// are skipped; the final line does not need a trailing newline.
func LoadDict(path string) (*Dict, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("ppocrv6: open dict %s: %w", path, err)
	}
	defer f.Close()

	var chars []rune
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r")
		if line == "" {
			continue
		}
		// The dictionary is one UTF-8 character per line; take the first
		// rune so surrogate/combining tails cannot corrupt the mapping.
		chars = append(chars, []rune(line)[0])
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("ppocrv6: read dict %s: %w", path, err)
	}
	if len(chars) == 0 {
		return nil, fmt.Errorf("ppocrv6: dict %s is empty", path)
	}
	return &Dict{chars: chars}, nil
}

// NewDict builds a Dict from a character slice (for tests and custom dicts).
func NewDict(chars []rune) *Dict {
	return &Dict{chars: slices.Clone(chars)}
}

// Len returns the number of dictionary characters (excluding blank/space).
func (d *Dict) Len() int { return len(d.chars) }

// ClassCount returns the full model output class count:
// 1 (blank) + Len() + 1 (space).
func (d *Dict) ClassCount() int { return len(d.chars) + 2 }

// SpaceClass returns the output class index of the space character.
func (d *Dict) SpaceClass() int { return len(d.chars) + 1 }

// At decodes one output class index into its character. It reports false for
// the CTC blank class and true otherwise.
func (d *Dict) At(class int) (rune, bool) {
	switch {
	case class <= 0:
		return 0, false
	case class <= len(d.chars):
		return d.chars[class-1], true
	case class == d.SpaceClass():
		return ' ', true
	default:
		return 0, false
	}
}

// Validate checks that the dictionary matches the model output width
// (classes == blank + chars + space).
func (d *Dict) Validate(modelClasses int) error {
	if d.ClassCount() != modelClasses {
		return fmt.Errorf(
			"ppocrv6: dict size %d + blank + space = %d classes, but model outputs %d",
			d.Len(), d.ClassCount(), modelClasses,
		)
	}
	return nil
}
