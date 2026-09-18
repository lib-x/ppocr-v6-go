package ppocrv6

import (
	"math"
	"testing"
)

// syntheticLogits builds logits where class 'want' wins at every step with a
// fixed margin, plus optional per-step overrides.
func syntheticLogits(seqLen, classCount, margin int, want []int) []float32 {
	logits := make([]float32, seqLen*classCount)
	base := float32(0.01)
	for i := 0; i < seqLen; i++ {
		step := i
		if step >= len(want) {
			step = len(want) - 1
		}
		best := want[step]
		for c := 0; c < classCount; c++ {
			if c == best {
				logits[i*classCount+c] = 1.0 - base*float32(margin)
			} else {
				logits[i*classCount+c] = base
			}
		}
	}
	return logits
}

func TestDecodeBasic(t *testing.T) {
	d := NewDict([]rune{'a', 'b', 'c', 'd'}) // classes: 0 blank, 1..4, 5 space
	// "ab": a a _ b b — the blank between resets the repeat memory.
	logits := syntheticLogits(5, 6, 2, []int{1, 1, 0, 2, 2})
	res := Decode(logits, 5, 6, d)
	if res.Text != "ab" {
		t.Errorf("Text = %q, want %q", res.Text, "ab")
	}
	// softmax of (0.98 vs 0.01x5) ≈ 0.345 per emitted row.
	if math.Abs(float64(res.Score)-0.345) > 0.01 {
		t.Errorf("Score = %v, want ~0.345", res.Score)
	}
}

func TestDecodeSpaceAndChinese(t *testing.T) {
	d := NewDict([]rune{'a', '中', 'c'}) // space class = 4
	// Steps: a 中 _ 中 _ (space) a  →  "a中中 a"
	logits := syntheticLogits(6, 5, 2, []int{1, 2, 0, 2, 4, 1})
	res := Decode(logits, 6, 5, d)
	if res.Text != "a中中 a" {
		t.Errorf("Text = %q, want %q", res.Text, "a中中 a")
	}
}

func TestDecodeRepeatCollapseAcrossBlank(t *testing.T) {
	// "aa _ aa" must stay "aa": repeat memory resets at blank.
	d := NewDict([]rune{'a'}) // classes 0 blank, 1 a, 2 space
	logits := syntheticLogits(6, 3, 2, []int{1, 1, 0, 1, 1, 2})
	res := Decode(logits, 6, 3, d)
	if res.Text != "aa " {
		t.Errorf("Text = %q, want %q", res.Text, "aa ")
	}
}

func TestDecodeZeroTail(t *testing.T) {
	// All-zero rows (padding beyond the real sequence) must decode to
	// nothing: argmax of a zero row is class 0 (blank).
	d := NewDict([]rune{'a', 'b'})
	logits := syntheticLogits(3, 4, 2, []int{1, 2, 0})
	res := Decode(logits, 8, 4, d) // 5 extra zero rows
	if res.Text != "ab" {
		t.Errorf("Text = %q, want %q", res.Text, "ab")
	}
}

func TestDecodeEmpty(t *testing.T) {
	d := NewDict([]rune{'a'})
	res := Decode(nil, 0, 3, d)
	if res.Text != "" || res.Score != 0 {
		t.Errorf("empty decode = %+v, want empty", res)
	}
	res = Decode(make([]float32, 12), 4, 3, d) // all zeros
	if res.Text != "" || res.Score != 0 {
		t.Errorf("zero decode = %+v, want empty", res)
	}
}

func TestDecodeOutOfRangeClass(t *testing.T) {
	// A class index beyond the dict must be skipped, not panic.
	d := NewDict([]rune{'a'})
	logits := make([]float32, 2*5)
	logits[5+4] = 99 // row 1, class 4 (invalid)
	res := Decode(logits, 2, 5, d)
	if res.Text != "" {
		t.Errorf("Text = %q, want empty", res.Text)
	}
}
