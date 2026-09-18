package ppocrv6

import "math"

// Result is the outcome of recognizing one text line.
type Result struct {
	// Text is the decoded text.
	Text string
	// Score is the mean softmax probability of the emitted (non-blank)
	// classes, matching PaddleOCR's reported confidence.
	Score float32
}

// Decode applies greedy CTC decoding to the raw model logits.
//
// logits is row-major with seqLen*classCount float32 values. The class
// layout follows [Dict]: 0 = blank, 1..Len() = dictionary characters,
// Len()+1 = space.
//
// Greedy CTC rules:
//   - take the argmax class per timestep;
//   - collapse consecutive repeats of the same class;
//   - drop blank classes (blanks also reset the repeat memory, so
//     "aa _ aa" stays "aa");
//   - map surviving classes through the dictionary.
func Decode(logits []float32, seqLen, classCount int, d *Dict) Result {
	if seqLen < 1 || classCount < 1 {
		return Result{}
	}
	if len(logits) < seqLen*classCount {
		seqLen = len(logits) / classCount
	}
	if seqLen < 1 {
		return Result{}
	}

	var sb []rune
	var scoreSum float32
	scoreCount := 0
	prev := -1

	for i := 0; i < seqLen; i++ {
		row := logits[i*classCount : (i+1)*classCount]
		probs := row
		if !looksLikeProbabilities(row) {
			// Some exports end before the softmax; normalize those rows.
			probs = softmax(row)
		}
		best, bestP := argmax(probs)
		if best == 0 { // blank: reset repeat memory
			prev = -1
			continue
		}
		if best == prev { // repeated class: collapse
			continue
		}
		prev = best
		ch, ok := d.At(best)
		if !ok {
			continue // out-of-range class (e.g. trailing zero padding)
		}
		sb = append(sb, ch)
		scoreSum += bestP
		scoreCount++
	}

	res := Result{Text: string(sb)}
	if scoreCount > 0 {
		res.Score = scoreSum / float32(scoreCount)
	}
	return res
}

// argmax returns the index and value of the maximum element.
func argmax(row []float32) (int, float32) {
	best := 0
	bestP := float32(math.Inf(-1))
	for i, v := range row {
		if v > bestP {
			bestP = v
			best = i
		}
	}
	return best, bestP
}

// looksLikeProbabilities reports whether a model output row is already a
// probability distribution (values in [0,1] summing to ~1). The PP-OCRv6
// ONNX export ends with a Softmax node, so its output is post-softmax; other
// exports stop at the logits and need normalization.
func looksLikeProbabilities(row []float32) bool {
	var sum float32
	for _, v := range row {
		if v < 0 || v > 1 {
			return false
		}
		sum += v
	}
	return sum > 0.99 && sum < 1.01
}

// softmax converts raw logits into per-class probabilities in place,
// subtracting the row maximum for numerical stability.
func softmax(row []float32) []float32 {
	var max float32 = float32(math.Inf(-1))
	for _, v := range row {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range row {
		row[i] = float32(math.Exp(float64(v - max)))
		sum += row[i]
	}
	for i := range row {
		row[i] /= sum
	}
	return row
}
