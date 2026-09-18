package ppocrv6

import (
	"path/filepath"
	"testing"
)

func TestLoadDict(t *testing.T) {
	d, err := LoadDict(filepath.Join("testdata", "ppocrv6_dict.txt"))
	if err != nil {
		t.Fatalf("LoadDict: %v", err)
	}
	if d.Len() != DefaultDictSize {
		t.Fatalf("dict length = %d, want %d", d.Len(), DefaultDictSize)
	}
	if d.ClassCount() != DefaultDictSize+2 {
		t.Fatalf("class count = %d, want %d", d.ClassCount(), DefaultDictSize+2)
	}
	// Spot checks on known positions from the real dictionary.
	cases := []struct {
		class int
		want  rune
		ok    bool
	}{
		{0, 0, false},                    // blank
		{1, '!', true},                   // first dict char
		{33, '0', true},                  // chars[32] = '0'
		{34, '1', true},                  // chars[33] = '1'
		{2129, '中', true},                // Chinese
		{DefaultDictSize, '🛅', true},     // last dict char
		{DefaultDictSize + 1, ' ', true}, // space
		{DefaultDictSize + 2, 0, false},  // out of range
	}
	for _, c := range cases {
		got, ok := d.At(c.class)
		if got != c.want || ok != c.ok {
			t.Errorf("At(%d) = %q,%v want %q,%v", c.class, got, ok, c.want, c.ok)
		}
	}
}

func TestDictValidate(t *testing.T) {
	d := NewDict([]rune{'a', 'b', 'c'})
	if err := d.Validate(5); err != nil {
		t.Errorf("Validate(5) failed: %v", err)
	}
	if err := d.Validate(4); err == nil {
		t.Error("Validate(4) should fail (3 chars + blank + space = 5)")
	}
}
