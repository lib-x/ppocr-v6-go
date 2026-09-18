package ppocrv6

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
)

// Preprocess resizes img to the model input [1, 3, Height, width] (NCHW,
// float32) and normalizes pixels with (pixel*Scale - Mean) / Std.
//
// The width is derived proportionally from the source aspect ratio so text
// is not distorted: width = ceil(Height * srcW / srcH), capped by
// Config.MaxWidth when set.
//
// The returned slice holds Height*width*3 float32 values in CHW order and
// the returned width is the actual resized width.
func Preprocess(img image.Image, cfg *Config) ([]float32, int, error) {
	h := cfg.Height
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	if srcW <= 0 || srcH <= 0 {
		return nil, 0, fmt.Errorf("ppocrv6: empty image %dx%d", srcW, srcH)
	}

	w := int(math.Ceil(float64(h) * float64(srcW) / float64(srcH)))
	if w < 1 {
		w = 1
	}
	if cfg.MaxWidth > 0 && w > cfg.MaxWidth {
		w = cfg.MaxWidth
	}

	rgba := toRGBA(img)
	rows := resizeRows(rgba, srcW, srcH, w, h)

	// NCHW output with normalization.
	out := make([]float32, 3*h*w)
	scale, mean, std := cfg.Scale, cfg.Mean, cfg.Std
	for y := 0; y < h; y++ {
		row := rows[y]
		for x := 0; x < w; x++ {
			px := row[x]
			for c := 0; c < 3; c++ {
				var ch uint8
				if cfg.RGB {
					switch c {
					case 0:
						ch = px.R
					case 1:
						ch = px.G
					default:
						ch = px.B
					}
				} else {
					switch c {
					case 0:
						ch = px.B
					case 1:
						ch = px.G
					default:
						ch = px.R
					}
				}
				out[c*h*w+y*w+x] = (float32(ch)*scale - mean) / std
			}
		}
	}
	return out, w, nil
}

// toRGBA normalizes any image.Image into an RGBA buffer for fast row access.
func toRGBA(img image.Image) *image.RGBA {
	if r, ok := img.(*image.RGBA); ok {
		return r
	}
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	return dst
}

// resizeRows resizes the RGBA image to (w, h) with bilinear interpolation
// and returns one []color.RGBA row per output row.
func resizeRows(src *image.RGBA, srcW, srcH, w, h int) [][]color.RGBA {
	rows := make([][]color.RGBA, h)
	// Scale factors in source pixels per destination pixel.
	sx := float64(srcW) / float64(w)
	sy := float64(srcH) / float64(h)
	// Clamp helper for the 2x2 sampling window.
	clampX := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= srcW {
			return srcW - 1
		}
		return v
	}
	clampY := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= srcH {
			return srcH - 1
		}
		return v
	}

	for dy := 0; dy < h; dy++ {
		// Continuous source y coordinate and its fractional part.
		y := float64(dy)*sy + 0.5*sy - 0.5
		y0 := clampY(int(math.Floor(y)))
		y1 := clampY(y0 + 1)
		fy := float32(y - math.Floor(y))
		if fy < 0 {
			fy = 0
		}

		row := make([]color.RGBA, w)
		top := src.PixOffset(0, y0)
		bot := src.PixOffset(0, y1)
		for dx := 0; dx < w; dx++ {
			x := float64(dx)*sx + 0.5*sx - 0.5
			x0 := clampX(int(math.Floor(x)))
			x1 := clampX(x0 + 1)
			fx := float32(x - math.Floor(x))
			if fx < 0 {
				fx = 0
			}

			// Bilinear blend of the four source pixels, channel by channel.
			w00 := (1 - fx) * (1 - fy)
			w10 := fx * (1 - fy)
			w01 := (1 - fx) * fy
			w11 := fx * fy
			var r, g, b, a float32
			blend := func(off int, wt float32) {
				r += float32(src.Pix[off]) * wt
				g += float32(src.Pix[off+1]) * wt
				b += float32(src.Pix[off+2]) * wt
				a += float32(src.Pix[off+3]) * wt
			}
			blend(top+x0*4, w00)
			blend(top+x1*4, w10)
			blend(bot+x0*4, w01)
			blend(bot+x1*4, w11)
			row[dx] = color.RGBA{
				R: uint8(r + 0.5),
				G: uint8(g + 0.5),
				B: uint8(b + 0.5),
				A: uint8(a + 0.5),
			}
		}
		rows[dy] = row
	}
	return rows
}

// NormalizeAlreadyResized converts an already-resized RGBA image at (w, h)
// into the model input tensor [1, 3, h, w] float32 (NCHW), applying
// (pixel*Scale - Mean) / Std with the configured channel order. Unlike
// [Preprocess], no resizing is performed.
func NormalizeAlreadyResized(rgba *image.RGBA, w, h int, cfg *Config) []float32 {
	out := make([]float32, 3*h*w)
	scale, mean, std := cfg.Scale, cfg.Mean, cfg.Std
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*rgba.Stride + x*4
			for c := 0; c < 3; c++ {
				var ch uint8
				if cfg.RGB {
					ch = rgba.Pix[i+c]
				} else {
					ch = rgba.Pix[i+2-c]
				}
				out[c*h*w+y*w+x] = (float32(ch)*scale - mean) / std
			}
		}
	}
	return out
}
