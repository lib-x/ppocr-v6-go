// Package onnxmeta reads the model contract (graph input/output names and
// shapes) directly from an ONNX protobuf file.
//
// Only the subset of the ONNX schema needed for inference wiring is decoded;
// everything else is skipped via wire-format field walking. Keeping this
// dependency-free avoids pulling a full protobuf runtime into the library.
package onnxmeta

import (
	"errors"
	"fmt"
	"io"
	"os"
)

// TensorElemType mirrors the ONNX TensorProto.DataType enum values used here.
const (
	ElemTypeFloat32 = 1
)

// Dim is a single tensor dimension: a concrete value or a symbolic parameter
// for dynamic dims (e.g. "W" or "DynamicDimension.1").
type Dim struct {
	Value int64  // concrete size; -1 when symbolic
	Param string // symbolic name; empty when concrete
}

// IsDynamic reports whether the dimension is symbolic.
func (d Dim) IsDynamic() bool { return d.Param != "" }

// TensorInfo describes one graph value: name, element type and shape.
type TensorInfo struct {
	Name     string
	ElemType int64
	Shape    []Dim
}

// Model holds the graph contract of an ONNX model.
type Model struct {
	// IRVersion is the ONNX IR version (ModelProto.ir_version).
	IRVersion int64
	Inputs    []TensorInfo
	Outputs   []TensorInfo
}

// Field numbers from onnx.proto3 (subset used here).
const (
	modelGraphField     = 7
	graphInputField     = 11
	graphOutputField    = 12
	valueInfoNameField  = 1
	valueInfoTypeField  = 2
	typeTensorField     = 1
	tensorElemTypeField = 1
	tensorShapeField    = 2
	shapeDimField       = 1
	dimValueField       = 1
	dimParamField       = 2
)

// ParseFile loads and inspects the model at path.
func ParseFile(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("onnxmeta: read %s: %w", path, err)
	}
	return Parse(data)
}

// Parse inspects the model contract from the raw ONNX protobuf bytes.
func Parse(data []byte) (*Model, error) {
	m := &Model{}
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return nil, err
		}
		switch field {
		case 1: // ir_version
			v, err := p.varint()
			if err != nil {
				return nil, err
			}
			m.IRVersion = v
		case modelGraphField:
			if wt != 2 {
				return nil, fmt.Errorf("onnxmeta: graph field has wire type %d", wt)
			}
			graph, err := p.bytes()
			if err != nil {
				return nil, err
			}
			if err := m.parseGraph(graph); err != nil {
				return nil, err
			}
		default:
			if err := p.skip(wt); err != nil {
				return nil, err
			}
		}
	}
	if len(m.Inputs) == 0 || len(m.Outputs) == 0 {
		return nil, errors.New("onnxmeta: model has no inputs or outputs")
	}
	return m, nil
}

func (m *Model) parseGraph(data []byte) error {
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return err
		}
		switch field {
		case graphInputField, graphOutputField:
			if wt != 2 {
				return fmt.Errorf("onnxmeta: value info field has wire type %d", wt)
			}
			vi, err := p.bytes()
			if err != nil {
				return err
			}
			info, err := parseValueInfo(vi)
			if err != nil {
				return err
			}
			if field == graphInputField {
				m.Inputs = append(m.Inputs, info)
			} else {
				m.Outputs = append(m.Outputs, info)
			}
		default:
			if err := p.skip(wt); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseValueInfo(data []byte) (TensorInfo, error) {
	info := TensorInfo{}
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return info, err
		}
		switch field {
		case valueInfoNameField:
			if wt != 2 {
				return info, fmt.Errorf("onnxmeta: name field has wire type %d", wt)
			}
			b, err := p.bytes()
			if err != nil {
				return info, err
			}
			info.Name = string(b)
		case valueInfoTypeField:
			if wt != 2 {
				return info, fmt.Errorf("onnxmeta: type field has wire type %d", wt)
			}
			typ, err := p.bytes()
			if err != nil {
				return info, err
			}
			if err := parseType(typ, &info); err != nil {
				return info, err
			}
		default:
			if err := p.skip(wt); err != nil {
				return info, err
			}
		}
	}
	if info.Name == "" {
		return info, errors.New("onnxmeta: unnamed value info")
	}
	return info, nil
}

func parseType(data []byte, info *TensorInfo) error {
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return err
		}
		switch field {
		case typeTensorField:
			if wt != 2 {
				return fmt.Errorf("onnxmeta: tensor_type field has wire type %d", wt)
			}
			t, err := p.bytes()
			if err != nil {
				return err
			}
			if err := parseTensorType(t, info); err != nil {
				return err
			}
		default:
			if err := p.skip(wt); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseTensorType(data []byte, info *TensorInfo) error {
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return err
		}
		switch field {
		case tensorElemTypeField:
			v, err := p.varint()
			if err != nil {
				return err
			}
			info.ElemType = v
		case tensorShapeField:
			if wt != 2 {
				return fmt.Errorf("onnxmeta: shape field has wire type %d", wt)
			}
			shape, err := p.bytes()
			if err != nil {
				return err
			}
			dims, err := parseShape(shape)
			if err != nil {
				return err
			}
			info.Shape = dims
		default:
			if err := p.skip(wt); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseShape(data []byte) ([]Dim, error) {
	var dims []Dim
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return nil, err
		}
		if field == shapeDimField && wt == 2 {
			dim, err := p.bytes()
			if err != nil {
				return nil, err
			}
			d, err := parseDim(dim)
			if err != nil {
				return nil, err
			}
			dims = append(dims, d)
			continue
		}
		if err := p.skip(wt); err != nil {
			return nil, err
		}
	}
	if len(dims) == 0 {
		return nil, errors.New("onnxmeta: tensor shape has no dims")
	}
	return dims, nil
}

func parseDim(data []byte) (Dim, error) {
	d := Dim{Value: -1}
	p := wireReader{data: data}
	for !p.done() {
		field, wt, err := p.tag()
		if err != nil {
			return d, err
		}
		switch field {
		case dimValueField:
			if wt != 0 {
				return d, fmt.Errorf("onnxmeta: dim_value has wire type %d", wt)
			}
			v, err := p.varint()
			if err != nil {
				return d, err
			}
			d.Value = v
		case dimParamField:
			if wt != 2 {
				return d, fmt.Errorf("onnxmeta: dim_param has wire type %d", wt)
			}
			b, err := p.bytes()
			if err != nil {
				return d, err
			}
			d.Param = string(b)
		default:
			if err := p.skip(wt); err != nil {
				return d, err
			}
		}
	}
	return d, nil
}

// wireReader is a minimal protobuf wire-format reader.
type wireReader struct {
	data []byte
	off  int
}

func (r *wireReader) done() bool { return r.off >= len(r.data) }

func (r *wireReader) tag() (field int, wt int, err error) {
	v, err := r.varint()
	if err != nil {
		return 0, 0, err
	}
	field = int(v >> 3)
	wt = int(v & 7)
	if field == 0 {
		return 0, 0, errors.New("onnxmeta: invalid field number 0")
	}
	return field, wt, nil
}

func (r *wireReader) varint() (int64, error) {
	var v uint64
	var shift uint
	for {
		if r.off >= len(r.data) {
			return 0, io.ErrUnexpectedEOF
		}
		b := r.data[r.off]
		r.off++
		v |= uint64(b&0x7f) << shift
		if b < 0x80 {
			return int64(v), nil
		}
		shift += 7
		if shift >= 64 {
			return 0, errors.New("onnxmeta: varint overflow")
		}
	}
}

func (r *wireReader) bytes() ([]byte, error) {
	n, err := r.varint()
	if err != nil {
		return nil, err
	}
	if n < 0 || r.off+int(n) > len(r.data) {
		return nil, io.ErrUnexpectedEOF
	}
	b := r.data[r.off : r.off+int(n)]
	r.off += int(n)
	return b, nil
}

func (r *wireReader) skip(wt int) error {
	switch wt {
	case 0:
		_, err := r.varint()
		return err
	case 1:
		if r.off+8 > len(r.data) {
			return io.ErrUnexpectedEOF
		}
		r.off += 8
		return nil
	case 2:
		_, err := r.bytes()
		return err
	case 5:
		if r.off+4 > len(r.data) {
			return io.ErrUnexpectedEOF
		}
		r.off += 4
		return nil
	default:
		return fmt.Errorf("onnxmeta: unsupported wire type %d", wt)
	}
}
