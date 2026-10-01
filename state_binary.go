package decoder

import "math"

// stateMagic opens every saved State; stateFormat is its layout version.
const (
	stateMagic  = "WTST"
	stateFormat = 1
)

// MarshalBinary saves the sequence state (position, KV caches, recurrent and convolution states)
// as bytes, so a runtime can keep a prompt's fixed prefix across sessions, for example in the
// browser's OPFS, instead of reading it again. Scratch buffers are not saved.
//
// Layout, little-endian: "WTST", format, position, layer count, then per layer the length of each
// slice (KCache, VCache, ConvState, RecState) as uint32 followed by its float32 values.
func (s *State) MarshalBinary() ([]byte, error) {
	size := len(stateMagic) + 3*4
	for i := range s.Layers {
		l := &s.Layers[i]
		size += 4*4 + 4*(len(l.KCache)+len(l.VCache)+len(l.ConvState)+len(l.RecState))
	}
	out := make([]byte, 0, size)
	out = append(out, stateMagic...)
	out = putUint32(out, stateFormat)
	out = putUint32(out, uint32(s.Pos))
	out = putUint32(out, uint32(len(s.Layers)))
	for i := range s.Layers {
		l := &s.Layers[i]
		for _, v := range [][]float32{l.KCache, l.VCache, l.ConvState, l.RecState} {
			out = putUint32(out, uint32(len(v)))
			for _, f := range v {
				out = putUint32(out, math.Float32bits(f))
			}
		}
	}
	return out, nil
}

// UnmarshalBinary restores a state saved by MarshalBinary into s, which must come from a Model of
// the same shape (NewState). Bytes that are not a saved state give ErrStateFormat; a state of
// another shape gives ErrStateMismatch. On error s is left unchanged.
func (s *State) UnmarshalBinary(data []byte) error {
	r := stateReader{data: data}
	if len(data) < len(stateMagic) || string(data[:len(stateMagic)]) != stateMagic {
		return ErrStateFormat
	}
	r.pos = len(stateMagic)
	format, pos, layers := r.uint32(), r.uint32(), r.uint32()
	if r.bad || format != stateFormat {
		return ErrStateFormat
	}
	if int(layers) != len(s.Layers) {
		return ErrStateMismatch
	}
	// Read everything first, so a failure leaves s as it was.
	read := make([][4][]float32, len(s.Layers))
	for i := range s.Layers {
		for j := 0; j < 4; j++ {
			n := int(r.uint32())
			if r.bad || n > (len(data)-r.pos)/4 {
				return ErrStateFormat
			}
			v := make([]float32, n)
			for k := range v {
				v[k] = math.Float32frombits(r.uint32())
			}
			read[i][j] = v
		}
		l := &s.Layers[i]
		k, v, conv, rec := read[i][0], read[i][1], read[i][2], read[i][3]
		if len(conv) != len(l.ConvState) || len(rec) != len(l.RecState) || len(k) != len(v) {
			return ErrStateMismatch
		}
	}
	if r.bad || r.pos != len(data) {
		return ErrStateFormat
	}
	s.Pos = int(pos)
	for i := range s.Layers {
		l := &s.Layers[i]
		l.KCache = append(l.KCache[:0], read[i][0]...)
		l.VCache = append(l.VCache[:0], read[i][1]...)
		copy(l.ConvState, read[i][2])
		copy(l.RecState, read[i][3])
	}
	return nil
}

func putUint32(b []byte, v uint32) []byte {
	return append(b, byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}

type stateReader struct {
	data []byte
	pos  int
	bad  bool
}

func (r *stateReader) uint32() uint32 {
	if r.pos+4 > len(r.data) {
		r.bad = true
		return 0
	}
	b := r.data[r.pos:]
	r.pos += 4
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}
