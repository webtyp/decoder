package decoder

import (
	"webtyp.com/fmt"
	"webtyp.com/nn"
	"webtyp.com/weights"
)

// matrix is a rows×cols weight matrix in the storage its tensor had: float32, or int8 in
// blocks of 32 with one scale per block. It is never converted on load.
type matrix struct {
	rows, cols int
	f32        []float32 // Float32 storage, rows×cols
	q          []byte    // Int8Block32 storage, rows×cols int8 values
	scales     []float32 // Int8Block32 scales, rows×ceil(cols/32)
}

// quantBuf holds an input vector quantized to int8 blocks, for the int8 × int8 kernel.
type quantBuf struct {
	xq []int8
	xs []float32
}

// mulVec computes dst = W·x (dst has rows entries, x has cols). An int8 matrix quantizes x into
// qb and runs the integer kernel nn.MatVecQ8Block32, which WebAssembly vectorizes (2.1× without
// SIMD, 4.3× with it; nn/docs/PERFORMANCE.md). A matrix whose width is not a multiple of 32 uses
// the float kernel.
func (m matrix) mulVec(dst, x []float32, qb *quantBuf) {
	switch {
	case m.q != nil && m.cols%nn.Int8BlockSize == 0:
		_ = nn.QuantizeBlocks32(qb.xq[:m.cols], qb.xs[:m.cols/nn.Int8BlockSize], x[:m.cols])
		_ = nn.MatVecQ8Block32(dst, qb.xq, qb.xs, m.q, m.scales, m.rows, m.cols)
	case m.q != nil:
		_ = nn.MatVecInt8Block32(dst, x, m.q, m.scales, m.rows, m.cols)
	default:
		_ = nn.MatmulT(dst, x, m.f32, 1, m.cols, m.rows)
	}
}

// rowsDot writes into out[k] the dot product of row rows[k] of W with x: the logits of chosen
// tokens only, without the whole output projection.
func (m matrix) rowsDot(out []float32, rows []int, x []float32, tmp []float32) {
	for k, r := range rows {
		m.row(tmp[:m.cols], r)
		var sum float32
		for c := 0; c < m.cols; c++ {
			sum += tmp[c] * x[c]
		}
		out[k] = sum
	}
}

// row writes row i of W into dst as float32 (the embedding lookup of a token).
func (m matrix) row(dst []float32, i int) {
	if m.q != nil {
		blocksPerRow := (m.cols + weights.BlockSize - 1) / weights.BlockSize
		qRow := m.q[i*m.cols : (i+1)*m.cols]
		scaleRow := m.scales[i*blocksPerRow : (i+1)*blocksPerRow]

		for b := 0; b < blocksPerRow; b++ {
			scale := scaleRow[b]
			start := b * weights.BlockSize
			end := start + weights.BlockSize
			if end > m.cols {
				end = m.cols
			}
			for col := start; col < end; col++ {
				dst[col] = float32(int8(qRow[col])) * scale
			}
		}
	} else {
		copy(dst, m.f32[i*m.cols:(i+1)*m.cols])
	}
}

func loadMatrix(a *weights.Artifact, name string, rows, cols int) (matrix, error) {
	t, ok := a.Tensor(name)
	if !ok {
		return matrix{}, MissingTensorError(name)
	}

	expectedLen := rows * cols
	switch t.DType {
	case weights.Float32:
		data, err := t.Float32s()
		if err != nil {
			return matrix{}, err
		}
		if len(data) != expectedLen {
			return matrix{}, WrongTensorSizeError(name, len(data), expectedLen)
		}
		return matrix{
			rows: rows,
			cols: cols,
			f32:  data,
		}, nil

	case weights.Int8Block32:
		if len(t.Data) != expectedLen {
			return matrix{}, WrongTensorSizeError(name, len(t.Data), expectedLen)
		}
		blocksPerRow := (cols + weights.BlockSize - 1) / weights.BlockSize
		expectedScales := rows * blocksPerRow
		if len(t.Scales) != expectedScales {
			return matrix{}, fmt.ErrType(makeErr(fmt.Sprintf("%s: %s", name, weights.ErrScalesMismatch.Error())), weights.ErrScalesMismatch)
		}
		return matrix{
			rows:   rows,
			cols:   cols,
			q:      t.Data,
			scales: t.Scales,
		}, nil

	default:
		return matrix{}, UnsupportedDTypeError(name, string(t.DType))
	}
}
