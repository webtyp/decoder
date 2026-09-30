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

// mulVec computes dst = W·x (dst has rows entries, x has cols).
func (m matrix) mulVec(dst, x []float32) {
	if m.q != nil {
		_ = nn.MatVecInt8Block32(dst, x, m.q, m.scales, m.rows, m.cols)
	} else {
		_ = nn.MatmulT(dst, x, m.f32, 1, m.cols, m.rows)
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
