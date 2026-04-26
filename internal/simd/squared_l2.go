// Package simd contains small vector math kernels with portable fallbacks.
package simd

// SquaredL2 returns the squared Euclidean distance between equal-length vectors.
func SquaredL2(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("simd: length mismatch")
	}
	return squaredL2(a, b)
}
