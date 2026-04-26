package simd

import (
	"math"
	"testing"
)

func TestSquaredL2(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		{name: "empty", a: nil, b: nil, want: 0},
		{name: "single", a: []float32{3}, b: []float32{1}, want: 4},
		{name: "tail", a: []float32{1, 2, 3}, b: []float32{0, 2, 5}, want: 5},
		{name: "vector and tail", a: []float32{1, 2, 3, 4, 5}, b: []float32{5, 4, 3, 2, 1}, want: 40},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SquaredL2(tt.a, tt.b)
			if math.Abs(float64(got-tt.want)) > 1e-6 {
				t.Fatalf("SquaredL2 = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSquaredL2PanicsOnLengthMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	SquaredL2([]float32{1}, []float32{1, 2})
}

func BenchmarkSquaredL2(b *testing.B) {
	a := make([]float32, 512)
	c := make([]float32, 512)
	for i := range a {
		a[i] = float32(i) / 512
		c[i] = float32(512-i) / 512
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = SquaredL2(a, c)
	}
}
