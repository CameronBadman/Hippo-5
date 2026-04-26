package vectorindex

import (
	"math"
	"reflect"
	"testing"
)

func TestDimensionIndexCandidateCounts(t *testing.T) {
	idx, err := NewDimensionIndex(3, 12)
	if err != nil {
		t.Fatalf("new index: %v", err)
	}

	vectors := [][]float32{
		{0.10, 0.20, 0.30},
		{0.10, 0.25, 0.35},
		{0.80, 0.20, 0.30},
	}
	for nodeID, vector := range vectors {
		if err := idx.Insert(vector, int32(nodeID)); err != nil {
			t.Fatalf("insert %d: %v", nodeID, err)
		}
	}

	counts, err := idx.CandidateCounts([]float32{0.10, 0.20, 0.30}, 0.05)
	if err != nil {
		t.Fatalf("candidate counts: %v", err)
	}

	want := map[int32]int{
		0: 3,
		1: 3,
		2: 2,
	}
	if !reflect.DeepEqual(counts, want) {
		t.Fatalf("counts mismatch: got %v, want %v", counts, want)
	}
}

func TestDimensionIndexCandidateCountsInBox(t *testing.T) {
	idx, err := NewDimensionIndex(2, 12)
	if err != nil {
		t.Fatalf("new index: %v", err)
	}

	vectors := [][]float32{
		{0.0, 0.0},
		{-0.2, 0.3},
		{0.4, -0.1},
		{-0.3, 0.1},
	}
	for nodeID, vector := range vectors {
		if err := idx.Insert(vector, int32(nodeID)); err != nil {
			t.Fatalf("insert %d: %v", nodeID, err)
		}
	}

	counts, err := idx.CandidateCountsInBox(
		[]float32{0, 0},
		[]float32{0.25, 0.05},
		[]float32{0.10, 0.35},
	)
	if err != nil {
		t.Fatalf("candidate counts in box: %v", err)
	}

	want := map[int32]int{
		0: 2,
		1: 2,
		3: 1,
	}
	if !reflect.DeepEqual(counts, want) {
		t.Fatalf("counts mismatch: got %v, want %v", counts, want)
	}
}

func TestBuildDimensionIndex(t *testing.T) {
	vectors := [][]float32{
		{-1, 2},
		{0, 3},
		{1, 4},
	}

	idx, err := BuildDimensionIndex(len(vectors), 2, 12, func(nodeID int32) []float32 {
		return vectors[nodeID]
	})
	if err != nil {
		t.Fatalf("build index: %v", err)
	}

	got, err := idx.Range(0, -0.5, 1.0, nil)
	if err != nil {
		t.Fatalf("range: %v", err)
	}
	want := []int32{1, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("range mismatch: got %v, want %v", got, want)
	}
}

func TestDimensionIndexValidation(t *testing.T) {
	if _, err := NewDimensionIndex(0, 12); err == nil {
		t.Fatal("expected dimension validation error")
	}

	idx, err := NewDimensionIndex(2, 12)
	if err != nil {
		t.Fatalf("new index: %v", err)
	}
	if err := idx.Insert([]float32{1}, 0); err == nil {
		t.Fatal("expected insert dimension mismatch")
	}
	if _, err := idx.CandidateCounts([]float32{1, 2, 3}, 0.1); err == nil {
		t.Fatal("expected query dimension mismatch")
	}
	if _, err := idx.CandidateCounts([]float32{1, 2}, -0.1); err == nil {
		t.Fatal("expected negative epsilon error")
	}
	if _, err := idx.Range(2, 0, 1, nil); err == nil {
		t.Fatal("expected dimension range error")
	}
	if _, err := idx.CandidateCountsInBox([]float32{1, 2}, []float32{1}, []float32{1, 1}); err == nil {
		t.Fatal("expected minus dimension mismatch")
	}
	if _, err := idx.CandidateCountsInBox([]float32{1, 2}, []float32{1, 1}, []float32{1}); err == nil {
		t.Fatal("expected plus dimension mismatch")
	}
	if _, err := idx.CandidateCountsInBox([]float32{1, 2}, []float32{-1, 1}, []float32{1, 1}); err == nil {
		t.Fatal("expected negative minus error")
	}
	if _, err := idx.CandidateCountsInBox([]float32{1, 2}, []float32{1, 1}, []float32{1, -1}); err == nil {
		t.Fatal("expected negative plus error")
	}
	if _, err := idx.CandidateCountsInBox([]float32{1, 2}, []float32{float32(math.NaN()), 1}, []float32{1, 1}); err == nil {
		t.Fatal("expected non-finite minus error")
	}
}
