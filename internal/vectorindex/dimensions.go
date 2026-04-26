package vectorindex

import (
	"fmt"
	"math"
)

// DimensionIndex maintains one ordered skiplist per vector dimension.
type DimensionIndex struct {
	dimensions int
	lists      []*SkipList
}

// NewDimensionIndex creates an empty per-dimension vector index.
func NewDimensionIndex(dimensions int, maxLevel int) (*DimensionIndex, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("dimensions must be positive")
	}

	idx := &DimensionIndex{
		dimensions: dimensions,
		lists:      make([]*SkipList, dimensions),
	}
	for dim := range idx.lists {
		idx.lists[dim] = NewSkipList(maxLevel)
	}
	return idx, nil
}

// Dimensions returns the number of dimensions indexed.
func (idx *DimensionIndex) Dimensions() int {
	if idx == nil {
		return 0
	}
	return idx.dimensions
}

// Insert indexes one vector under nodeID.
func (idx *DimensionIndex) Insert(vector []float32, nodeID int32) error {
	if idx == nil {
		return fmt.Errorf("nil dimension index")
	}
	if len(vector) != idx.dimensions {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", idx.dimensions, len(vector))
	}

	for dim, value := range vector {
		if err := idx.lists[dim].Insert(value, nodeID); err != nil {
			return fmt.Errorf("dimension %d: %w", dim, err)
		}
	}
	return nil
}

// Range appends node ids whose value in dim is inside [min, max].
func (idx *DimensionIndex) Range(dim int, min float32, max float32, dst []int32) ([]int32, error) {
	if idx == nil {
		return dst, fmt.Errorf("nil dimension index")
	}
	if dim < 0 || dim >= idx.dimensions {
		return dst, fmt.Errorf("dimension %d out of range [0,%d)", dim, idx.dimensions)
	}
	return idx.lists[dim].Range(min, max, dst), nil
}

// CandidateCounts counts how many dimensions each node matched for an
// epsilon-box around query.
func (idx *DimensionIndex) CandidateCounts(query []float32, epsilon float32) (map[int32]int, error) {
	if idx == nil {
		return nil, fmt.Errorf("nil dimension index")
	}
	if len(query) != idx.dimensions {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", idx.dimensions, len(query))
	}
	if epsilon < 0 {
		return nil, fmt.Errorf("epsilon must be non-negative")
	}

	counts := make(map[int32]int)
	var matches []int32
	for dim, value := range query {
		matches = matches[:0]
		matches = idx.lists[dim].Range(value-epsilon, value+epsilon, matches)
		for _, nodeID := range matches {
			counts[nodeID]++
		}
	}
	return counts, nil
}

// CandidateCountsInBox counts how many dimensions each node matched for an
// asymmetric per-dimension box around query.
func (idx *DimensionIndex) CandidateCountsInBox(query []float32, minus []float32, plus []float32) (map[int32]int, error) {
	if idx == nil {
		return nil, fmt.Errorf("nil dimension index")
	}
	if len(query) != idx.dimensions {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", idx.dimensions, len(query))
	}
	if len(minus) != idx.dimensions {
		return nil, fmt.Errorf("minus dimension mismatch: expected %d, got %d", idx.dimensions, len(minus))
	}
	if len(plus) != idx.dimensions {
		return nil, fmt.Errorf("plus dimension mismatch: expected %d, got %d", idx.dimensions, len(plus))
	}

	counts := make(map[int32]int)
	var matches []int32
	for dim, value := range query {
		if math.IsNaN(float64(minus[dim])) || math.IsInf(float64(minus[dim]), 0) || minus[dim] < 0 {
			return nil, fmt.Errorf("minus[%d] must be finite and non-negative", dim)
		}
		if math.IsNaN(float64(plus[dim])) || math.IsInf(float64(plus[dim]), 0) || plus[dim] < 0 {
			return nil, fmt.Errorf("plus[%d] must be finite and non-negative", dim)
		}
		matches = matches[:0]
		matches = idx.lists[dim].Range(value-minus[dim], value+plus[dim], matches)
		for _, nodeID := range matches {
			counts[nodeID]++
		}
	}
	return counts, nil
}

// BuildDimensionIndex indexes all vectors in order. The getVector callback
// should return the vector for nodeID.
func BuildDimensionIndex(count int, dimensions int, maxLevel int, getVector func(nodeID int32) []float32) (*DimensionIndex, error) {
	if count < 0 {
		return nil, fmt.Errorf("count must be non-negative")
	}
	if getVector == nil {
		return nil, fmt.Errorf("getVector is required")
	}

	idx, err := NewDimensionIndex(dimensions, maxLevel)
	if err != nil {
		return nil, err
	}
	for nodeID := int32(0); nodeID < int32(count); nodeID++ {
		if err := idx.Insert(getVector(nodeID), nodeID); err != nil {
			return nil, err
		}
	}
	return idx, nil
}
