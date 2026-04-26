// Package database provides the file-backed Hippocampus vector database core.
package database

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"time"

	"hippo5/internal/simd"
	"hippo5/internal/vectorindex"
)

const defaultIndexMaxLevel = 24

// Metadata stores application-defined record fields.
type Metadata map[string]any

// Record is one stored vector and payload.
type Record struct {
	ID        int32
	Vector    []float32
	Text      string
	Metadata  Metadata
	Timestamp time.Time
}

// Filter constrains search results before distance scoring.
type Filter struct {
	Metadata      map[string]any
	TimestampFrom *time.Time
	TimestampTo   *time.Time
}

// SearchOptions controls vector search.
type SearchOptions struct {
	Epsilon   float32
	Threshold float32
	TopK      int
	Filter    *Filter
}

// BoxSearchOptions controls asymmetric per-dimension box search.
type BoxSearchOptions struct {
	Minus     []float32
	Plus      []float32
	Threshold float32
	TopK      int
	Filter    *Filter
}

// SoftBoxSearchOptions controls weighted soft box search.
type SoftBoxSearchOptions struct {
	Minus                    []float32
	Plus                     []float32
	Weights                  []float32
	TopK                     int
	Filter                   *Filter
	TieBreakByAnchorDistance bool
}

// Result is one scored search hit.
type Result struct {
	Record     Record
	Distance   float32
	Similarity float32
	Score      float32
}

// DB stores vectors and a skiplist-backed per-dimension index.
type DB struct {
	dimensions int
	records    []Record
	index      *vectorindex.DimensionIndex
}

// New creates an empty database.
func New(dimensions int) (*DB, error) {
	if dimensions <= 0 {
		return nil, fmt.Errorf("dimensions must be positive")
	}

	index, err := vectorindex.NewDimensionIndex(dimensions, defaultIndexMaxLevel)
	if err != nil {
		return nil, err
	}

	return &DB{
		dimensions: dimensions,
		records:    make([]Record, 0),
		index:      index,
	}, nil
}

// Dimensions returns the configured vector width.
func (db *DB) Dimensions() int {
	if db == nil {
		return 0
	}
	return db.dimensions
}

// Len returns the number of records.
func (db *DB) Len() int {
	if db == nil {
		return 0
	}
	return len(db.records)
}

// Records returns a copy of the record slice. Record vectors are not deep-copied.
func (db *DB) Records() []Record {
	if db == nil {
		return nil
	}
	out := make([]Record, len(db.records))
	copy(out, db.records)
	return out
}

// Insert stores a vector and text payload.
func (db *DB) Insert(vector []float32, text string, metadata Metadata) (int32, error) {
	if db == nil {
		return 0, fmt.Errorf("nil database")
	}
	if err := db.validateVector(vector); err != nil {
		return 0, err
	}

	id := int32(len(db.records))
	vectorCopy := make([]float32, db.dimensions)
	copy(vectorCopy, vector)

	record := Record{
		ID:        id,
		Vector:    vectorCopy,
		Text:      text,
		Metadata:  cloneMetadata(metadata),
		Timestamp: time.Now().UTC(),
	}

	if err := db.index.Insert(record.Vector, record.ID); err != nil {
		return 0, err
	}
	db.records = append(db.records, record)
	return id, nil
}

// Search returns the nearest records inside the exact epsilon box.
func (db *DB) Search(query []float32, opts SearchOptions) ([]Result, error) {
	if db == nil {
		return nil, fmt.Errorf("nil database")
	}
	if err := db.validateVector(query); err != nil {
		return nil, err
	}
	if opts.Epsilon < 0 {
		return nil, fmt.Errorf("epsilon must be non-negative")
	}
	if opts.Threshold < 0 || opts.Threshold > 1 {
		return nil, fmt.Errorf("threshold must be in [0,1]")
	}
	if opts.TopK <= 0 {
		return nil, fmt.Errorf("topK must be positive")
	}
	if len(db.records) == 0 {
		return nil, nil
	}

	counts, err := db.index.CandidateCounts(query, opts.Epsilon)
	if err != nil {
		return nil, err
	}

	maxDistance := opts.Epsilon * float32(math.Sqrt(float64(db.dimensions))) * (1 - opts.Threshold)
	results := make([]Result, 0, opts.TopK)

	for id, count := range counts {
		if count != db.dimensions || int(id) >= len(db.records) {
			continue
		}

		record := db.records[id]
		if !record.matches(opts.Filter) {
			continue
		}

		distanceSquared := simd.SquaredL2(query, record.Vector)
		if distanceSquared > maxDistance*maxDistance {
			continue
		}
		distance := float32(math.Sqrt(float64(distanceSquared)))

		results = append(results, Result{
			Record:     copyRecord(record),
			Distance:   distance,
			Similarity: similarity(distance, opts.Epsilon, db.dimensions),
		})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Distance == results[j].Distance {
			return results[i].Record.ID < results[j].Record.ID
		}
		return results[i].Distance < results[j].Distance
	})

	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	return results, nil
}

// SearchBox returns the nearest records inside an asymmetric per-dimension box.
func (db *DB) SearchBox(query []float32, opts BoxSearchOptions) ([]Result, error) {
	if db == nil {
		return nil, fmt.Errorf("nil database")
	}
	if err := db.validateVector(query); err != nil {
		return nil, err
	}
	if err := db.validateBoxOptions(opts); err != nil {
		return nil, err
	}
	if len(db.records) == 0 {
		return nil, nil
	}

	counts, err := db.index.CandidateCountsInBox(query, opts.Minus, opts.Plus)
	if err != nil {
		return nil, err
	}

	boxDistance := boxMaxDistance(opts.Minus, opts.Plus)
	maxDistance := boxDistance * (1 - opts.Threshold)
	results := make([]Result, 0, opts.TopK)

	for id, count := range counts {
		if count != db.dimensions || int(id) >= len(db.records) {
			continue
		}

		record := db.records[id]
		if !record.matches(opts.Filter) {
			continue
		}

		distanceSquared := simd.SquaredL2(query, record.Vector)
		if distanceSquared > maxDistance*maxDistance {
			continue
		}
		distance := float32(math.Sqrt(float64(distanceSquared)))

		results = append(results, Result{
			Record:     copyRecord(record),
			Distance:   distance,
			Similarity: similarityFromMax(distance, boxDistance),
		})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Distance == results[j].Distance {
			return results[i].Record.ID < results[j].Record.ID
		}
		return results[i].Distance < results[j].Distance
	})

	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	return results, nil
}

// SearchSoftBox ranks records by weighted distance outside an asymmetric box.
//
// Records inside the box have score 0. Records outside the box are scored as:
//
//	sum(weight_i * max(0, lower_i - x_i, x_i - upper_i)) / sum(weight_i)
//
// Lower scores are better. When TieBreakByAnchorDistance is true, equal scores
// are ordered by Euclidean distance to query.
func (db *DB) SearchSoftBox(query []float32, opts SoftBoxSearchOptions) ([]Result, error) {
	if db == nil {
		return nil, fmt.Errorf("nil database")
	}
	if err := db.validateVector(query); err != nil {
		return nil, err
	}
	weightSum, err := db.validateSoftBoxOptions(opts)
	if err != nil {
		return nil, err
	}
	if len(db.records) == 0 {
		return nil, nil
	}

	results := make([]Result, 0, minInt(opts.TopK, len(db.records)))
	for i := range db.records {
		record := db.records[i]
		if !record.matches(opts.Filter) {
			continue
		}

		score := softBoxScore(query, record.Vector, opts.Minus, opts.Plus, opts.Weights, weightSum)
		result := Result{
			Record:     copyRecord(record),
			Score:      score,
			Similarity: 1 / (1 + score),
		}
		if opts.TieBreakByAnchorDistance {
			result.Distance = float32(math.Sqrt(float64(simd.SquaredL2(query, record.Vector))))
		}
		results = append(results, result)
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			if opts.TieBreakByAnchorDistance && results[i].Distance != results[j].Distance {
				return results[i].Distance < results[j].Distance
			}
			return results[i].Record.ID < results[j].Record.ID
		}
		return results[i].Score < results[j].Score
	})

	if len(results) > opts.TopK {
		results = results[:opts.TopK]
	}
	if !opts.TieBreakByAnchorDistance {
		for i := range results {
			results[i].Distance = float32(math.Sqrt(float64(simd.SquaredL2(query, results[i].Record.Vector))))
		}
	}
	return results, nil
}

func (db *DB) validateVector(vector []float32) error {
	if len(vector) != db.dimensions {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", db.dimensions, len(vector))
	}
	for dim, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("dimension %d must be finite", dim)
		}
	}
	return nil
}

func (db *DB) validateBoxOptions(opts BoxSearchOptions) error {
	if len(opts.Minus) != db.dimensions {
		return fmt.Errorf("minus dimension mismatch: expected %d, got %d", db.dimensions, len(opts.Minus))
	}
	if len(opts.Plus) != db.dimensions {
		return fmt.Errorf("plus dimension mismatch: expected %d, got %d", db.dimensions, len(opts.Plus))
	}
	if opts.Threshold < 0 || opts.Threshold > 1 {
		return fmt.Errorf("threshold must be in [0,1]")
	}
	if opts.TopK <= 0 {
		return fmt.Errorf("topK must be positive")
	}
	for dim := 0; dim < db.dimensions; dim++ {
		if math.IsNaN(float64(opts.Minus[dim])) || math.IsInf(float64(opts.Minus[dim]), 0) || opts.Minus[dim] < 0 {
			return fmt.Errorf("minus[%d] must be finite and non-negative", dim)
		}
		if math.IsNaN(float64(opts.Plus[dim])) || math.IsInf(float64(opts.Plus[dim]), 0) || opts.Plus[dim] < 0 {
			return fmt.Errorf("plus[%d] must be finite and non-negative", dim)
		}
	}
	return nil
}

func (db *DB) validateSoftBoxOptions(opts SoftBoxSearchOptions) (float32, error) {
	if len(opts.Minus) != db.dimensions {
		return 0, fmt.Errorf("minus dimension mismatch: expected %d, got %d", db.dimensions, len(opts.Minus))
	}
	if len(opts.Plus) != db.dimensions {
		return 0, fmt.Errorf("plus dimension mismatch: expected %d, got %d", db.dimensions, len(opts.Plus))
	}
	if opts.Weights != nil && len(opts.Weights) != db.dimensions {
		return 0, fmt.Errorf("weights dimension mismatch: expected %d, got %d", db.dimensions, len(opts.Weights))
	}
	if opts.TopK <= 0 {
		return 0, fmt.Errorf("topK must be positive")
	}

	var weightSum float32
	for dim := 0; dim < db.dimensions; dim++ {
		if math.IsNaN(float64(opts.Minus[dim])) || math.IsInf(float64(opts.Minus[dim]), 0) || opts.Minus[dim] < 0 {
			return 0, fmt.Errorf("minus[%d] must be finite and non-negative", dim)
		}
		if math.IsNaN(float64(opts.Plus[dim])) || math.IsInf(float64(opts.Plus[dim]), 0) || opts.Plus[dim] < 0 {
			return 0, fmt.Errorf("plus[%d] must be finite and non-negative", dim)
		}
		weight := float32(1)
		if opts.Weights != nil {
			weight = opts.Weights[dim]
			if math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) || weight < 0 {
				return 0, fmt.Errorf("weights[%d] must be finite and non-negative", dim)
			}
		}
		weightSum += weight
	}
	if weightSum == 0 {
		return 0, fmt.Errorf("sum of weights must be positive")
	}
	return weightSum, nil
}

func (record Record) matches(filter *Filter) bool {
	if filter == nil {
		return true
	}
	if filter.TimestampFrom != nil && record.Timestamp.Before(*filter.TimestampFrom) {
		return false
	}
	if filter.TimestampTo != nil && record.Timestamp.After(*filter.TimestampTo) {
		return false
	}
	for key, expected := range filter.Metadata {
		actual, ok := record.Metadata[key]
		if !ok || !reflect.DeepEqual(actual, expected) {
			return false
		}
	}
	return true
}

func similarity(distance float32, epsilon float32, dimensions int) float32 {
	return similarityFromMax(distance, epsilon*float32(math.Sqrt(float64(dimensions))))
}

func similarityFromMax(distance float32, max float32) float32 {
	if max == 0 {
		if distance == 0 {
			return 1
		}
		return 0
	}
	score := 1 - distance/max
	if score < 0 {
		return 0
	}
	return score
}

func boxMaxDistance(minus []float32, plus []float32) float32 {
	var sum float32
	for dim := range minus {
		bound := minus[dim]
		if plus[dim] > bound {
			bound = plus[dim]
		}
		sum += bound * bound
	}
	return float32(math.Sqrt(float64(sum)))
}

func softBoxScore(query []float32, vector []float32, minus []float32, plus []float32, weights []float32, weightSum float32) float32 {
	var weightedViolation float32
	for dim, x := range vector {
		lower := query[dim] - minus[dim]
		upper := query[dim] + plus[dim]

		var violation float32
		if x < lower {
			violation = lower - x
		} else if x > upper {
			violation = x - upper
		}

		weight := float32(1)
		if weights != nil {
			weight = weights[dim]
		}
		weightedViolation += weight * violation
	}
	return weightedViolation / weightSum
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func copyRecord(record Record) Record {
	record.Vector = append([]float32(nil), record.Vector...)
	record.Metadata = cloneMetadata(record.Metadata)
	return record
}

func cloneMetadata(metadata Metadata) Metadata {
	if metadata == nil {
		return nil
	}
	out := make(Metadata, len(metadata))
	for key, value := range metadata {
		out[key] = value
	}
	return out
}
