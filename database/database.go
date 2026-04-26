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

// Result is one scored search hit.
type Result struct {
	Record     Record
	Distance   float32
	Similarity float32
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
	max := epsilon * float32(math.Sqrt(float64(dimensions)))
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
