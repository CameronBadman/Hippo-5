package database

import (
	"bytes"
	"math"
	"reflect"
	"testing"
	"time"
)

func TestInsertAndSearch(t *testing.T) {
	db, err := New(3)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}

	if _, err := db.Insert([]float32{0.1, 0.2, 0.3}, "first", Metadata{"user": "alice"}); err != nil {
		t.Fatalf("insert first: %v", err)
	}
	if _, err := db.Insert([]float32{0.1, 0.3, 0.2}, "second", Metadata{"user": "bob"}); err != nil {
		t.Fatalf("insert second: %v", err)
	}
	if _, err := db.Insert([]float32{0.9, 0.1, 0.05}, "third", Metadata{"user": "alice"}); err != nil {
		t.Fatalf("insert third: %v", err)
	}

	results, err := db.Search([]float32{0.1, 0.25, 0.25}, SearchOptions{
		Epsilon:   0.2,
		Threshold: 0.0,
		TopK:      5,
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d: %+v", len(results), results)
	}
	if results[0].Record.Text != "first" {
		t.Fatalf("nearest result = %q, want first", results[0].Record.Text)
	}
}

func TestSearchFilter(t *testing.T) {
	db, err := New(2)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}

	if _, err := db.Insert([]float32{0, 0}, "alice note", Metadata{"user": "alice"}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if _, err := db.Insert([]float32{0, 0.1}, "bob note", Metadata{"user": "bob"}); err != nil {
		t.Fatalf("insert: %v", err)
	}

	results, err := db.Search([]float32{0, 0}, SearchOptions{
		Epsilon:   0.2,
		Threshold: 0,
		TopK:      5,
		Filter: &Filter{
			Metadata: map[string]any{"user": "bob"},
		},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 || results[0].Record.Text != "bob note" {
		t.Fatalf("unexpected filtered results: %+v", results)
	}
}

func TestTimestampFilter(t *testing.T) {
	db, err := New(1)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	if _, err := db.Insert([]float32{1}, "now", nil); err != nil {
		t.Fatalf("insert: %v", err)
	}

	future := time.Now().UTC().Add(time.Hour)
	results, err := db.Search([]float32{1}, SearchOptions{
		Epsilon:   0,
		Threshold: 0,
		TopK:      1,
		Filter:    &Filter{TimestampFrom: &future},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected timestamp filter to exclude result: %+v", results)
	}
}

func TestValidation(t *testing.T) {
	db, err := New(2)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}

	if _, err := db.Insert([]float32{1}, "bad", nil); err == nil {
		t.Fatal("expected insert dimension mismatch")
	}
	if _, err := db.Insert([]float32{float32(math.NaN()), 1}, "bad", nil); err == nil {
		t.Fatal("expected insert finite validation")
	}
	if _, err := db.Search([]float32{1, 2}, SearchOptions{Epsilon: -1, TopK: 1}); err == nil {
		t.Fatal("expected epsilon validation")
	}
	if _, err := db.Search([]float32{1, 2}, SearchOptions{Epsilon: 1, Threshold: 2, TopK: 1}); err == nil {
		t.Fatal("expected threshold validation")
	}
	if _, err := db.Search([]float32{1, 2}, SearchOptions{Epsilon: 1}); err == nil {
		t.Fatal("expected topK validation")
	}
}

func TestPersistenceRoundTrip(t *testing.T) {
	db, err := New(3)
	if err != nil {
		t.Fatalf("new db: %v", err)
	}
	if _, err := db.Insert([]float32{0.1, 0.2, 0.3}, "first", Metadata{"rank": float64(1)}); err != nil {
		t.Fatalf("insert first: %v", err)
	}
	if _, err := db.Insert([]float32{0.4, 0.5, 0.6}, "second", nil); err != nil {
		t.Fatalf("insert second: %v", err)
	}

	var buf bytes.Buffer
	if err := db.Write(&buf); err != nil {
		t.Fatalf("write: %v", err)
	}

	loaded, err := Read(&buf)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if loaded.Dimensions() != 3 || loaded.Len() != 2 {
		t.Fatalf("loaded db shape = dims %d len %d", loaded.Dimensions(), loaded.Len())
	}

	results, err := loaded.Search([]float32{0.1, 0.2, 0.3}, SearchOptions{
		Epsilon:   0,
		Threshold: 0,
		TopK:      1,
	})
	if err != nil {
		t.Fatalf("search loaded: %v", err)
	}
	if len(results) != 1 || results[0].Record.Text != "first" {
		t.Fatalf("unexpected loaded results: %+v", results)
	}
	if !reflect.DeepEqual(results[0].Record.Metadata, Metadata{"rank": float64(1)}) {
		t.Fatalf("metadata mismatch: %+v", results[0].Record.Metadata)
	}
}
