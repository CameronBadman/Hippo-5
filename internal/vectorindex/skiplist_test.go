package vectorindex

import (
	"math"
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

func TestSkipListRange(t *testing.T) {
	sl := NewSkipList(12)
	for _, entry := range []Entry{
		{Value: 0.30, NodeID: 3},
		{Value: 0.10, NodeID: 1},
		{Value: 0.20, NodeID: 2},
		{Value: 0.20, NodeID: 4},
		{Value: -0.50, NodeID: 5},
	} {
		if err := sl.Insert(entry.Value, entry.NodeID); err != nil {
			t.Fatalf("insert %+v: %v", entry, err)
		}
	}

	got := sl.Range(0.15, 0.30, nil)
	want := []int32{2, 4, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("range mismatch: got %v, want %v", got, want)
	}
}

func TestSkipListDelete(t *testing.T) {
	sl := NewSkipList(8)
	for _, entry := range []Entry{
		{Value: 1.0, NodeID: 10},
		{Value: 1.0, NodeID: 11},
		{Value: 2.0, NodeID: 12},
	} {
		if err := sl.Insert(entry.Value, entry.NodeID); err != nil {
			t.Fatalf("insert %+v: %v", entry, err)
		}
	}

	if !sl.Delete(1.0, 10) {
		t.Fatal("expected delete to succeed")
	}
	if sl.Delete(1.0, 10) {
		t.Fatal("expected second delete to fail")
	}

	got := sl.Range(0.0, 2.0, nil)
	want := []int32{11, 12}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("range after delete mismatch: got %v, want %v", got, want)
	}
}

func TestSkipListRejectsBadInput(t *testing.T) {
	sl := NewSkipList(8)
	if err := sl.Insert(0.1, -1); err == nil {
		t.Fatal("expected negative node id error")
	}
	if err := sl.Insert(float32(math.NaN()), 1); err == nil {
		t.Fatal("expected NaN value error")
	}
	if err := sl.Insert(0.1, 1); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := sl.Insert(0.1, 1); err == nil {
		t.Fatal("expected duplicate entry error")
	}
}

func TestSkipListRandomRangesAgainstModel(t *testing.T) {
	sl := NewSkipList(20)
	model := make([]Entry, 0)
	rng := rand.New(rand.NewSource(42))

	for i := int32(0); i < 1000; i++ {
		value := float32(rng.Intn(2000)-1000) / 100
		entry := Entry{Value: value, NodeID: i}
		if err := sl.Insert(entry.Value, entry.NodeID); err != nil {
			t.Fatalf("insert %+v: %v", entry, err)
		}
		model = append(model, entry)
	}

	sort.Slice(model, func(i, j int) bool {
		return less(model[i], model[j])
	})

	for i := 0; i < 200; i++ {
		a := float32(rng.Intn(2000)-1000) / 100
		b := float32(rng.Intn(2000)-1000) / 100
		if a > b {
			a, b = b, a
		}

		got := sl.Range(a, b, nil)
		want := make([]int32, 0)
		for _, entry := range model {
			if entry.Value >= a && entry.Value <= b {
				want = append(want, entry.NodeID)
			}
		}

		if !reflect.DeepEqual(got, want) {
			t.Fatalf("range [%v,%v] mismatch:\ngot  %v\nwant %v", a, b, got, want)
		}
	}
}
