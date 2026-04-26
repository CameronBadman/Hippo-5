// Package vectorindex contains ordered indexes for vector coordinates.
package vectorindex

import (
	"errors"
	"math"
	"math/rand"
)

const defaultMaxLevel = 24

// Entry is one vector node's value for a single dimension.
type Entry struct {
	Value  float32
	NodeID int32
}

// SkipList stores dimension values ordered by (Value, NodeID).
//
// It is designed for Hippocampus-style vector indexes where each dimension
// needs O(log n) insertion and efficient range scans over [min, max].
// SkipList is not safe for concurrent writes.
type SkipList struct {
	head     *node
	level    int
	count    int
	maxLevel int
	rng      *rand.Rand
}

type node struct {
	entry Entry
	next  []*node
}

// NewSkipList returns an empty ordered skiplist.
func NewSkipList(maxLevel int) *SkipList {
	if maxLevel <= 0 {
		maxLevel = defaultMaxLevel
	}

	return &SkipList{
		head:     &node{next: make([]*node, maxLevel)},
		level:    1,
		maxLevel: maxLevel,
		rng:      rand.New(rand.NewSource(1)),
	}
}

// Len returns the number of indexed entries.
func (sl *SkipList) Len() int {
	if sl == nil {
		return 0
	}
	return sl.count
}

// Insert adds a dimension value for a node.
func (sl *SkipList) Insert(value float32, nodeID int32) error {
	if sl == nil {
		return errors.New("nil skiplist")
	}
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		return errors.New("value must be finite")
	}
	if nodeID < 0 {
		return errors.New("node id must be non-negative")
	}

	entry := Entry{Value: value, NodeID: nodeID}
	update := make([]*node, sl.maxLevel)
	x := sl.head

	for i := sl.level - 1; i >= 0; i-- {
		for x.next[i] != nil && less(x.next[i].entry, entry) {
			x = x.next[i]
		}
		update[i] = x
	}

	if next := update[0].next[0]; next != nil && equal(next.entry, entry) {
		return errors.New("duplicate entry")
	}

	lvl := sl.randomLevel()
	if lvl > sl.level {
		for i := sl.level; i < lvl; i++ {
			update[i] = sl.head
		}
		sl.level = lvl
	}

	n := &node{
		entry: entry,
		next:  make([]*node, lvl),
	}
	for i := 0; i < lvl; i++ {
		n.next[i] = update[i].next[i]
		update[i].next[i] = n
	}

	sl.count++
	return nil
}

// Delete removes an exact dimension value/node pair.
func (sl *SkipList) Delete(value float32, nodeID int32) bool {
	if sl == nil || sl.count == 0 || math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || nodeID < 0 {
		return false
	}

	entry := Entry{Value: value, NodeID: nodeID}
	update := make([]*node, sl.maxLevel)
	x := sl.head

	for i := sl.level - 1; i >= 0; i-- {
		for x.next[i] != nil && less(x.next[i].entry, entry) {
			x = x.next[i]
		}
		update[i] = x
	}

	target := update[0].next[0]
	if target == nil || !equal(target.entry, entry) {
		return false
	}

	for i := 0; i < sl.level; i++ {
		if update[i].next[i] != target {
			continue
		}
		update[i].next[i] = target.nextAt(i)
	}

	for sl.level > 1 && sl.head.next[sl.level-1] == nil {
		sl.level--
	}
	sl.count--
	return true
}

// Range appends node ids whose values are in [min, max] to dst.
func (sl *SkipList) Range(min, max float32, dst []int32) []int32 {
	if sl == nil || sl.count == 0 || min > max || math.IsNaN(float64(min)) || math.IsNaN(float64(max)) {
		return dst
	}

	x := sl.head
	for i := sl.level - 1; i >= 0; i-- {
		for x.next[i] != nil && x.next[i].entry.Value < min {
			x = x.next[i]
		}
	}

	for x = x.next[0]; x != nil && x.entry.Value <= max; x = x.next[0] {
		dst = append(dst, x.entry.NodeID)
	}
	return dst
}

// ForEach visits entries in sorted order until visit returns false.
func (sl *SkipList) ForEach(visit func(Entry) bool) {
	if sl == nil || visit == nil {
		return
	}
	for x := sl.head.next[0]; x != nil; x = x.next[0] {
		if !visit(x.entry) {
			return
		}
	}
}

func (n *node) nextAt(level int) *node {
	if n == nil || level < 0 || level >= len(n.next) {
		return nil
	}
	return n.next[level]
}

func (sl *SkipList) randomLevel() int {
	level := 1
	for level < sl.maxLevel && sl.rng.Int63()&1 == 0 {
		level++
	}
	return level
}

func less(a, b Entry) bool {
	if a.Value != b.Value {
		return a.Value < b.Value
	}
	return a.NodeID < b.NodeID
}

func equal(a, b Entry) bool {
	return a.Value == b.Value && a.NodeID == b.NodeID
}
