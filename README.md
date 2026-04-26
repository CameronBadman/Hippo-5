# Hippo-5

Rewrite experiments for Hippocampus.

## Vector Index

`internal/vectorindex` contains a purpose-built skiplist for Hippocampus's
per-dimension vector indexes. It is based on the tower/path mechanics from the
Go-libs skiplist, but it is ordered by `(float32 coordinate, node id)` instead
of weighted text offsets.

The intended search flow is:

1. Store one skiplist per vector dimension.
2. Insert each vector coordinate as `(dimension value, node id)`.
3. For a query vector, range-scan each dimension over
   `[query[dim]-epsilon, query[dim]+epsilon]`.
4. Count node-id matches across dimensions, then score full vectors only for
   candidates that pass the desired match threshold.

## Database Core

`database` is the first pass at the rewritten Hippocampus core:

- file-backed vector records with text, metadata, and timestamps
- one skiplist-backed coordinate index per dimension
- incremental inserts without shifting sorted slices
- exact epsilon-box candidate search followed by Euclidean scoring
- binary save/load that rebuilds indexes on open

The tradeoff is intentional: skiplists use more memory than dense sorted
`[]int32` arrays, but inserts are easier to keep online because each dimension
gets O(log n) pointer updates instead of slice insertion and shifting.
