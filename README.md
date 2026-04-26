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
- asymmetric per-dimension box search for tuned region generators
- SIMD-accelerated squared L2 scoring on amd64 with a pure-Go fallback
- binary save/load that rebuilds indexes on open

The tradeoff is intentional: skiplists use more memory than dense sorted
`[]int32` arrays, but inserts are easier to keep online because each dimension
gets O(log n) pointer updates instead of slice insertion and shifting.

Symmetric search uses one global epsilon for every dimension. Box search accepts
per-dimension `Minus` and `Plus` widths, so dimension `d` searches:

```text
query[d] - Minus[d] <= value <= query[d] + Plus[d]
```

Soft box search scores every filtered record by weighted distance outside the
box instead of requiring containment:

```text
score = sum(weight[d] * max(0, lower[d] - x[d], x[d] - upper[d])) / sum(weight)
```

Lower scores rank first. A score of `0` means the vector is fully inside the
box. `SoftBoxSearchOptions.ScoreThreshold` can be set to filter records whose
soft-box violation is too large; leave it `nil` for forced top-k retrieval.
Callers can optionally break equal-score ties by anchor distance to the query
vector.

For the Hippo encoder region benchmark, the strongest default was the plain
mean-overflow score above, not max-overflow or L2-added variants. A practical
retrieval profile is `TopK: 20` for recall-oriented search, optionally with
`ScoreThreshold: 0.005` when the caller wants a precision-filtered set.
