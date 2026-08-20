# sporadik

**A sparse array wire format.** One zarr prefix, one child per *axis made contiguous*, and a block
written last that says which children the writer finished.

It is strictly a *format*: no network, no client, no storage opinions, and no idea what the numbers
mean. Two dependencies — numpy and zarr — and that is the whole of it.

It defines no container of its own. A layout is anndata's spelling, so at rank two the three arrays
are **byte-identical** to what `scanpy` writes and a browser reads one with `zarrita`'s `open` and
`get` rather than a decoder. What sporadik adds is the part anndata has no opinion about: which axis
a layout makes contiguous, how to hold more than one of them, how to tell a finished upload from a
torn one, and what all of that means above rank two.

---

## The idea

A **layout** is one axis made contiguous. Pick the axis a reader will select along, ravel the
remaining axes into `indices`, and `indptr` names one run per position along the chosen one.

At rank two that construction *is* CSR (axis 0) and CSC (axis 1). At rank three it is the same
construction with more axes folded into the same `indices`. One invariant holds at every rank, and
it is the spine of the format:

```
len(indptr) == shape[indexed_axis] + 1
```

So an array of rank *n* has up to *n* layouts, and each buys exactly one question — *"everything at
this position along axis k"* — at the cost of another copy of the nonzeros. Ask a layout the
question it does not compress for and there is no range to read at all, only a scan: **1 777 ms
against 2.2 ms**, measured on a 16 µm spatial-transcriptomics matrix.

---

## The layout on disk

```text
<prefix>/
  zarr.json                     attributes: {"sporadik": {...}}   <- written LAST
  layouts/
    axis0/                      a sparse group, anndata-spelled
      zarr.json                 encoding-type, encoding-version, shape
      data/ indices/ indptr/
    axis1/                      the same array, contiguous along another axis
```

Children are named for **the axis they make contiguous**, not for an encoding: `csr`/`csc` name the
two cases of rank two and say nothing at rank three. A reader recomputes the name and compares, so a
layout filed under the wrong one is refused rather than silently indexed along the wrong axis.

---

## The block — spec 1

Under the key `sporadik` in the root group's own `attributes`:

<!-- spec-block: parsed by tests/test_spec_document.py -- keep in step with sporadik.spec -->
```json
{
  "spec": "1",
  "complete": true,
  "shape": [91033, 19059],
  "layouts": [
    {"path": "layouts/axis0", "indexed_axis": 0, "index_order": [1]},
    {"path": "layouts/axis1", "indexed_axis": 1, "index_order": [0]}
  ]
}
```

| key | declared or derived | meaning |
|---|---|---|
| `spec` | declared | how to read the prefix. **Unknown ⇒ refuse.** |
| `complete` | declared | the writer finished. See *Why the block is last*. |
| `shape` | declared | the array's own shape, at its own rank. Checked against every layout. |
| `layouts[].path` | declared | the child group. Must equal `layouts/axis{indexed_axis}`. |
| `layouts[].indexed_axis` | declared | which axis this layout makes contiguous. |
| `layouts[].index_order` | declared | the axes it did *not* compress, in the order `indices` was raveled over them. |

Everything else a reader needs — each layout's encoding, nonzero count, dtype, chunking, and whether
it is byte-addressable — is **derived from the artifact and never declared.** That distinction is
the design: a fact read off the bytes cannot be stated wrongly.

`index_order` is the one exception, and it has to be stated because it **cannot be recovered from
the bytes**. At rank two it has one member and says nothing. Above it, a wrong `index_order` does not
fail anywhere — it reads a different cell.

---

## A layout child

Each named child is an anndata-spelled sparse group: attributes `encoding-type`, `encoding-version`
and `shape`, plus three 1-D arrays `data`, `indices`, `indptr`.

**`encoding-type` by rank.** At rank two it is anndata's exactly — `csr_matrix` compresses axis 0,
`csc_matrix` axis 1. Above rank two there is no anndata spelling of the thing, so the child holds the
array *raveled to two axes*, which genuinely is a `csr_matrix`, and says so.

**Declared `shape` by rank.** At rank two the child declares the array's own shape, so it is a real
anndata group over the real matrix. Above it, the raveled pair it literally holds:
`[shape[indexed_axis], prod(shape[a] for a in index_order)]`.

---

## Why the block is last

Everything else in the prefix declares something *before* it is true: zarr writes an array's
`zarr.json` ahead of its chunks, and a layout's `encoding-type` when its group is created. Only the
block — one object, written once, after every chunk is durable — is a statement made **after** the
thing it describes exists.

That matters more than it sounds, because zarr substitutes the fill value for a chunk it cannot
fetch. That is a legitimate convention for genuinely-sparse arrays and, for an interrupted upload,
indistinguishable from success. Measured, before the block existed: deleting every chunk of `data`
left a store that passed every other check, recorded the right `nnz`, and returned the right *number*
of values for a slice — all of them zero, with nothing raised anywhere.

---

## What a conforming reader checks

Each one is silent if skipped, which is why each is listed.

1. **`len(indptr) == shape[indexed_axis] + 1`**, at every rank — what makes `indptr[i:i+2]` a run.
2. **`len(data) == len(indices)`** — parallel arrays; one without the other stopped partway.
3. **`path == layouts/axis{indexed_axis}`** — else it is read along the wrong axis.
4. **`index_order` is a permutation** of the axes other than `indexed_axis`.
5. **The child's declared shape and `encoding-type`** match the rank rules above.
6. **At most one layout per axis**, and at most *rank* of them.
7. **The block is present and `complete`.**

---

## Chunking

All three arrays are chunked, `indptr` included, at ~128 KB (32 768 four-byte elements) — sized for
**one object-store request**, because on S3 the cost is round trips rather than bytes. Measured: one
slice costs 0.95 ms at 32 768-element chunks, 3.00 ms at 512, and 23.55 ms at 4 Mi.

A chunk is also a **cache unit**: consecutive slices are adjacent in the array, so a reader walking
nearby positions hits chunks it already holds, which an exact byte range never does. `indptr` is
chunked for the same reason rather than written whole — over 5.4 M positions it is a ~22 MB object
that two entries would otherwise pull entirely.

`write_store(..., byte_addressable=True)` writes one uncompressed chunk per array instead, so the
stored object *is* the raw little-endian buffer and `indptr` names an exact byte range. Fewer bytes
(376 against 131 072 for one 94-nonzero slice), the same number of round trips, no reuse — worth it
for a reader that makes one cold lookup and caches nothing. **Which was written is derived from the
codecs, never declared.**

---

## anndata interop

At **rank two** a layout is a genuine anndata sparse group and its arrays are byte-identical to
anndata's own — asserted in the test suite, not remembered. `read_elem(group["layouts"]["axis0"])`
returns the matrix.

Above rank two, `read_elem` on a child returns the **raveled two-axis view**, which is what the child
literally holds. A degradation, not a lie; the real shape is in the block.

`read_elem` on the **prefix root** does not work at any rank: the root is a container for layouts,
not itself a sparse group.

---

## Install

```bash
pip install sporadik              # numpy + zarr
pip install sporadik[scipy]       # + read_layout(), which returns a scipy.sparse matrix
pip install sporadik[obstore]     # + writing into an object store rather than a directory
```

## Use

```python
import scipy.sparse as sp
import sporadik

counts = sp.random(20_000, 1_200, density=0.01, format="csr")
sporadik.write_store("expression.zarr", [counts, counts.tocsc()])

with sporadik.open_store("expression.zarr", layout=1) as feature_major:
    positions, values = feature_major.slice_at(7)      # two reads, nothing else fetched
```

Above rank two, build a layout per axis and unravel what comes back:

```python
layout = sporadik.layout_over(shape, axis, data=..., indices=..., indptr=..., index_order=(0, 2))
with sporadik.open_store(path, layout=1) as reader:
    (a, c), values = reader.coords_at(7)               # through index_order, not axis order
```

---

## Conformance

`sporadik.spec` is the normative half; everything else in the package is one implementation of it.
A second implementation — in another language, or in a server that must not depend on this package —
reproduces `spec` and nothing more.

That independence is deliberate. A reader that imports its writer inherits the writer's dependencies
and its release cycle, and a version skew between them becomes an outage rather than a refusal. Two
independent implementations of a written-down format is the only arrangement in which *"the format is
specified"* is a testable claim rather than a shared object file.

The block example above is parsed by `tests/test_spec_document.py` and asserted against
`sporadik.spec`, so this document cannot drift from the code it documents.
