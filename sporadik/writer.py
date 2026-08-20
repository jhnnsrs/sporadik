"""Writing a sparse store: the layouts, then the block, in that order.

The order is the design. Everything except the last write declares something *before* it is true --
``create_array`` writes an array's ``zarr.json`` ahead of its chunks, and a layout's
``encoding-type`` is written when its group is created. Only ``group.attrs[BLOCK_KEY] = ...`` --
one object, written once, after every chunk is durable -- is a statement that the thing it
describes actually exists.

Measured, before the block existed: deleting every chunk of ``data`` left a store that passed every
check on both sides, recorded the right ``nnz``, and returned the right *number* of values for a
slice -- all of them zero, with nothing raised anywhere. Zarr substitutes the fill value for a chunk
it cannot fetch, which is a legitimate convention for genuinely-sparse arrays and, for an
interrupted upload, indistinguishable from success.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from sporadik.layout import MatrixLike, layouts_of, validate_layout
from sporadik.spec import BLOCK_KEY, LAYOUTS_GROUP, block_for

__all__ = ["DEFAULT_CHUNK", "TARGET_CHUNK_BYTES", "chunk_for", "write_store", "write_store_into"]

#: Elements per chunk of ``data``, ``indices`` **and** ``indptr``.
#:
#: Sweeping chunk size over a 16 um Visium HD matrix, both layouts are fastest at the *same*
#: 32 768 -- 0.95 ms for one feature, 0.74 ms for one object -- even though a feature slice averages
#: 4 616 nonzeros and an object slice 967, a factor of five apart. The optimum does not track the
#: slice size, it tracks the **read granularity**: 32 768 four-byte elements is 128 KB, about one
#: object-store request. Below it per-chunk overhead dominates (512 elements: 3.00 ms); above it
#: over-read does (4 Mi elements: 23.55 ms).
#:
#: **Sized for an object store, where the cost is requests rather than bytes.** A byte-addressable
#: store -- one uncompressed chunk per array, so ``indptr`` names an exact byte range -- reads a
#: slice in fewer *bytes*, measured at 376 against 131 072 for one 94-nonzero slice. It does not
#: read it in fewer *round trips*. What a chunk buys that a byte range cannot is **reuse**: a chunk
#: is a cache unit, consecutive slices are adjacent in the array, and a reader walking nearby
#: positions hits chunks it already has, where a range read is fresh every time.
DEFAULT_CHUNK = 32_768

#: What a chunk should cost in bytes, from the measurement above. Used by :func:`chunk_for` when the
#: dtype is not four bytes wide, so the *bytes* stay at the read granularity rather than the element
#: count staying at a number that only happened to be right for float32.
TARGET_CHUNK_BYTES = DEFAULT_CHUNK * 4


def chunk_for(dtype: MatrixLike, nnz: int | None = None) -> int:
    """The chunk size to write, in elements, for an array of this dtype.

    Holds the chunk at :data:`TARGET_CHUNK_BYTES` rather than at a fixed element count, and never
    exceeds the array itself -- a chunk larger than the data is one chunk with a misleading number
    on it.
    """
    width = max(int(np.dtype(dtype).itemsize), 1)
    chunk = max(TARGET_CHUNK_BYTES // width, 1)
    return min(chunk, nnz) if nnz else chunk


def write_store(path: Path | str, matrices: MatrixLike, *, chunk: int | None = None, byte_addressable: bool = False) -> Path:
    """Write an array's layouts as a sparse store at ``path``."""
    write_store_into(zarr.open_group(str(Path(path)), mode="w"), matrices, chunk=chunk, byte_addressable=byte_addressable)
    return Path(path)


def write_store_into(group: MatrixLike, matrices: MatrixLike, *, chunk: int | None = None, byte_addressable: bool = False) -> MatrixLike:
    """Write the layouts into an already-opened zarr group, and return it.

    Split from :func:`write_store` so the same bytes can be written to a local directory or straight
    into a granted object-store prefix -- a caller opens a group over whatever store it has and
    hands it here, so there is one writer rather than one per destination.

    ``byte_addressable=True`` writes one uncompressed chunk per array instead of the default
    chunking, so the stored object *is* the raw little-endian buffer and ``indptr`` names a byte
    range a reader can ask for exactly. Fewer bytes, the same number of round trips, and no reuse --
    worth it for a reader that makes one cold lookup and caches nothing. Nothing declares which was
    done; :func:`sporadik.reader.describe` reads it back off the codecs, because a fact derived from
    the artifact cannot be declared wrong.
    """
    layouts = layouts_of(matrices)
    for layout in layouts.values():
        validate_layout(
            data=layout.data, indices=layout.indices, indptr=layout.indptr, shape=layout.shape, indexed_axis=layout.indexed_axis
        )

    shape = next(iter(layouts.values())).shape
    parent = group.create_group(LAYOUTS_GROUP)
    for _, layout in sorted(layouts.items()):
        child = parent.create_group(layout.path.rsplit("/", 1)[-1])
        child.attrs["encoding-type"] = layout.encoding
        child.attrs["encoding-version"] = "0.1.0"
        child.attrs["shape"] = [int(size) for size in layout.declared_shape]
        for name, array in (("data", layout.data), ("indices", layout.indices), ("indptr", layout.indptr)):
            # `indptr` is chunked like the other two rather than written whole. Whole is fine for a
            # 152 KB one over 19 059 features and a ~22 MB transfer for the 5.4 M-bin one, where two
            # entries would pull the entire object; chunked, they cost one 128 KB GET that then
            # serves the next 32 768 consecutive positions.
            size = len(array) if byte_addressable else (chunk or chunk_for(array.dtype, len(array)))
            child.create_array(
                name,
                shape=array.shape,
                dtype=array.dtype,
                chunks=(max(int(size), 1),),
                compressors=None if byte_addressable else "auto",
            )[:] = array

    group.attrs[BLOCK_KEY] = block_for(shape, layouts)
    return group
