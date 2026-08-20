"""sporadik: a sparse array wire format.

A sparse array is written as **one zarr prefix holding one child per axis made contiguous**, plus a
block in the root's own attributes -- written last -- saying which children the writer finished::

    <prefix>/
      zarr.json                     attributes: {"sporadik": {...}}   <- written LAST
      layouts/
        axis0/                      a sparse group, anndata-spelled
          zarr.json                 encoding-type, encoding-version, shape
          data/ indices/ indptr/
        axis1/                      the same array, contiguous along another axis

**This package is a format, and a thin one on purpose.** It defines no container of its own: a
layout is anndata's spelling, so at rank two the three arrays are byte-identical to what `scanpy`
writes and a browser reads one with `zarrita`'s ``open`` and ``get`` rather than a decoder. What
sporadik adds is the part anndata has no opinion about -- which axis a layout makes contiguous, how
to hold more than one of them, how to tell a finished upload from a torn one, and what any of that
means above rank two.

It builds nothing and fetches nothing. numpy and zarr are the only dependencies, and there is no
third thing this needs to be a complete reference implementation of the spec.

**Two axes is one case, not the definition.** A layout is one axis made contiguous: pick the axis a
reader will select along, ravel the rest into ``indices``, and ``indptr`` names a run per position
along the chosen one. At rank two that is exactly CSR (axis 0) and CSC (axis 1). At rank three it is
the same construction with more axes folded into the same ``indices``, so a reader does not change:
one invariant holds throughout, and it is the spine of the format::

    len(indptr) == shape[indexed_axis] + 1

An array of rank *n* therefore has up to *n* layouts, and each one buys exactly one question --
"everything at this position along axis k" -- at the cost of another copy of the nonzeros.

Writing one::

    import scipy.sparse as sp
    import sporadik

    counts = sp.random(20_000, 1_200, density=0.01, format="csr")
    sporadik.write_store("expression.zarr", [counts, counts.tocsc()])

Reading one slice::

    with sporadik.open_store("expression.zarr", layout=1) as feature_major:
        positions, values = feature_major.slice_at(7)     # two reads, nothing else fetched

Above rank two, build a layout per axis and unravel what comes back::

    layout = sporadik.layout_over(shape, axis, data=..., indices=..., indptr=..., index_order=(0, 2))
    with sporadik.open_store(path, layout=1) as reader:
        (a, c), values = reader.coords_at(7)              # through index_order, not axis order

The normative half of all this is :mod:`sporadik.spec`; everything else here is one implementation
of it. A second implementation -- in another language, or in a server that must not depend on this
package -- reproduces `spec` and nothing more. That is deliberate: a reader importing its writer
inherits the writer's dependencies and its release cycle, and a version skew between the two becomes
an outage rather than a refusal.
"""

from sporadik.errors import IncompleteError, LayoutError, SpecError, SporadikError
from sporadik.layout import Layout, layout_of, layout_over, layouts_of, validate_layout
from sporadik.reader import SliceInfo, SparseReader, StoreInfo, describe, open_store, read_layout
from sporadik.spec import (
    BLOCK_KEY,
    ENCODINGS,
    LAYOUTS_GROUP,
    MIN_RANK,
    SPEC_VERSION,
    SUPPORTED_SPECS,
    anndata_encoding,
    block_for,
    layout_path,
    raveled_shape,
)
from sporadik.writer import DEFAULT_CHUNK, TARGET_CHUNK_BYTES, chunk_for, write_store, write_store_into

__all__ = [
    "BLOCK_KEY",
    "DEFAULT_CHUNK",
    "ENCODINGS",
    "LAYOUTS_GROUP",
    "MIN_RANK",
    "SPEC_VERSION",
    "SUPPORTED_SPECS",
    "TARGET_CHUNK_BYTES",
    "IncompleteError",
    "Layout",
    "LayoutError",
    "SliceInfo",
    "SparseReader",
    "SpecError",
    "SporadikError",
    "StoreInfo",
    "anndata_encoding",
    "block_for",
    "chunk_for",
    "describe",
    "layout_of",
    "layout_over",
    "layout_path",
    "layouts_of",
    "open_store",
    "raveled_shape",
    "read_layout",
    "validate_layout",
    "write_store",
    "write_store_into",
]
