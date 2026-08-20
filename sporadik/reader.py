"""Reading a sparse store: what it says about itself, and one slice at a time.

Two reads answer any question this format is for::

    lo, hi = indptr[i], indptr[i + 1]        # where the run for position i is
    indices[lo:hi], data[lo:hi]              # the run

Nothing else is fetched. That is the entire reason to store the layout that makes the wanted axis
contiguous, and the entire cost of asking the other question of it -- there is then no range to
read at all, only a scan: 1 777 ms against 2.2 ms, measured on a 16 um matrix.

The refusals in :func:`describe` are most of this module, and each marks a place where a mistake is
otherwise silent. A store with no block is an upload that died; a layout filed under the wrong name
is read along the wrong axis and returns real, wrong numbers; an ``index_order`` that is not a
permutation reads a different cell every time.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr

from sporadik.errors import IncompleteError, LayoutError, SpecError, SporadikError
from sporadik.layout import MatrixLike
from sporadik.spec import (
    BLOCK_KEY,
    MIN_RANK,
    SUPPORTED_SPECS,
    anndata_encoding,
    layout_path,
    raveled_shape,
)

__all__ = ["SliceInfo", "SparseReader", "StoreInfo", "describe", "open_store", "read_layout"]

#: An open zarr node -- a group or an array. `Any` on purpose: sporadik only ever calls the few
#: methods the two share, and zarr's 3.x line types them differently enough that annotating the
#: union precisely would be describing zarr's version history rather than this format.
ZarrNode = Any


def _open(path: Path | str) -> ZarrNode:
    """The root group of a store, as a node this module can index without narrowing at each step."""
    return zarr.open_group(str(Path(path)), mode="r")


@dataclass(frozen=True)
class SliceInfo:
    """What one layout states about itself. Every field comes off the artifact."""

    path: str
    encoding: str
    encoding_version: str
    shape: tuple[int, ...]
    indexed_axis: int
    index_order: tuple[int, ...]
    nnz: int
    dtype: str
    chunks: dict[str, int]
    #: Whether a slice costs the bytes of the slice rather than the bytes of a chunk.
    #:
    #: **Derived, never declared** -- true exactly when each array is one uncompressed chunk, so the
    #: stored object is the raw buffer and `indptr` names byte offsets into it. False is the
    #: ordinary case and not a defect: the default trades bytes for cache reuse, which is the better
    #: trade when the cost is requests.
    range_readable: bool = False

    @property
    def rank(self) -> int:
        """How many axes the array has."""
        return len(self.shape)

    @property
    def slices(self) -> int:
        """How many contiguous runs this layout holds -- the extent of the compressed axis."""
        return int(self.shape[self.indexed_axis])

    @property
    def index_shape(self) -> tuple[int, ...]:
        """The shape ``indices`` is raveled over: the other axes, in :attr:`index_order`."""
        return tuple(int(self.shape[axis]) for axis in self.index_order)


@dataclass(frozen=True)
class StoreInfo:
    """What one store states about itself: its block, and every layout it names."""

    spec: str
    shape: tuple[int, ...]
    layouts: dict[int, SliceInfo]

    @property
    def rank(self) -> int:
        """How many axes the array has."""
        return len(self.shape)

    def indexing(self, axis: int) -> SliceInfo | None:
        """The layout whose one contiguous read selects along ``axis``, if this store has it.

        The question every reader asks before offering itself. A store holding neither of the
        layouts a question needs does not answer it slowly; it does not answer it at all without
        reading everything.
        """
        return self.layouts.get(int(axis))


def _is_byte_addressable(array: ZarrNode) -> bool:
    """Whether this array's stored object is the raw buffer, so a byte range reads elements.

    One chunk and no compressor. Both halves matter: a compressor makes an offset meaningless, and
    more than one chunk means the offset is into a chunk rather than into the array.
    """
    return not array.compressors and tuple(array.chunks) == tuple(array.shape)


def _describe_layout(group: ZarrNode, entry: dict[str, Any], shape: tuple[int, ...]) -> SliceInfo:
    """Read one layout, refusing anything that contradicts itself or the store around it."""
    path = str(entry.get("path"))
    indexed_axis = entry.get("indexed_axis")
    if not isinstance(indexed_axis, int) or not 0 <= indexed_axis < len(shape):
        raise LayoutError(f"Layout '{path}' declares indexed_axis {indexed_axis!r}, which is not an axis of {shape}")
    if path != layout_path(indexed_axis):
        raise LayoutError(
            f"Layout '{path}' compresses axis {indexed_axis}, so it is filed under the wrong name -- it belongs at "
            f"'{layout_path(indexed_axis)}'. Read from the wrong path it would be indexed along the wrong axis, and "
            "every lookup would return a real, wrong slice."
        )

    others = tuple(axis for axis in range(len(shape)) if axis != indexed_axis)
    order = tuple(entry.get("index_order") or ())
    if sorted(order) != sorted(others):
        raise LayoutError(
            f"Layout '{path}' declares index_order {order}, which is not a permutation of the axes it did not "
            f"compress {others}. That order is how `indices` was raveled and cannot be recovered from the bytes, so a "
            "wrong one does not fail -- it puts every value in a different cell."
        )

    attrs = dict(group.attrs)
    encoding = attrs.get("encoding-type")
    expected_encoding = anndata_encoding(len(shape), indexed_axis)
    if encoding != expected_encoding:
        raise LayoutError(
            f"Layout '{path}' declares encoding-type {encoding!r}, but a layout compressing axis {indexed_axis} of a "
            f"rank-{len(shape)} array is {expected_encoding!r}. The group's own attributes and the block disagree."
        )

    declared = tuple(int(size) for size in attrs.get("shape", ()))
    expected_shape = raveled_shape(shape, indexed_axis, order)
    if declared != expected_shape:
        raise LayoutError(
            f"Layout '{path}' declares shape {declared}, but compressing axis {indexed_axis} of {shape} gives "
            f"{expected_shape}. A store is one array in up to one layout per axis, so every layout has to be that array."
        )

    present = set(group.array_keys())
    missing = [name for name in ("data", "indices", "indptr") if name not in present]
    if missing:
        raise LayoutError(f"Layout '{path}' is missing {', '.join(missing)}; a sparse layout holds all three")

    data, indices, indptr = group["data"], group["indices"], group["indptr"]
    expected = int(shape[indexed_axis]) + 1
    if indptr.shape[0] != expected:
        raise LayoutError(
            f"Layout '{path}' compresses axis {indexed_axis} of {shape}, so `indptr` should have {expected} entries, "
            f"but it has {indptr.shape[0]}. The declaration and the arrays disagree."
        )
    if data.shape != indices.shape:
        raise LayoutError(f"Layout '{path}' has {data.shape[0]} values and {indices.shape[0]} indices; they are parallel")

    return SliceInfo(
        path=path,
        encoding=str(encoding),
        encoding_version=str(attrs.get("encoding-version", "")),
        shape=shape,
        indexed_axis=indexed_axis,
        index_order=order,
        nnz=int(data.shape[0]),
        dtype=str(data.dtype),
        chunks={name: int(group[name].chunks[0]) for name in ("data", "indices", "indptr")},
        range_readable=all(_is_byte_addressable(group[name]) for name in ("data", "indices", "indptr")),
    )


def describe(path: Path | str) -> StoreInfo:
    """Read what a store says about itself, refusing one that says nothing coherent.

    A prefix has no atomic "finished" flag of its own, so the block is one: written last, in a
    single object, after every chunk. A prefix without it is an upload that died partway -- and
    because zarr fills a missing chunk with zeros rather than failing, that is otherwise a store
    which reads back the right *count* of values, every one of them zero, and raises nothing at all.
    """
    path = Path(path)
    group = _open(path)
    block = dict(group.attrs).get(BLOCK_KEY)

    if not isinstance(block, dict):
        raise IncompleteError(
            f"{path} carries no '{BLOCK_KEY}' block, so it is not a sparse store this can read -- or it is an upload "
            "that did not finish. The block is written last, after every chunk, which is the only point at which what "
            "it describes is actually there."
        )
    spec = str(block.get("spec", ""))
    if spec not in SUPPORTED_SPECS:
        raise SpecError(
            f"{path} declares '{BLOCK_KEY}' spec {spec!r}, and this reads {sorted(SUPPORTED_SPECS)}. A spec selects "
            "how every byte in the prefix is read, so an unknown one is refused rather than guessed at."
        )
    if not block.get("complete"):
        raise IncompleteError(f"{path} declares '{BLOCK_KEY}' complete={block.get('complete')!r}; only a finished store is readable")

    shape = tuple(int(size) for size in block.get("shape", ()))
    if len(shape) < MIN_RANK:
        raise SporadikError(f"{path} declares shape {block.get('shape')!r}; a sparse array has at least {MIN_RANK} axes")

    named = block.get("layouts") or []
    if not isinstance(named, list) or not named:
        raise SporadikError(f"{path} names no layouts, so it holds no array. A store is its layouts.")
    if len(named) > len(shape):
        raise SporadikError(
            f"{path} names {len(named)} layouts over a rank-{len(shape)} array, but there is one axis to compress per "
            "axis it has. Any more would be a copy of one of the others."
        )

    layouts: dict[int, SliceInfo] = {}
    for entry in named:
        if not isinstance(entry, dict):
            raise SporadikError(f"{path} names layout {entry!r}, which is not a layout entry -- each names its path, indexed_axis and index_order")
        try:
            child = group[str(entry.get("path"))]
        except KeyError as missing:
            raise IncompleteError(
                f"{path} names layout '{entry.get('path')}', which is not in the prefix. The block lists what the "
                "writer finished, so a name with nothing behind it is an upload that stopped between the two."
            ) from missing
        info = _describe_layout(child, entry, shape)
        if info.indexed_axis in layouts:
            raise LayoutError(f"{path} names two layouts compressing axis {info.indexed_axis}. That is one capability twice.")
        layouts[info.indexed_axis] = info

    return StoreInfo(spec=spec, shape=shape, layouts=layouts)


class SparseReader:
    """One contiguous slice at a time, from one layout, without materialising the array."""

    def __init__(self, path: Path | str, layout: int | None = None) -> None:
        """Open the layout compressing ``layout``, ready to read slices out of it."""
        self.path = Path(path)
        self.store = describe(self.path)
        self.info = self._pick(layout)
        self._group: ZarrNode = _open(self.path)[self.info.path]
        #: `indptr` is cached only when it cannot be byte-addressed. A byte-addressable store is the
        #: point of that variant: an object-major layout over 5.4 M positions has a ~22 MB `indptr`,
        #: and reading two entries out of it should cost 16 bytes. When the layout is chunked there
        #: is no such thing as two entries -- the chunk is the unit -- so it is read once and kept.
        self._indptr = None if self.info.range_readable else self._group["indptr"][:]

    def _pick(self, layout: int | None) -> SliceInfo:
        """The layout to read, refusing an ambiguous choice rather than making it silently."""
        if layout is not None:
            info = self.store.indexing(layout)
            if info is None:
                raise LayoutError(
                    f"{self.path} compresses {sorted(self.store.layouts)}, not axis {layout}. Reading along an axis no "
                    "layout compresses is not slower, it is a scan of every byte rather than one contiguous range."
                )
            return info
        if len(self.store.layouts) != 1:
            raise LayoutError(
                f"{self.path} compresses axes {sorted(self.store.layouts)}, so which one to read is a decision: it is "
                "the whole of what they differ in. Pass `layout=<axis>`."
            )
        return next(iter(self.store.layouts.values()))

    def _read_range(self, name: str, start: int, stop: int) -> npt.NDArray[Any]:
        """Elements ``[start:stop)`` of one array, fetching only their bytes.

        Goes to the store's own range API -- a seek locally, an HTTP Range GET against S3 -- rather
        than through zarr's indexing, which would fetch whole chunks and hand back a slice of them.
        """
        import asyncio

        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        array = self._group[name]
        width = array.dtype.itemsize
        if stop <= start:
            return np.empty(0, dtype=array.dtype)

        store_path = array.store_path
        key = f"{store_path.path}/{array.metadata.chunk_key_encoding.encode_chunk_key((0,))}"
        buffers = asyncio.run(
            store_path.store.get_partial_values(default_buffer_prototype(), [(key, RangeByteRequest(start * width, stop * width))])
        )
        if not buffers or buffers[0] is None:
            raise IncompleteError(
                f"{self.path}: the chunk object behind '{self.info.path}/{name}' is missing. A store that got this far "
                "has a complete block, so the bytes were removed after it was written."
            )
        return np.frombuffer(buffers[0].to_bytes(), dtype=array.dtype)

    def bounds_at(self, position: int) -> tuple[int, int]:
        """The two ``indptr`` entries bracketing one slice: where it starts and ends."""
        if not 0 <= position < self.info.slices:
            raise IndexError(
                f"{position} is not a position along axis {self.info.indexed_axis} of {self.info.shape}, which this "
                f"layout compresses ({self.info.slices} slices)."
            )
        if self._indptr is not None:
            return int(self._indptr[position]), int(self._indptr[position + 1])
        edges = self._read_range("indptr", position, position + 2)
        return int(edges[0]), int(edges[1])

    def slice_at(self, position: int) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """The raveled positions and values of one slice along the compressed axis.

        The positions index the *other* axes raveled together, in ``index_order``. At rank two that
        is simply the other axis; above it, use :meth:`coords_at`.
        """
        low, high = self.bounds_at(position)
        if self.info.range_readable:
            return self._read_range("indices", low, high), self._read_range("data", low, high)
        return self._group["indices"][low:high], self._group["data"][low:high]

    def coords_at(self, position: int) -> tuple[tuple[npt.NDArray[Any], ...], npt.NDArray[Any]]:
        """One slice as one coordinate array per uncompressed axis, plus the values.

        The *i*-th array is the coordinate along ``info.index_order[i]``, not along axis *i*. At
        rank two this is the same thing in a tuple; above it, it is the only honest way to read the
        result, because a raveled position means nothing without the order it was raveled in.
        """
        raveled, values = self.slice_at(position)
        return np.unravel_index(np.asarray(raveled), self.info.index_shape), values

    def dense_slice(self, position: int) -> npt.NDArray[Any]:
        """One slice scattered into a dense array over the uncompressed axes."""
        raveled, values = self.slice_at(position)
        dense = np.zeros(self.info.index_shape, dtype=values.dtype).reshape(-1)
        dense[np.asarray(raveled)] = values
        return dense.reshape(self.info.index_shape)

    def maxima(self) -> npt.NDArray[Any]:
        """The largest absolute value of every slice, in one pass.

        The one method here that is *not* a range read and cannot be: it reduces over every value,
        so it fetches every value. That is what makes it an ingest-time operation -- 1.8 s over the
        88 M nonzeros of a 16 um matrix, against 0.07 s for the same reduction in memory. Do it once
        when writing and store the answer; never per read.
        """
        data = self._group["data"][:]
        edges = np.asarray(self._indptr if self._indptr is not None else self._group["indptr"][:])
        out = np.zeros(self.info.slices, dtype=data.dtype)
        if data.size:
            np.maximum.reduceat(np.abs(data), edges[:-1], out=out)
        out[np.diff(edges) == 0] = 0
        return out

    def close(self) -> None:
        """Drop the cached ``indptr``. Present so the context manager has something to do."""
        self._indptr = np.empty(0, dtype=np.int64)


@contextmanager
def open_store(path: Path | str, layout: int | None = None) -> Generator[SparseReader]:
    """A :class:`SparseReader` over one layout of ``path``, closed on the way out."""
    reader = SparseReader(path, layout)
    try:
        yield reader
    finally:
        reader.close()


def read_layout(path: Path | str, layout: int | None = None) -> MatrixLike:
    """One whole rank-two layout, as `scipy.sparse` -- the one thing here that needs scipy.

    Here so a round trip is provable in three lines and a layout can be handed straight to scanpy.
    A reader answering one question should not call it: it materialises everything, which is what
    :class:`SparseReader` exists to avoid.
    """
    try:
        import scipy.sparse as sp
    except ModuleNotFoundError as missing:  # pragma: no cover - depends on the environment
        raise ModuleNotFoundError(
            "read_layout returns a scipy.sparse matrix, so it needs scipy: pip install sporadik[scipy]. Use "
            "open_store() to read slices without it."
        ) from missing

    reader = SparseReader(path, layout)
    info = reader.info
    if info.rank != MIN_RANK:
        raise SporadikError(
            f"{path} is a rank-{info.rank} array, and `scipy.sparse` has no rank-{info.rank} matrix to return. Read it "
            "a slice at a time with open_store(), which unravels the positions for you."
        )
    group = _open(path)[info.path]
    builder = sp.csr_matrix if info.encoding == "csr_matrix" else sp.csc_matrix
    return builder((group["data"][:], group["indices"][:], group["indptr"][:]), shape=info.shape)
