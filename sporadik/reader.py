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

import asyncio
from collections.abc import Coroutine, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import zarr

from sporadik.errors import IncompleteError, LayoutError, SpecError, SporadikError
from sporadik.layout import Layout, MatrixLike, layout_over, maxima_of
from sporadik.spec import (
    BLOCK_KEY,
    MIN_RANK,
    SUPPORTED_SPECS,
    anndata_encoding,
    layout_path,
    raveled_shape,
)

__all__ = ["Selection", "SliceInfo", "SparseReader", "StoreInfo", "ZarrNode", "describe", "open_store", "read_layout"]

#: An open zarr node -- a group or an array. `Any` on purpose: sporadik only ever calls the few
#: methods the two share, and zarr's 3.x line types them differently enough that annotating the
#: union precisely would be describing zarr's version history rather than this format.
ZarrNode = Any

#: The value one zarr coroutine resolves to, so :func:`_run` hands it back unnarrowed.
_T = TypeVar("_T")


def _open(path: Path | str) -> ZarrNode:
    """The root group of a store, as a node this module can index without narrowing at each step."""
    return zarr.open_group(str(Path(path)), mode="r")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    """Drive one zarr coroutine to completion, refusing to do it inside a running loop.

    Nothing in this package is async, and that is a decision rather than an omission: the two waves
    a read makes are strictly dependent -- the second's byte ranges are computed from the first's
    bytes -- so an async reader would buy no latency that batching does not already buy, at the
    price of a second implementation of every refusal in this module.

    What the stance owed a caller and never paid is *saying so*. Inside a running loop `asyncio.run`
    raised asyncio's own error about asyncio, naming neither this package nor the way out.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise SporadikError(
        "This is a synchronous reader, called from inside a running event loop, where the range read it makes cannot "
        "drive a loop of its own. Read in a worker thread: `await asyncio.to_thread(reader.slice_at, position)`."
    )


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
    #: Whether the layout carries the per-slice maxima its writer computed. Derived, never declared:
    #: the array is either in the group or it is not, and a store written before it existed says
    #: nothing about it rather than saying something false.
    has_maxima: bool = False

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
            f"{expected_shape}. A store is one array in up to one layout per axis, so every layout has to be that "
            "array."
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
        raise LayoutError(
            f"Layout '{path}' has {data.shape[0]} values and {indices.shape[0]} indices; they are parallel"
        )

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
        has_maxima="maxima" in present,
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
        raise IncompleteError(
            f"{path} declares '{BLOCK_KEY}' complete={block.get('complete')!r}; only a finished store is readable"
        )

    shape = tuple(int(size) for size in block.get("shape", ()))
    if len(shape) < MIN_RANK:
        raise SporadikError(
            f"{path} declares shape {block.get('shape')!r}; a sparse array has at least {MIN_RANK} axes"
        )

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
            raise SporadikError(
                f"{path} names layout {entry!r}, which is not a layout entry -- each names its path, indexed_axis "
                "and index_order"
            )
        try:
            child = group[str(entry.get("path"))]
        except KeyError as missing:
            raise IncompleteError(
                f"{path} names layout '{entry.get('path')}', which is not in the prefix. The block lists what the "
                "writer finished, so a name with nothing behind it is an upload that stopped between the two."
            ) from missing
        info = _describe_layout(child, entry, shape)
        if info.indexed_axis in layouts:
            raise LayoutError(
                f"{path} names two layouts compressing axis {info.indexed_axis}. That is one capability twice."
            )
        layouts[info.indexed_axis] = info

    return StoreInfo(spec=spec, shape=shape, layouts=layouts)


def _coalesce(starts: npt.NDArray[Any], stops: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Sorted ``[start, stop)`` runs, with touching and overlapping ones merged into one.

    **Only exactly adjacent ranges are merged, and that is a decision.** Merging across a *gap*
    buys one request and pays for the bytes in between, and where those cross is a measurement
    nobody here has taken -- while the backend this format is for has taken it: zarr's obstore store
    groups a call's ranges per object and hands them to ``get_ranges``, which merges nearby ones
    with a heuristic tuned for object stores. Guessing a second threshold on top of a measured one,
    with less information than it has, would make reads worse in a way nothing here would catch.
    What is left is the merge that costs nothing, and this does that one.

    `maximum.accumulate` rather than a pairwise comparison because runs may nest: a position whose
    slice is empty sits inside its neighbour's range and must not break one in two.
    """
    keep = np.flatnonzero(stops > starts)
    if keep.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(starts[keep], kind="stable")
    lo = np.asarray(starts[keep])[order].astype(np.int64, copy=False)
    hi = np.asarray(stops[keep])[order].astype(np.int64, copy=False)
    reach = np.maximum.accumulate(hi)
    breaks = np.empty(lo.size, dtype=bool)
    breaks[0] = True
    breaks[1:] = lo[1:] > reach[:-1]
    heads = np.flatnonzero(breaks)
    return lo[heads], np.maximum.reduceat(hi, heads)


def _ragged_take(base: npt.NDArray[Any], counts: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """The selection's own ``indptr``, and the take-array that gathers it out of what was fetched.

    ``base[i]`` is where slice *i*'s run begins in whatever buffer it is being taken from -- an
    offset into the concatenated fetch on a byte-addressable store, a global element index on a
    chunked one. The same arithmetic serves both, which is why the two paths differ only in what
    they put in ``base``. Requested order and repeats fall out of it: neither is a special case,
    because ``base`` is built in the caller's order and nothing here says the entries are distinct.
    """
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64, copy=False)
    total = int(indptr[-1])
    if total == 0:
        return indptr, np.empty(0, dtype=np.intp)
    take = np.arange(total, dtype=np.intp) - np.repeat(indptr[:-1], counts) + np.repeat(base, counts)
    return indptr, take.astype(np.intp, copy=False)


@dataclass(frozen=True, eq=False)
class Selection:
    """Many slices, read together -- the bytes, and the frame they came out of.

    **Row *i* of a selection is ``positions[i]``, not *i*.** That is the whole reason this is its
    own type rather than a :class:`~sporadik.Layout`. The arrangement genuinely is a layout: same
    ``indexed_axis``, same ``index_order``, an ``indptr`` naming one run per slice. But a `Layout`
    whose ``shape[indexed_axis]`` is 200 is indistinguishable from a real 200-slice array -- hand it
    to a writer and you get a valid, complete store of something that never existed, and read a
    coordinate off it and you get a real, wrong one. That is the failure `index_order` exists to
    prevent, and it is not worth reintroducing for the convenience of skipping a method call.

    So the frame stays attached to the bytes: :attr:`positions` is what row *i* means, :meth:`coords`
    converts back to the original array's frame, and :meth:`as_layout` is there for a caller who
    *means* "these slices are my array now" and says so.

    A new allocation, not a view: ``indptr`` is recomputed over the selection and the runs are
    gathered into fresh arrays. ``indices`` still ravel over the *original* ``index_shape``, which a
    selection does not change, so they are read exactly as :meth:`SparseReader.coords_at` reads one
    slice's.
    """

    #: The positions asked for, in the order they were asked for. Row *i* of this selection is
    #: ``positions[i]`` of the array it was read from.
    positions: npt.NDArray[Any]
    #: One run per requested position: ``indptr[i]:indptr[i + 1]`` into `indices` and `data`.
    indptr: npt.NDArray[Any]
    indices: npt.NDArray[Any]
    data: npt.NDArray[Any]
    #: The shape of the array this came out of -- not of the selection.
    shape: tuple[int, ...]
    indexed_axis: int
    index_order: tuple[int, ...]

    def __len__(self) -> int:
        """How many slices were asked for, repeats included."""
        return len(self.positions)

    @property
    def nnz(self) -> int:
        """How many nonzeros the selection holds."""
        return len(self.data)

    @property
    def index_shape(self) -> tuple[int, ...]:
        """The uncompressed axes, in ``index_order`` -- what `indices` ravel over."""
        return tuple(self.shape[axis] for axis in self.index_order)

    def slice_at(self, which: int) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """The *which*-th slice of this selection -- by its place in ``positions``, not by position."""
        if not 0 <= which < len(self):
            raise IndexError(f"{which} is not one of the {len(self)} slices in this selection")
        low, high = int(self.indptr[which]), int(self.indptr[which + 1])
        return self.indices[low:high], self.data[low:high]

    def coords(self) -> tuple[tuple[npt.NDArray[Any], ...], npt.NDArray[Any]]:
        """The whole selection as one coordinate array per axis, **in the original array's frame**.

        The compressed axis's coordinate comes from :attr:`positions`, repeated once per nonzero in
        its run -- which is the step a caller doing this by hand forgets, and the reason forgetting
        is expensive: the ids come out plausible and wrong.
        """
        along = np.repeat(np.asarray(self.positions), np.diff(np.asarray(self.indptr)))
        rest = np.unravel_index(np.asarray(self.indices), self.index_shape)
        coords: list[npt.NDArray[Any]] = [np.empty(0, dtype=np.int64)] * len(self.shape)
        coords[self.indexed_axis] = along
        for place, axis in enumerate(self.index_order):
            coords[axis] = rest[place]
        return tuple(coords), np.asarray(self.data)

    def to_dense(self) -> npt.NDArray[Any]:
        """The selection scattered into ``(len(self), *index_shape)``, dense over the other axes."""
        rows = np.repeat(np.arange(len(self)), np.diff(np.asarray(self.indptr)))
        dense = np.zeros((len(self), int(np.prod(self.index_shape))), dtype=self.data.dtype)
        dense[rows, np.asarray(self.indices)] = self.data
        return dense.reshape(len(self), *self.index_shape)

    def as_layout(self) -> Layout:
        """This selection as an array in its own right, of ``len(self)`` slices along its axis.

        Deliberately a method and not what :meth:`SparseReader.slices_at` returns. Calling it is a
        caller saying that the selection *is* the array now -- that row *i* means *i* and
        :attr:`positions` has been dealt with or does not matter. Written out through
        :func:`~sporadik.layout_over`, so the result is refused here if it would not have been a
        layout.
        """
        shape = tuple(len(self) if axis == self.indexed_axis else size for axis, size in enumerate(self.shape))
        return layout_over(
            shape, self.indexed_axis, data=self.data, indices=self.indices, indptr=self.indptr,
            index_order=self.index_order,
        )

    def __repr__(self) -> str:
        """How many slices, how many nonzeros, and which axis they were taken along."""
        return f"Selection({len(self)} slices along axis{self.indexed_axis}, nnz={self.nnz}, of shape={self.shape})"


class SparseReader:
    """One contiguous slice at a time, from one layout, without materialising the array."""

    def __init__(self, path: Path | str, axis: int | None = None) -> None:
        """Open the layout compressing ``axis``, ready to read slices out of it."""
        self.path = Path(path)
        self.store = describe(self.path)
        self.info = self._pick(axis)
        self._group: ZarrNode = _open(self.path)[self.info.path]
        #: `indptr` is cached only when it cannot be byte-addressed. A byte-addressable store is the
        #: point of that variant: an object-major layout over 5.4 M positions has a ~22 MB `indptr`,
        #: and reading two entries out of it should cost 16 bytes. When the layout is chunked there
        #: is no such thing as two entries -- the chunk is the unit -- so it is read once and kept.
        self._indptr = None if self.info.range_readable else self._group["indptr"][:]
        #: Set by :meth:`close`. A reader is not reopened; :class:`~sporadik.SparseStore` drops it
        #: and opens another, which is why this only has to refuse rather than recover.
        self._closed = False

    def _pick(self, axis: int | None) -> SliceInfo:
        """The layout to read, refusing an ambiguous choice rather than making it silently."""
        if axis is not None:
            info = self.store.indexing(axis)
            if info is None:
                raise LayoutError(
                    f"{self.path} compresses {sorted(self.store.layouts)}, not axis {axis}. Reading along an axis no "
                    "layout compresses is not slower, it is a scan of every byte rather than one contiguous range."
                )
            return info
        if len(self.store.layouts) != 1:
            raise LayoutError(
                f"{self.path} compresses axes {sorted(self.store.layouts)}, so which one to read is a decision: it is "
                "the whole of what they differ in. Pass `axis=<axis>`."
            )
        return next(iter(self.store.layouts.values()))

    def _read_ranges(self, wanted: Sequence[tuple[str, int, int]]) -> list[npt.NDArray[Any]]:
        """Elements ``[start:stop)`` of each named array, in **one** request to the store.

        The plural is the point. ``get_partial_values`` has always taken a list of (key, range)
        pairs and has always fetched them concurrently -- zarr's own stores map the list through
        `concurrent_map`, and its obstore backend additionally merges nearby ranges per object --
        and this module was handing it a list of one, three times, to read a single slice. One call
        for the `indices` run and the `data` run takes that to two. Nothing about the bytes changes;
        what changes is how many times we wait.

        Goes to the store's own range API -- a seek locally, an HTTP Range GET against S3 -- rather
        than through zarr's indexing, which would fetch whole chunks and hand back a slice of them.

        Returns one array per request, in request order, so a caller can unpack positionally. An
        empty request (``stop <= start``) never reaches the store and its slot is filled from
        nothing -- a batch of slices is full of empty ones and none of them are worth a byte range.
        """
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        arrays: dict[str, ZarrNode] = {name: self._group[name] for name, _, _ in wanted}
        pairs: list[tuple[str, RangeByteRequest]] = []
        slots: list[int | None] = []
        for name, start, stop in wanted:
            if stop <= start:
                slots.append(None)
                continue
            array = arrays[name]
            width = int(array.dtype.itemsize)
            store_path = array.store_path
            key = f"{store_path.path}/{array.metadata.chunk_key_encoding.encode_chunk_key((0,))}"
            slots.append(len(pairs))
            pairs.append((key, RangeByteRequest(start * width, stop * width)))

        buffers: list[Any] = []
        if pairs:
            # Every array of a layout is a sibling in one group, so one store serves them all and
            # `indices` and `data` ranges belong in the same call rather than in two.
            store = arrays[wanted[0][0]].store_path.store
            buffers = _run(store.get_partial_values(default_buffer_prototype(), pairs))

        out: list[npt.NDArray[Any]] = []
        for (name, _, _), slot in zip(wanted, slots):
            dtype = arrays[name].dtype
            if slot is None:
                out.append(np.empty(0, dtype=dtype))
                continue
            buffer = buffers[slot] if slot < len(buffers) else None
            if buffer is None:
                raise IncompleteError(
                    f"{self.path}: the chunk object behind '{self.info.path}/{name}' is missing. A store that got this "
                    "far has a complete block, so the bytes were removed after it was written."
                )
            out.append(np.frombuffer(buffer.to_bytes(), dtype=dtype))
        return out

    def _read_range(self, name: str, start: int, stop: int) -> npt.NDArray[Any]:
        """Elements ``[start:stop)`` of one array.

        :meth:`_read_ranges` for a single request, kept because `bounds_at` genuinely wants one and
        a list of one reads worse at the call site than a call of one.
        """
        return self._read_ranges([(name, start, stop)])[0]

    def bounds_at(self, position: int) -> tuple[int, int]:
        """The two ``indptr`` entries bracketing one slice: where it starts and ends."""
        if self._closed:
            raise SporadikError(f"This reader over {self.path} is closed; open another to read from it again.")
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

        Two waves on a byte-addressable store, not three: the `indices` run and the `data` run are
        the same range over sibling arrays and go in one request. It is two rather than one because
        the second wave's byte ranges are computed from the first's bytes -- `indptr` has to say
        where the run is before there is a run to ask for -- and no arrangement of this format
        removes that dependency.
        """
        low, high = self.bounds_at(position)
        if self.info.range_readable:
            indices, data = self._read_ranges([("indices", low, high), ("data", low, high)])
            return indices, data
        return self._group["indices"][low:high], self._group["data"][low:high]

    def _edges_at(self, positions: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """The ``indptr`` bracket of every position, in one wave rather than one per position."""
        if self._closed:
            raise SporadikError(f"This reader over {self.path} is closed; open another to read from it again.")
        if self._indptr is not None:
            edges = np.asarray(self._indptr)
            return edges[positions], edges[positions + 1]
        # Each position wants `indptr[p:p+2]`, so adjacent positions ask for overlapping windows and
        # `_coalesce` folds them into one range. A contiguous batch therefore costs one range here,
        # not one per slice.
        lo, hi = _coalesce(positions, positions + 2)
        buffers = self._read_ranges([("indptr", int(a), int(b)) for a, b in zip(lo, hi)])
        edges = np.concatenate(buffers) if buffers else np.empty(0, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(hi - lo)))[:-1]
        which = np.searchsorted(lo, positions, side="right") - 1
        at = offsets[which] + (positions - lo[which])
        return edges[at], edges[at + 1]

    def slices_at(self, positions: Sequence[int] | npt.NDArray[Any]) -> Selection:
        """Many slices, in as few waves as the format allows.

        **Two waves, whatever ``len(positions)`` is** -- one for the `indptr` brackets, one for every
        `indices` and `data` run together -- against three *per position* before this existed. A
        two-hundred-cell read was six hundred round trips; it is two. On a chunked store the
        brackets are already in memory, so it is one.

        It is two rather than one because the second wave's byte ranges are computed from the
        first's bytes. No arrangement of this format removes that; it is what `indptr` is for.

        Order is the caller's, never the sorted order the bytes were fetched in. A position may
        repeat: a batch of ids from a join legitimately does, refusing would make the caller dedupe
        and then re-expand the very work this does to coalesce, and a selection holding a slice
        twice is a perfectly good selection. An empty batch is an empty selection rather than a
        refusal, because that is what an empty filter returns.

        Returns a :class:`Selection`, which carries the positions along with the bytes -- see there
        for why that is not a :class:`~sporadik.Layout`.
        """
        wanted = np.asarray(positions, dtype=np.intp).reshape(-1)
        if wanted.size and (int(wanted.min()) < 0 or int(wanted.max()) >= self.info.slices):
            off = int(wanted.min()) if int(wanted.min()) < 0 else int(wanted.max())
            raise IndexError(
                f"{off} is not a position along axis {self.info.indexed_axis} of {self.info.shape}, which this "
                f"layout compresses ({self.info.slices} slices)."
            )

        low, high = self._edges_at(wanted)
        low, high = np.asarray(low).astype(np.int64), np.asarray(high).astype(np.int64)
        counts = high - low
        run_lo, run_hi = _coalesce(low, high)

        if run_lo.size == 0:
            indices = np.empty(0, dtype=self._group["indices"].dtype)
            data = np.empty(0, dtype=self._group["data"].dtype)
            indptr = np.zeros(wanted.size + 1, dtype=np.int64)
        elif self.info.range_readable:
            runs = [(int(a), int(b)) for a, b in zip(run_lo, run_hi)]
            fetched = self._read_ranges(
                [("indices", a, b) for a, b in runs] + [("data", a, b) for a, b in runs]
            )
            index_buffer = np.concatenate(fetched[: len(runs)])
            data_buffer = np.concatenate(fetched[len(runs) :])
            offsets = np.concatenate(([0], np.cumsum(run_hi - run_lo)))[:-1]
            which = np.searchsorted(run_lo, low, side="right") - 1
            indptr, take = _ragged_take(offsets[which] + (low - run_lo[which]), counts)
            indices, data = index_buffer[take], data_buffer[take]
        elif run_lo.size == 1:
            # One run is a *basic slice*, and the difference from a fancy index is not cosmetic: a
            # fancy index over a contiguous selection builds an eight-byte offset per nonzero to
            # read a range that `[lo:hi]` reads directly. Without this branch `slices_at([p])` would
            # cost more than the `slice_at(p)` it replaces.
            start, stop = int(run_lo[0]), int(run_hi[0])
            index_buffer = self._group["indices"][start:stop]
            data_buffer = self._group["data"][start:stop]
            indptr, take = _ragged_take(low - start, counts)
            indices, data = index_buffer[take], data_buffer[take]
        else:
            # Several runs: one fancy index in global element space, which zarr answers by fetching
            # each touched chunk once, concurrently -- where reading the runs one at a time refetches
            # every chunk two neighbouring runs share.
            indptr, take = _ragged_take(low, counts)
            indices, data = self._group["indices"][take], self._group["data"][take]

        return Selection(
            positions=wanted, indptr=indptr, indices=np.asarray(indices), data=np.asarray(data),
            shape=self.info.shape, indexed_axis=self.info.indexed_axis, index_order=self.info.index_order,
        )

    def slices_over(self, start: int, stop: int) -> Selection:
        """The slices at positions ``[start:stop)`` -- one contiguous run, one range read.

        Exactly ``slices_at(range(start, stop))``, and kept as its own method because the guarantee
        is invisible at the call site otherwise: a contiguous run of positions is *already* one
        contiguous byte range in this format, from ``indptr[start]`` to ``indptr[stop]``, so nothing
        is coalesced, nothing is over-read and there is no fancy index.

        Not spelled ``slice_at(slice(start, stop))``. A parameter whose type decides the return type
        is a method that has to be read twice at every call site to know what came back.
        """
        return self.slices_at(range(start, stop))

    def dense_slices(self, positions: Sequence[int] | npt.NDArray[Any]) -> npt.NDArray[Any]:
        """A batch of slices scattered into ``(len(positions), *index_shape)``.

        :meth:`dense_slice` for many, and the shape a caller feeding a model a batch wants. Dense
        over the uncompressed axes only; the compressed axis stays the selection, in the order asked
        for.
        """
        return self.slices_at(positions).to_dense()

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
        """The largest absolute value of every slice.

        One small read when the store carries the answer, which one written by this package now
        does: the writer reduces once, at ingest, and stores a float per slice. This module said to
        do exactly that -- 1.8 s over the 88 M nonzeros of a 16 um matrix against 0.07 s for the same
        reduction in memory -- while giving no way to, so the advice cost every caller a side
        channel of its own.

        The fallback is the reduction, and it is not deprecated: a store written before the array
        existed is a legal store, and reducing over every value is the only honest answer for one.
        Both go through :func:`~sporadik.maxima_of`, so the stored answer and the computed one
        cannot be two different numbers.
        """
        if self.info.has_maxima:
            return np.asarray(self._group["maxima"][:])
        edges = self._indptr if self._indptr is not None else self._group["indptr"][:]
        return maxima_of(self._group["data"][:], np.asarray(edges))

    def to_layout(self) -> Layout:
        """This whole layout, read back into a :class:`~sporadik.Layout`.

        The opposite of what the rest of this class is for -- it fetches every value, where every
        other method exists to avoid that -- and it is here because two callers need it and both
        were otherwise reaching past the reader to zarr: reading a store back into memory, and
        re-compressing one layout into another axis. A reader answering one question should not
        call it.
        """
        return layout_over(
            self.info.shape,
            self.info.indexed_axis,
            data=np.asarray(self._group["data"][:]),
            indices=np.asarray(self._group["indices"][:]),
            indptr=np.asarray(self._group["indptr"][:]),
            index_order=self.info.index_order,
        )

    def close(self) -> None:
        """Drop the cached ``indptr`` and refuse further reads.

        The cache is emptied rather than set to `None`, which would not mean "closed" but "byte
        addressable" -- and would route every later read at byte offsets into compressed multi-chunk
        arrays, returning real, wrong numbers. The flag is what says closed, because an empty cache
        on its own only produced `IndexError` from numpy about an array of size 0.
        """
        self._indptr = np.empty(0, dtype=np.int64)
        self._closed = True


@contextmanager
def open_store(path: Path | str, axis: int | None = None) -> Generator[SparseReader]:
    """A :class:`SparseReader` over the layout compressing ``axis``, closed on the way out."""
    reader = SparseReader(path, axis)
    try:
        yield reader
    finally:
        reader.close()


def read_layout(path: Path | str, axis: int | None = None) -> MatrixLike:
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

    reader = SparseReader(path, axis)
    info = reader.info
    if info.rank != MIN_RANK:
        raise SporadikError(
            f"{path} is a rank-{info.rank} array, and `scipy.sparse` has no rank-{info.rank} matrix to return. Read it "
            "a slice at a time with open_store(), which unravels the positions for you."
        )
    group = _open(path)[info.path]
    builder = sp.csr_matrix if info.encoding == "csr_matrix" else sp.csc_matrix
    return builder((group["data"][:], group["indices"][:], group["indptr"][:]), shape=info.shape)
