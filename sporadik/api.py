"""The convenience layer: one object that holds an array's layouts, and one that opens a store.

Everything here is composed out of :mod:`sporadik.layout`, :mod:`sporadik.writer` and
:mod:`sporadik.reader`, and nothing here is normative. :mod:`sporadik.spec` is still the format; a
second implementation reproduces that module and none of this one.

What it adds is the part every caller was writing itself. A layout is *one axis made contiguous*,
so an array that answers two questions is two layouts -- and building the second one from the first
is the same handful of lines every time, spelled ``.tocsc()`` at rank two and a ravel-per-axis loop
above it. :class:`SparseArray` holds the set, the converters build it from whatever the values came
in as, and :class:`SparseStore` opens a written one and keeps a reader per axis::

    import sporadik

    array = sporadik.SparseArray.from_matrix(counts)       # both layouts, from one matrix
    array.write("expression.zarr")

    with sporadik.open_array("expression.zarr") as store:
        positions, values = store.slice_at(7, axis=1)      # the axis is still named

Above rank two nothing about the call site changes -- which is the point, because that is where
hand-written layout code went wrong::

    array = sporadik.SparseArray.from_coords(shape, (cells, metabolites, adducts), intensity)
    array.write("spacem.zarr")                             # three layouts, one per axis

A :class:`SparseArray` is accepted anywhere a list of matrices is: :func:`sporadik.layouts_of`
takes it, so :func:`sporadik.write_store` and any consumer that funnels through it -- the mikro
client's ``store=`` among them -- need no special case.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import zarr

from sporadik.errors import LayoutError, SporadikError
from sporadik.layout import Layout, MatrixLike, layout_of, layout_over, layouts_of
from sporadik.reader import Selection, SparseReader, StoreInfo, ZarrNode, describe, read_layout
from sporadik.spec import BLOCK_KEY, ENCODINGS, LAYOUTS_GROUP, MIN_RANK, block_for
from sporadik.writer import write_layout_into, write_store, write_store_into

__all__ = ["Duplicates", "SparseArray", "SparseStore", "as_sparse_array", "open_array"]

#: What to do with two nonzeros at the same coordinate.
#:
#: ``"sum"`` is what ``scipy.sparse.csr_matrix((values, (rows, cols)))`` does, silently, and is the
#: default here for that reason: a caller moving off that construction gets the same numbers.
#: ``"refuse"`` is for a caller who believes its coordinates are already unique and wants to find
#: out when they are not -- because summed duplicates are indistinguishable from larger values.
Duplicates = Literal["sum", "refuse"]

#: Both, not just ``.format`` -- an ordinary `str` has a ``.format``, and dispatching on that alone
#: sends one into the matrix branch to fail there about a missing ``.data``. A `scipy.sparse` matrix
#: is what carries a format *and* knows how to become a compressed one, COO and LIL included.
_MATRIX_ATTRS = ("format", "tocsr", "shape")



def _index_dtype(largest: int, nnz: int) -> npt.DTypeLike:
    """The width `indices` and `indptr` are written at.

    Matches what `scipy` chooses -- 32 bits until an index or an offset would not fit -- so a store
    built here and one built through scipy hold the same bytes rather than merely the same numbers.
    """
    return np.int32 if max(largest, nnz) < 2**31 else np.int64


def _compress(
    shape: tuple[int, ...],
    coords: tuple[npt.NDArray[Any], ...],
    values: npt.NDArray[Any],
    indexed_axis: int,
    duplicates: Duplicates,
) -> Layout:
    """One layout, built from coordinates by making ``indexed_axis`` contiguous.

    This is the whole of what a layout *is*, in code: ravel every other axis into one position,
    sort by ``(indexed_axis, that position)`` so each slice is one run with its indices ascending,
    and count the runs into ``indptr``. numpy only -- the rank-two case does not become the general
    case by going through `scipy`, it becomes the general case by folding more axes into the ravel.
    """
    order = tuple(axis for axis in range(len(shape)) if axis != indexed_axis)
    index_shape = tuple(shape[axis] for axis in order)
    raveled = np.ravel_multi_index(tuple(coords[axis] for axis in order), index_shape)

    walk = np.lexsort((raveled, coords[indexed_axis]))
    positions = np.asarray(coords[indexed_axis])[walk]
    indices = np.asarray(raveled)[walk]
    data = np.asarray(values)[walk]

    repeated = np.zeros(len(data), dtype=bool)
    if len(data) > 1:
        repeated[1:] = (positions[1:] == positions[:-1]) & (indices[1:] == indices[:-1])
    if repeated.any():
        if duplicates == "refuse":
            raise LayoutError(
                f"{int(repeated.sum())} of {len(data)} coordinates name a cell another one already names. A layout "
                "holds one value per cell, so this is either a duplicate to drop or two measurements to combine -- "
                "pass duplicates='sum' to add them together, which is what scipy does."
            )
        starts = np.flatnonzero(~repeated)
        data = np.add.reduceat(data, starts)
        indices = indices[starts]
        positions = positions[starts]

    counts = np.bincount(positions.astype(np.intp, copy=False), minlength=shape[indexed_axis])
    indptr = np.concatenate(([0], np.cumsum(counts)))
    width = _index_dtype(int(np.prod(index_shape)), len(data))
    return layout_over(
        shape,
        indexed_axis,
        data=data,
        indices=indices.astype(width, copy=False),
        indptr=indptr.astype(width, copy=False),
        index_order=order,
    )


def _wanted(shape: tuple[int, ...], axes: Sequence[int] | None) -> tuple[int, ...]:
    """The axes to build a layout for, defaulting to every one of them.

    Every axis is the default because it is what makes a store answer every question it *could*
    answer, and because a caller who has not thought about it has not chosen the other thing. Each
    layout costs another copy of the nonzeros, so ``axes=(1,)`` is how you say you only ever select
    along one -- a choice, made explicitly, rather than a default that quietly halves the store.
    """
    if axes is None:
        return tuple(range(len(shape)))
    wanted = tuple(int(axis) for axis in axes)
    if not wanted:
        raise LayoutError("A sparse store is its layouts; there is no state in which one exists and holds none.")
    for axis in wanted:
        if not 0 <= axis < len(shape):
            raise LayoutError(f"axis {axis} is not an axis of shape {shape}, so nothing could compress along it")
    return wanted


def _only(built: dict[int, Layout], axes: Sequence[int] | None) -> dict[int, Layout]:
    """``built``, narrowed to ``axes`` -- refusing an axis that is not in it.

    The counterpart of :func:`_wanted` for layouts that already exist. `_wanted` says which axes to
    *build*; this says which of the built ones to keep, and the difference matters because an axis
    absent here cannot be conjured from a layout of another axis without going back to the
    coordinates. Silently returning every axis, which is what this argument used to do on these
    paths, is the one answer that is never what was asked for.
    """
    if axes is None:
        return built
    wanted = tuple(int(axis) for axis in axes)
    if not wanted:
        raise LayoutError("A sparse store is its layouts; there is no state in which one exists and holds none.")
    for axis in wanted:
        if axis not in built:
            raise LayoutError(
                f"These layouts compress {tuple(sorted(built))}, not axis {axis}. They are already built, so this "
                "cannot compress another one for you -- build it from the coordinates, or from a SparseArray with "
                f"with_axis({axis})."
            )
    return {axis: built[axis] for axis in sorted(wanted)}


def _sparse_group(source: MatrixLike) -> MatrixLike | None:
    """The anndata-spelled sparse group behind ``source``, if there is one.

    Accepts the group itself, or anything holding one under ``X`` -- an `AnnData`, or the zarr or
    HDF5 group an ``.h5ad``/``.zarr`` table was written as. Read through the mapping rather than
    through `anndata`, because the three arrays and the encoding are all that is wanted and every
    store that holds them can be indexed.
    """
    attrs = getattr(source, "attrs", None)
    if attrs is not None and str(dict(attrs).get("encoding-type", "")) in ENCODINGS:
        return source
    if isinstance(source, Mapping) or hasattr(source, "keys"):
        try:
            return source["X"]  # pyright: ignore[reportIndexIssue]
        except (KeyError, TypeError):
            return None
    return None


@dataclass(frozen=True, eq=False)
class SparseArray:
    """One array, in up to one layout per axis -- what a writer holds before it writes.

    The layouts are keyed by the axis each makes contiguous, which is the only key they have: two
    layouts over the same axis are one capability twice, and two shapes are two arrays. Both are
    refused where the layouts are collected, so an instance of this is a set that could be written.

    Build one with a converter rather than by hand. :meth:`from_matrix` takes a `scipy` CSR or CSC
    and produces every layout from it; :meth:`from_coords` takes coordinates at any rank;
    :meth:`from_anndata` takes an ``X``; :meth:`from_dense` takes an ordinary array. All of them
    default to a layout per axis, and all of them accept ``axes=`` to buy fewer.
    """

    #: The layouts to write, keyed by the axis each makes contiguous. A read-only mapping: the
    #: refusals below are what make an instance one that could be written, and a dict handed back
    #: could be reached around and added to afterwards, which would make them a formality.
    layouts: Mapping[int, Layout]

    def __post_init__(self) -> None:
        """Put the layouts through the same refusals a list of matrices would be put through.

        The class docstring says an instance of this is a set that could be written. That was only
        true of instances built by the converters, which all funnel through :func:`layouts_of`;
        ``SparseArray({})`` went around it, and then `shape`, `nnz` and `rank` all raised
        `StopIteration` off an empty iterator -- the worst available failure, because raised inside
        a generator it does not propagate, it silently ends the generator.

        Re-keying is part of it: a layout filed under an axis it does not compress ends up under the
        one it does, because :func:`layouts_of` keys by what the arrays actually say.
        """
        object.__setattr__(self, "layouts", MappingProxyType(layouts_of(list(self.layouts.values()))))

    def __eq__(self, other: object) -> bool:
        """The same axes holding the same layouts. See :meth:`Layout.__eq__` for why by hand."""
        if not isinstance(other, SparseArray):
            return NotImplemented
        return self.indexed_axes == other.indexed_axes and all(
            self.layouts[axis] == other.layouts[axis] for axis in self.indexed_axes
        )

    #: Unhashable, for :class:`Layout`'s reason and one more: the field is a mapping, and the hash
    #: `frozen=True` generates raises `TypeError` on it before ever reaching the arrays.
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    # ------------------------------------------------------------------ converters

    @classmethod
    def from_layouts(cls, layouts: MatrixLike, *, axes: Sequence[int] | None = None) -> SparseArray:
        """From layouts that are already built: one, or any iterable of them.

        ``axes`` selects among them rather than building any: these are already compressed, and an
        axis that is not among them cannot be produced here without the coordinates. Asking for one
        is refused rather than quietly ignored.
        """
        built = layouts_of(layouts if isinstance(layouts, Layout) else list(layouts))
        return cls(_only(built, axes))

    @classmethod
    def from_matrices(cls, matrices: MatrixLike, *, axes: Sequence[int] | None = None) -> SparseArray:
        """From the matrices as they are, adding nothing.

        The literal form of what callers wrote before this class existed -- ``[counts,
        counts.tocsc()]`` -- so migrating to it changes no bytes. :meth:`from_matrix` is the one
        that does the ``.tocsc()`` for you.

        ``axes`` selects among the matrices given, for :meth:`from_layouts`' reason.
        """
        return cls(_only(layouts_of(matrices), axes))

    @classmethod
    def from_matrix(
        cls, matrix: MatrixLike, *, axes: Sequence[int] | None = None, duplicates: Duplicates = "sum"
    ) -> SparseArray:
        """From one `scipy` CSR or CSC matrix, re-compressed along every axis asked for.

        The matrix states which axis it already compresses through its own ``.format``, and the
        others are built from it -- with ``.tocsc()``/``.tocsr()`` at rank two, where scipy has a
        spelling for the answer, and through the coordinates above it, where it has none.

        Which axes you ask for is the decision this format is about: ``.tocsc()`` over an
        ``(objects, features)`` matrix makes one *feature* contiguous, ``.tocsr()`` one *object*,
        and asking a layout the question it does not compress is a scan of every byte rather than a
        slower read. The default is every axis, which answers both.

        A COO or LIL matrix is converted on the way in rather than refused. :func:`layout_of`
        refuses one, correctly -- it builds *one* layout, and which axis that compresses is a
        decision COO does not carry. Here the axes are the argument, so there is nothing left to
        guess at.

        ``duplicates`` reaches the COO case only, because it is the only one that can hold any: a
        matrix that is already CSR or CSC has had them summed by whatever built it. Passing
        ``"refuse"`` routes the conversion through :meth:`from_coords` rather than through scipy's
        ``.tocsr()``, which sums them silently -- and silently summed duplicates are
        indistinguishable from larger values.
        """
        if duplicates == "refuse" and str(getattr(matrix, "format", "")) not in ("csr", "csc"):
            coo = matrix.tocoo()
            return cls.from_coords(
                tuple(int(size) for size in coo.shape), (coo.row, coo.col), coo.data, axes=axes, duplicates="refuse"
            )
        base = layout_of(_as_compressed(matrix))
        wanted = _wanted(base.shape, axes)
        built: dict[int, Layout] = {base.indexed_axis: base}
        for axis in wanted:
            if axis in built:
                continue
            other = _recompress_with_scipy(matrix, axis)
            if other is not None:
                built[axis] = layout_of(other)
            else:
                built = dict(cls(built).with_axis(axis).layouts)
        return cls({axis: built[axis] for axis in sorted(wanted)})

    @classmethod
    def from_coords(
        cls,
        shape: Sequence[int],
        coords: Sequence[npt.NDArray[Any]],
        values: npt.NDArray[Any],
        *,
        axes: Sequence[int] | None = None,
        duplicates: Duplicates = "sum",
    ) -> SparseArray:
        """From coordinates and values -- the COO form -- at any rank.

        ``coords`` is one array per axis of ``shape``, each as long as ``values``. This is the
        general constructor and the only one that needs no other representation to exist first:
        above rank two `scipy` has no matrix to hand, and the ravel this does per axis is precisely
        what a layout at rank three *is*.

        ``duplicates`` decides what two values at one coordinate mean; see :data:`Duplicates`.
        """
        shape = tuple(int(size) for size in shape)
        if len(shape) < MIN_RANK:
            raise LayoutError(f"a sparse array has at least {MIN_RANK} axes, but shape is {shape}")
        if len(coords) != len(shape):
            raise LayoutError(
                f"`coords` holds {len(coords)} coordinate arrays for a shape of {len(shape)} axes. There is one per "
                "axis, in axis order, each as long as `values`."
            )
        values = np.asarray(values)
        for axis, coordinate in enumerate(coords):
            if len(coordinate) != len(values):
                raise LayoutError(
                    f"the coordinates along axis {axis} number {len(coordinate)} against {len(values)} values; they "
                    "are parallel -- one coordinate per value, per axis."
                )
        prepared = tuple(np.asarray(coordinate) for coordinate in coords)
        for axis, coordinate in enumerate(prepared):
            # Checked here rather than left to what happens downstream, because what happens
            # downstream is two different silences. `np.ravel_multi_index` catches an out-of-range
            # coordinate on the axes it ravels, but reports it as "invalid entry in coordinates
            # array" -- naming neither the axis nor the value. On the axis being *compressed* it is
            # not caught at all: `np.bincount(..., minlength=)` returns a longer array, and the
            # result is a SparseArray whose `shape` lies, refused much later by `validate_layout`
            # with a message about `indptr` length that says nothing about the coordinate.
            if coordinate.size and (int(coordinate.min()) < 0 or int(coordinate.max()) >= shape[axis]):
                offender = int(coordinate.min()) if int(coordinate.min()) < 0 else int(coordinate.max())
                raise LayoutError(
                    f"the coordinates along axis {axis} include {offender}, which is not a position along an axis of "
                    f"{shape[axis]}. Every coordinate names a cell of the array it declares."
                )
        return cls({axis: _compress(shape, prepared, values, axis, duplicates) for axis in _wanted(shape, axes)})

    @classmethod
    def from_dense(cls, array: npt.NDArray[Any], *, axes: Sequence[int] | None = None) -> SparseArray:
        """From an ordinary dense array, keeping the cells that are not zero.

        The package's own test suite builds its rank-three fixtures this way and its helper says,
        deliberately, that it lives in the test rather than here -- because a caller writing a store
        already holds its values in whatever form they came in. That held while there was nothing
        for a converter to live in. With one, refusing the commonest in-memory form would be
        arbitrary, so this reverses that decision knowingly.

        It is still the wrong entry point for anything large: a dense array of the size these stores
        are for does not fit in memory, which is what :meth:`from_coords` is for.
        """
        array = np.asarray(array)
        coords = np.nonzero(array)
        return cls.from_coords(array.shape, coords, array[coords], axes=axes, duplicates="refuse")

    @classmethod
    def from_anndata(cls, source: MatrixLike, *, axes: Sequence[int] | None = None) -> SparseArray:
        """From an anndata ``X``: an `AnnData`, a zarr or HDF5 group holding one, or the group itself.

        A layout *is* anndata's spelling -- ``data``, ``indices``, ``indptr`` and an
        ``encoding-type`` -- so an on-disk ``X`` is read straight into a layout without rehydrating
        a `scipy` matrix in between, and without importing `anndata`. What sporadik adds is the
        second layout, which anndata has no way to hold.
        """
        held = getattr(source, "X", None)
        if held is not None:
            return as_sparse_array(held, axes=axes)

        group = _sparse_group(source)
        if group is None:
            raise TypeError(
                f"An anndata source is an AnnData, a group holding one under 'X', or a sparse group itself -- "
                f"something with an 'encoding-type' of {sorted(ENCODINGS)} in its attributes. "
                f"{type(source).__name__} has neither .X nor an X."
            )

        attrs = dict(group.attrs)
        encoding = str(attrs.get("encoding-type", ""))
        if encoding not in ENCODINGS:
            raise LayoutError(
                f"That group declares encoding-type {encoding!r}, which is not a sparse spelling this reads. A dense X "
                "is an ordinary array: pass it to from_dense()."
            )
        shape = tuple(int(size) for size in attrs["shape"])
        base = layout_over(
            shape,
            ENCODINGS[encoding],
            data=np.asarray(group["data"][:]),
            indices=np.asarray(group["indices"][:]),
            indptr=np.asarray(group["indptr"][:]),
        )
        built = cls({base.indexed_axis: base})
        for axis in _wanted(shape, axes):
            built = built.with_axis(axis)
        return cls({axis: built.layouts[axis] for axis in sorted(_wanted(shape, axes))})

    # ------------------------------------------------------------------ what it holds

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's shape, which every layout agrees on -- that is what makes them one array."""
        return next(iter(self.layouts.values())).shape

    @property
    def rank(self) -> int:
        """How many axes the array has."""
        return len(self.shape)

    @property
    def nnz(self) -> int:
        """How many nonzeros the array holds -- once, not once per layout."""
        return len(next(iter(self.layouts.values())).data)

    @property
    def indexed_axes(self) -> tuple[int, ...]:
        """The axes this array can be selected along in one contiguous read."""
        return tuple(sorted(self.layouts))

    def layout(self, axis: int) -> Layout:
        """The layout compressing ``axis``, or a refusal naming the ones there are."""
        held = self.layouts.get(int(axis))
        if held is None:
            raise LayoutError(
                f"This array compresses {self.indexed_axes}, not axis {axis}. Add it with with_axis({axis}), which "
                "costs another copy of the nonzeros, or select along an axis it already compresses."
            )
        return held

    def with_axis(self, axis: int) -> SparseArray:
        """The same array, also compressed along ``axis``.

        Re-compression goes through the coordinates rather than through `scipy`, so it works at
        every rank: ``.tocsc()`` spells one answer and only at rank two, while unravelling a layout
        back to coordinates and compressing again spells all of them. Returns a new array; nothing
        here mutates.
        """
        axis = int(axis)
        if axis in self.layouts:
            return self
        if not 0 <= axis < self.rank:
            raise LayoutError(f"axis {axis} is not an axis of shape {self.shape}, so nothing could compress along it")
        coords, values = self.to_coords()
        return SparseArray({**self.layouts, axis: _compress(self.shape, coords, values, axis, "refuse")})

    def to_coords(self) -> tuple[tuple[npt.NDArray[Any], ...], npt.NDArray[Any]]:
        """Back to coordinates and values: one array per axis, plus the values, in axis order.

        The inverse of the compression, and the exchange format between layouts. The compressed
        axis's coordinate comes out of ``indptr`` -- one repeat per entry in each run -- and the
        rest unravel through ``index_order``, which is why that has to be recorded.
        """
        held = next(iter(self.layouts.values()))
        counts = np.diff(np.asarray(held.indptr))
        along = np.repeat(np.arange(held.shape[held.indexed_axis]), counts)
        index_shape = tuple(held.shape[axis] for axis in held.index_order)
        rest = np.unravel_index(np.asarray(held.indices), index_shape)

        coords: list[npt.NDArray[Any]] = [np.empty(0, dtype=np.int64)] * held.rank
        coords[held.indexed_axis] = along
        for position, axis in enumerate(held.index_order):
            coords[axis] = rest[position]
        return tuple(coords), np.asarray(held.data)

    def to_dense(self) -> npt.NDArray[Any]:
        """Back to an ordinary dense array, zeros and all -- the inverse of :meth:`from_dense`.

        Carries that method's warning in the other direction and more sharply: this allocates
        ``prod(shape)`` cells whatever the density, so an array of the size these stores are for
        does not fit in memory. :meth:`to_coords` is the one that scales.
        """
        coords, values = self.to_coords()
        dense = np.zeros(self.shape, dtype=values.dtype)
        dense[coords] = values
        return dense

    def to_scipy(self, *, axis: int) -> MatrixLike:
        """One layout as a `scipy.sparse` matrix. Rank two only, because scipy has nothing above it."""
        try:
            import scipy.sparse as sp
        except ModuleNotFoundError as missing:  # pragma: no cover - depends on the environment
            raise ModuleNotFoundError(
                "to_scipy returns a scipy.sparse matrix, so it needs scipy: pip install sporadik[scipy]. The layouts "
                "themselves are plain numpy arrays and need nothing."
            ) from missing

        held = self.layout(axis)
        if held.rank != MIN_RANK:
            raise SporadikError(
                f"This is a rank-{held.rank} array, and `scipy.sparse` has no rank-{held.rank} matrix to return. Read "
                "it a slice at a time, or take the coordinates with to_coords()."
            )
        builder = sp.csr_matrix if held.encoding == "csr_matrix" else sp.csc_matrix
        return builder((held.data, held.indices, held.indptr), shape=held.shape)

    def __iter__(self) -> Iterator[Layout]:
        """The layouts, in axis order -- so ``list(array)`` is what the writer takes."""
        return iter(layout for _, layout in sorted(self.layouts.items()))

    def __len__(self) -> int:
        """How many layouts this array holds, which is how many questions it answers cheaply."""
        return len(self.layouts)

    def __repr__(self) -> str:
        """Shape, nonzeros and which axes are contiguous -- the three things that decide a read."""
        axes = "+".join(f"axis{axis}" for axis in self.indexed_axes)
        return f"SparseArray(shape={self.shape}, nnz={self.nnz}, {axes})"

    # ------------------------------------------------------------------ what it does

    def write(self, path: Path | str, *, chunk: int | None = None, byte_addressable: bool = False) -> Path:
        """Write these layouts as a sparse store at ``path``."""
        return write_store(path, self, chunk=chunk, byte_addressable=byte_addressable)

    def write_into(self, group: MatrixLike, *, chunk: int | None = None, byte_addressable: bool = False) -> MatrixLike:
        """Write these layouts into an already-opened zarr group, and return it."""
        return write_store_into(group, self, chunk=chunk, byte_addressable=byte_addressable)


def _as_compressed(matrix: MatrixLike) -> MatrixLike:
    """``matrix`` in a compressed spelling, converting a COO or LIL rather than refusing it.

    Untouched unless it has to be: a matrix that is already CSR or CSC is handed straight back, so
    the common path allocates nothing.
    """
    if str(getattr(matrix, "format", "")) in ("csr", "csc"):
        return matrix
    convert = getattr(matrix, "tocsr", None)
    return matrix if convert is None else convert()


def _recompress_with_scipy(matrix: MatrixLike, axis: int) -> MatrixLike | None:
    """``matrix`` compressed along ``axis`` by scipy, or ``None`` if scipy cannot say it.

    Rank two is the only place scipy has a name for the answer, and there it is one call over an
    implementation in C. Above it there is no CSR-of-three-axes, so the caller falls back to the
    coordinates -- which is the general path, not a worse one.
    """
    shape = tuple(int(size) for size in matrix.shape)
    if len(shape) != MIN_RANK:
        return None
    convert = getattr(matrix, "tocsr" if axis == 0 else "tocsc", None)
    return None if convert is None else convert()


def as_sparse_array(
    source: MatrixLike, *, axes: Sequence[int] | None = None, duplicates: Duplicates = "sum"
) -> SparseArray:
    """Whatever was passed, as a :class:`SparseArray`.

    Dispatches on what the value carries rather than on what it is, exactly as :func:`layout_of`
    does, so neither `scipy` nor `anndata` has to be importable for the branch that does not need
    it. A :class:`SparseArray` passes through; a :class:`Layout` or a list of matrices is collected;
    anything with a ``.format`` is a scipy matrix; anything with an ``.X`` or an ``X`` is an
    anndata source; an ordinary array is dense.

    ``from_coords`` has no branch here on purpose -- coordinates are three arguments, not one value.

    ``axes`` reaches every branch, including the two built from layouts that already exist, where it
    selects rather than builds. It used to reach only three of them and be dropped without a word on
    the rest, which is how ``as_sparse_array(layouts, axes=(1,))`` came back holding every axis.
    ``duplicates`` reaches the branches that can hold any; see :data:`Duplicates`.
    """
    if isinstance(source, SparseArray):
        # Handed straight back when nothing was asked of it, because it is already the answer and
        # callers rely on that identity. `axes` is the one thing that can still narrow it.
        return source if axes is None else SparseArray(_only(dict(source.layouts), axes))
    if isinstance(source, Layout):
        return SparseArray.from_layouts(source, axes=axes)
    if isinstance(source, (list, tuple)):
        return SparseArray.from_matrices(source, axes=axes)
    if all(hasattr(source, name) for name in _MATRIX_ATTRS):
        return SparseArray.from_matrix(source, axes=axes, duplicates=duplicates)
    if getattr(source, "X", None) is not None or _sparse_group(source) is not None:
        return SparseArray.from_anndata(source, axes=axes)
    if isinstance(source, np.ndarray):
        return SparseArray.from_dense(source, axes=axes)
    raise TypeError(
        f"There is no sparse array in a {type(source).__name__}. This takes a scipy CSR or CSC matrix, a list of "
        "them, a sporadik.Layout, an anndata source (an AnnData or a group holding X), or a dense numpy array. "
        "Coordinates go to SparseArray.from_coords(shape, coords, values), which cannot be dispatched on."
    )


class SparseStore:
    """A store opened from a prefix: what it says about itself, and one reader per axis.

    :class:`SparseReader` reads one layout, which is the right unit for the format and the wrong one
    for a caller with two questions -- it opens the prefix twice and describes it twice. This holds
    one description and a reader per axis, opened when first asked for.

    **The axis is still named at the call site.** ``axis=None`` resolves only when the store holds
    exactly one layout, where there is no decision to make; with more than one it raises, because
    which layout to read *is* the whole of what they differ in. That refusal is the reader's, and
    wrapping it must not become defaulting it.

    Every read here is synchronous, and on a byte-addressable store the underlying range read calls
    ``asyncio.run`` -- so it raises inside a running event loop. Nothing in this package is async;
    a caller that is should read in a worker thread.
    """

    def __init__(self, path: Path | str) -> None:
        """Open the store at ``path``, reading only what it says about itself."""
        self.path = Path(path)
        self.info: StoreInfo = describe(self.path)
        self._readers: dict[int, SparseReader] = {}

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's shape, as the store declares it."""
        return self.info.shape

    @property
    def rank(self) -> int:
        """How many axes the array has."""
        return self.info.rank

    @property
    def indexed_axes(self) -> tuple[int, ...]:
        """The axes this store answers in one contiguous read. Every other question is a scan."""
        return tuple(sorted(self.info.layouts))

    @property
    def nnz(self) -> int:
        """How many nonzeros the array holds -- once, not once per layout, as the store declares it."""
        return next(iter(self.info.layouts.values())).nnz

    @property
    def dtype(self) -> str:
        """The dtype of the values, as the store declares it."""
        return next(iter(self.info.layouts.values())).dtype

    def __len__(self) -> int:
        """How many layouts the store holds, which is how many questions it answers cheaply."""
        return len(self.info.layouts)

    def _axis(self, axis: int | None) -> int:
        """Which layout a call meant, refusing to choose when the choice is the caller's."""
        if axis is not None:
            if int(axis) not in self.info.layouts:
                raise LayoutError(
                    f"{self.path} compresses {self.indexed_axes}, not axis {axis}. Reading along an axis no layout "
                    "compresses is not slower, it is a scan of every byte rather than one contiguous range."
                )
            return int(axis)
        if len(self.info.layouts) != 1:
            raise LayoutError(
                f"{self.path} compresses axes {self.indexed_axes}, so which one to read is a decision: it is the "
                "whole of what they differ in. Pass `axis=<axis>`."
            )
        return next(iter(self.info.layouts))

    def along(self, axis: int | None = None) -> SparseReader:
        """The reader for one layout, opened once and kept."""
        chosen = self._axis(axis)
        if chosen not in self._readers:
            self._readers[chosen] = SparseReader(self.path, chosen)
        return self._readers[chosen]

    def bounds_at(self, position: int, *, axis: int | None = None) -> tuple[int, int]:
        """The two ``indptr`` entries bracketing one slice: where it starts and ends."""
        return self.along(axis).bounds_at(position)

    def slice_at(self, position: int, *, axis: int | None = None) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """The raveled positions and values of one slice along ``axis``."""
        return self.along(axis).slice_at(position)

    def coords_at(
        self, position: int, *, axis: int | None = None
    ) -> tuple[tuple[npt.NDArray[Any], ...], npt.NDArray[Any]]:
        """One slice as one coordinate array per uncompressed axis, plus the values."""
        return self.along(axis).coords_at(position)

    def slices_at(self, positions: Sequence[int] | npt.NDArray[Any], *, axis: int | None = None) -> Selection:
        """Many slices along ``axis``, in two waves rather than two per slice."""
        return self.along(axis).slices_at(positions)

    def slices_over(self, start: int, stop: int, *, axis: int | None = None) -> Selection:
        """The slices at positions ``[start:stop)`` along ``axis`` -- one contiguous run, one read."""
        return self.along(axis).slices_over(start, stop)

    def dense_slices(self, positions: Sequence[int] | npt.NDArray[Any], *, axis: int | None = None) -> npt.NDArray[Any]:
        """A batch of slices along ``axis``, dense over the uncompressed axes."""
        return self.along(axis).dense_slices(positions)

    def dense_slice(self, position: int, *, axis: int | None = None) -> npt.NDArray[Any]:
        """One slice scattered into a dense array over the uncompressed axes."""
        return self.along(axis).dense_slice(position)

    def maxima(self, *, axis: int | None = None) -> npt.NDArray[Any]:
        """The largest absolute value of every slice along ``axis``, in one pass over every value."""
        return self.along(axis).maxima()

    def to_scipy(self, *, axis: int | None = None) -> MatrixLike:
        """One whole layout as a `scipy.sparse` matrix. Rank two only.

        Keyword-only, like every other read here: the axis is the decision these methods are about,
        and a bare positional integer at the call site does not say which decision it is.
        """
        return read_layout(self.path, self._axis(axis))

    def to_array(self) -> SparseArray:
        """Every layout, read whole, back into a :class:`SparseArray`.

        Deliberately *not* routed through :meth:`along`: this asks for all of them, so resolving a
        single axis -- and refusing when there are two -- would be answering a question nobody
        asked. It also fetches everything, which is the opposite of what a reader is for; it is here
        for a round trip and for re-writing a store with another layout added, not for a read path.
        """
        return SparseArray({axis: SparseReader(self.path, axis).to_layout() for axis in self.indexed_axes})

    def add_axis(self, axis: int, *, chunk: int | None = None, byte_addressable: bool = False) -> SparseStore:
        """Write one more layout into this store, leaving the ones already there alone.

        Adding a third way to read a store used to mean :meth:`to_array` -- which fetches *every*
        value of *every* layout -- then ``with_axis``, then a full rewrite of all of them. The
        layouts are independent groups under ``layouts/``, so only the new one has to be written.
        The values still have to be read once, because re-compression needs every coordinate and no
        layout holds them in the order another one wants; what does not have to happen is rewriting
        the bytes that were already correct.

        **The block is rewritten last**, after the new group's chunks are durable -- the same
        ordering :mod:`sporadik.writer` argues for, and for the same reason: until the block names a
        layout, a half-written one is a group the store does not claim, rather than a store that
        reads zeros and raises nothing. An interrupted `add_axis` leaves the store exactly as it was.

        Returns ``self``, reopened: the description and every cached reader are dropped, because
        both describe a store that now holds one more layout than they were built from.
        """
        axis = int(axis)
        if axis in self.info.layouts:
            return self
        if not 0 <= axis < self.rank:
            raise LayoutError(f"axis {axis} is not an axis of shape {self.shape}, so nothing could compress along it")

        held = self.along(next(iter(self.indexed_axes))).to_layout()
        coords, values = SparseArray({held.indexed_axis: held}).to_coords()
        built = _compress(self.shape, coords, values, axis, "refuse")

        root: ZarrNode = zarr.open_group(str(self.path), mode="r+")
        write_layout_into(root[LAYOUTS_GROUP], built, chunk=chunk, byte_addressable=byte_addressable)
        layouts = {info.indexed_axis: info for info in self.info.layouts.values()}
        root.attrs[BLOCK_KEY] = block_for(self.shape, {**layouts, axis: built})

        self.close()
        self.info = describe(self.path)
        return self

    def close(self) -> None:
        """Drop every open reader and what it cached."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> Self:
        """Enter a ``with`` block; the store is already open."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close every reader on the way out."""
        self.close()

    def __repr__(self) -> str:
        """Shape and which axes are contiguous -- what decides whether a question is a read or a scan."""
        axes = "+".join(f"axis{axis}" for axis in self.indexed_axes)
        return f"SparseStore({self.path}, shape={self.shape}, {axes})"


def open_array(path: Path | str) -> SparseStore:
    """A :class:`SparseStore` over ``path``, usable directly or as a ``with`` block.

    The counterpart to :func:`sporadik.open_store`, which opens one layout: this opens the store and
    lets a caller ask along more than one axis without describing the prefix again.
    """
    return SparseStore(path)
