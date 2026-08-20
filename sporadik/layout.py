"""One layout: the three arrays, and which axis they make contiguous.

A *layout* is the unit this format is built from. It is not "a CSR matrix" -- that is the rank-two
case of it -- it is **one axis made contiguous**: pick the axis a reader will select along, store
the remaining axes raveled into ``indices``, and ``indptr`` then names a run per position along the
chosen one.

At rank two that construction is exactly CSR (axis 0) and CSC (axis 1), and the arrays are
byte-identical to what `scipy` and anndata produce. At rank three and above it is the same
construction with more axes folded into the same ``indices``, so nothing about a reader changes:
``indptr[i:i+2]`` still names one contiguous run, and the run's positions unravel through
:attr:`Layout.index_order` instead of being read directly.

One invariant holds at every rank and is the spine of the format::

    len(indptr) == shape[indexed_axis] + 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sporadik.errors import LayoutError
from sporadik.spec import ENCODINGS, MIN_RANK, anndata_encoding, layout_path, raveled_shape

__all__ = ["Layout", "layout_of", "layout_over", "layouts_of", "validate_layout"]

#: Anything three arrays and a shape can be read out of: a `scipy.sparse` CSR or CSC matrix, or a
#: :class:`Layout` built from the arrays directly. Duck-typed, so `scipy` need not be importable.
MatrixLike = Any


@dataclass(frozen=True)
class Layout:
    """The three arrays, the array's own shape, and which axis they make contiguous.

    :attr:`index_order` is the axes it did *not* compress, in the order ``indices`` was raveled over
    them. At rank two it has exactly one member and therefore says nothing; at rank three or more
    it is load-bearing and **not derivable from the bytes** -- a wrong one does not fail, it puts
    every value in a different cell, which is why it is stated and checked rather than assumed.
    """

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, ...]
    indexed_axis: int
    index_order: tuple[int, ...]

    @property
    def rank(self) -> int:
        """How many axes the array has. Two is one case of this, not the definition."""
        return len(self.shape)

    @property
    def encoding(self) -> str:
        """The ``encoding-type`` this layout's own group declares."""
        return anndata_encoding(self.rank, self.indexed_axis)

    @property
    def declared_shape(self) -> tuple[int, ...]:
        """The shape the layout's own group declares, which is not always the array's."""
        return raveled_shape(self.shape, self.indexed_axis, self.index_order)

    @property
    def path(self) -> str:
        """Where this layout sits inside the store's prefix."""
        return layout_path(self.indexed_axis)


def validate_layout(*, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: Sequence[int], indexed_axis: int) -> None:
    """Refuse arrays that contradict what they declare, before anything is written.

    A reader checks all of this too, and has to -- but it checks it *after* the bytes have moved,
    and a gigabyte is an expensive way to learn that ``indptr`` is the wrong length. The same rule
    stated on both sides, on purpose.
    """
    shape = tuple(int(size) for size in shape)
    if len(shape) < MIN_RANK:
        raise LayoutError(f"a sparse array has at least {MIN_RANK} axes, but shape is {shape}")
    if not 0 <= indexed_axis < len(shape):
        raise LayoutError(f"axis {indexed_axis} is not an axis of shape {shape}, so nothing could compress along it")
    expected = shape[indexed_axis] + 1
    if len(indptr) != expected:
        raise LayoutError(
            f"a layout compressing axis {indexed_axis} of {shape} has `indptr` of {expected} entries "
            f"(one per slice, plus the end), but it has {len(indptr)}."
        )
    if len(data) != len(indices):
        raise LayoutError(f"`data` has {len(data)} entries and `indices` {len(indices)}; they are parallel")


def layout_over(
    shape: Sequence[int],
    indexed_axis: int,
    *,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    index_order: Sequence[int] | None = None,
) -> Layout:
    """A :class:`Layout` at any rank, from the three arrays and the axis they compress.

    ``index_order`` defaults to the remaining axes in ascending order, which is what ``reshape``
    after ``transpose(indexed_axis, *others)`` produces and the only sensible default -- but it is
    recorded either way, because a reader cannot recover it from the bytes.
    """
    shape = tuple(int(size) for size in shape)
    others = tuple(axis for axis in range(len(shape)) if axis != indexed_axis)
    order = tuple(int(axis) for axis in index_order) if index_order is not None else others
    if sorted(order) != sorted(others):
        raise LayoutError(
            f"`index_order` must be the axes other than {indexed_axis}, in the order `indices` was raveled over -- "
            f"expected a permutation of {others}, got {order}."
        )
    return Layout(
        data=np.asarray(data),
        indices=np.asarray(indices),
        indptr=np.asarray(indptr),
        shape=shape,
        indexed_axis=int(indexed_axis),
        index_order=order,
    )


def layout_of(matrix: MatrixLike) -> Layout:
    """A :class:`Layout` from a rank-two CSR or CSC matrix, taking its axis from the encoding.

    Duck-typed rather than isinstance-checked, so `scipy` need not be importable here. The axis is
    read off ``.format`` and never passed in: the encoding *is* which axis ``indptr`` walks, and a
    second statement of it is free to disagree with the arrays underneath.
    """
    if isinstance(matrix, Layout):
        return matrix
    try:
        fmt = str(matrix.format)
        data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
        shape = tuple(int(size) for size in matrix.shape)
    except AttributeError as missing:
        raise TypeError(
            f"A layout comes from a CSR or CSC matrix -- something carrying .data, .indices, .indptr, .shape and "
            f".format -- but {type(matrix).__name__} has no {missing.name!r}. A COO or LIL matrix carries its "
            "nonzeros differently: convert with .tocsr() or .tocsc(), choosing by which axis a reader will select "
            "along. Above rank two, build one Layout per axis with layout_over()."
        ) from missing

    encoding = f"{fmt}_matrix"
    if encoding not in ENCODINGS:
        raise LayoutError(
            f"A {fmt!r} matrix has no anndata spelling this writes. Convert it with .tocsr() or .tocsc() first, "
            "choosing by which axis a reader will select along."
        )
    return layout_over(shape, ENCODINGS[encoding], data=data, indices=indices, indptr=indptr)


def layouts_of(matrices: MatrixLike) -> dict[int, Layout]:
    """The layouts to write, keyed by the axis each makes contiguous.

    Accepts one matrix or an iterable of them. What it refuses is the set that would make a store
    meaningless: two layouts compressing the same axis, which is one capability twice with nothing
    to say which a reader should use, and two different shapes, which is two arrays and therefore
    two stores.
    """
    candidates = matrices if isinstance(matrices, (list, tuple)) else [matrices]
    if not candidates:
        raise LayoutError("A sparse store is its layouts; there is no state in which one exists and holds none.")

    layouts: dict[int, Layout] = {}
    for candidate in candidates:
        layout = layout_of(candidate)
        if layout.indexed_axis in layouts:
            raise LayoutError(
                f"Two layouts compress axis {layout.indexed_axis}. That is one capability twice, and nothing could "
                "say which a reader should use. Give one per axis you need to select along."
            )
        layouts[layout.indexed_axis] = layout

    shapes = {layout.shape for layout in layouts.values()}
    if len(shapes) > 1:
        raise LayoutError(
            f"The layouts hold different shapes {sorted(shapes)}. A store is one array in up to one layout per axis; "
            "two different arrays are two stores. Did one of them get transposed rather than re-compressed?"
        )
    return layouts
