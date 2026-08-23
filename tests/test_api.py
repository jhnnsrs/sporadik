"""The convenience layer: that it builds the same layouts a caller would have built by hand.

Every assertion here is a comparison against the construction the converter replaces. That is the
only claim worth making about a convenience layer -- it is not a new way to write a store, it is
the same bytes with the boilerplate absorbed -- and it is the claim that fails if a ravel, an
`index_order` or an index dtype drifts.

The one place a converter is *not* byte-identical is stated in
`test_summed_duplicates_agree_with_scipy_to_float_rounding`, and the reason it cannot be is in that
test's docstring.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from sporadik import (
    Layout,
    LayoutError,
    SparseArray,
    SparseStore,
    SporadikError,
    as_sparse_array,
    describe,
    layout_over,
    layouts_of,
    open_array,
    read_layout,
    write_store,
)

#: Deliberately not square, for the reason `test_format` gives: a square matrix lets an axis
#: mix-up pass every shape check in the file.
ROWS, COLS, DENSITY = 400, 90, 0.05

#: Three different extents, for the same reason at rank three.
CUBE = (6, 4, 5)


@pytest.fixture
def matrix() -> Any:
    """One reproducible CSR matrix. Objects on axis 0, features on axis 1."""
    return sp.random(ROWS, COLS, density=DENSITY, format="csr", dtype=np.float32, random_state=0)


@pytest.fixture
def hand(matrix: Any) -> dict[int, Layout]:
    """The layouts a caller built before this module existed: the matrix, and its transpose-encoded self."""
    return layouts_of([matrix, matrix.tocsc()])


@pytest.fixture
def cube() -> np.ndarray:
    """A reproducible rank-three array that is mostly zeros."""
    rng = np.random.default_rng(0)
    return (rng.random(CUBE) * (rng.random(CUBE) < 0.3)).astype(np.float32)


def _assert_same(built: Layout, expected: Layout) -> None:
    """The three arrays and the ravel order, exactly -- not merely the same shape."""
    assert built.index_order == expected.index_order
    assert built.shape == expected.shape
    for name in ("data", "indices", "indptr"):
        assert np.array_equal(getattr(built, name), getattr(expected, name)), name
        assert getattr(built, name).dtype == getattr(expected, name).dtype, f"{name} dtype"


# --------------------------------------------------------------------------- #
# The converters build what a caller built by hand
# --------------------------------------------------------------------------- #


def test_from_matrix_builds_both_layouts_of_one_matrix(matrix: Any, hand: dict[int, Layout]) -> None:
    """The `.tocsc()` line every ingest script wrote, absorbed -- and to the same bytes.

    If this drifts, a store still writes and still reads; it simply holds a layout that is not the
    one the caller would have written, which nothing downstream can notice.
    """
    built = SparseArray.from_matrix(matrix)
    assert built.indexed_axes == (0, 1)
    for axis in (0, 1):
        _assert_same(built.layouts[axis], hand[axis])


def test_asking_for_one_axis_buys_one_layout(matrix: Any, hand: dict[int, Layout]) -> None:
    """Each layout is another copy of the nonzeros, so `axes=` is how a caller declines to pay.

    The default is every axis because that answers every question; halving the store is a decision
    and has to be made out loud.
    """
    built = SparseArray.from_matrix(matrix, axes=(1,))
    assert built.indexed_axes == (1,)
    _assert_same(built.layouts[1], hand[1])


def test_from_coords_builds_what_the_scipy_construction_builds(matrix: Any, hand: dict[int, Layout]) -> None:
    """Coordinates in, the same layouts out -- the rank-two case of the general constructor."""
    coo = matrix.tocoo()
    built = SparseArray.from_coords(matrix.shape, (coo.row, coo.col), coo.data)
    for axis in (0, 1):
        _assert_same(built.layouts[axis], hand[axis])


def test_from_coords_at_rank_three_matches_the_hand_written_ravel(cube: np.ndarray) -> None:
    """The 18-line loop this replaces, asserted against line by line.

    A wrong `index_order` here does not fail: it reads a different cell, forever, which is why the
    order is compared and not just the arrays.
    """
    coords = np.nonzero(cube)
    values = cube[coords]
    built = SparseArray.from_coords(cube.shape, coords, values)

    for axis in range(3):
        others = tuple(other for other in range(3) if other != axis)
        stride = cube.shape[others[1]]
        flat = sp.csr_matrix(
            (values, (coords[axis], coords[others[0]] * stride + coords[others[1]])),
            shape=(cube.shape[axis], cube.shape[others[0]] * stride),
        )
        expected = layout_over(
            cube.shape, axis, data=flat.data, indices=flat.indices, indptr=flat.indptr, index_order=others
        )
        _assert_same(built.layouts[axis], expected)


def test_from_dense_keeps_only_the_cells_that_are_not_zero(cube: np.ndarray) -> None:
    """The fixture helper that lived in this suite on purpose, now a converter.

    Reversed knowingly: the reason it stayed out was that there was nothing for it to live in.
    """
    built = SparseArray.from_dense(cube)
    assert built.nnz == int(np.count_nonzero(cube))
    assert built.shape == CUBE
    assert built.indexed_axes == (0, 1, 2)


def test_from_anndata_reads_an_x_without_rehydrating_a_matrix(
    tmp_path: Path, matrix: Any, hand: dict[int, Layout]
) -> None:
    """A layout *is* anndata's spelling, so an on-disk X is already one -- in every spelling of it.

    An `AnnData`, the group holding it and the sparse group itself all reach the same three arrays;
    if one of them diverges, a caller's choice of how to open its own file changes what gets stored.
    """
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(X=matrix)
    adata.write_zarr(tmp_path / "table.zarr")
    group = zarr.open_group(str(tmp_path / "table.zarr"), mode="r")

    for source in (adata, group, group["X"]):
        built = SparseArray.from_anndata(source)
        for axis in (0, 1):
            _assert_same(built.layouts[axis], hand[axis])


def test_summed_duplicates_agree_with_scipy_to_float_rounding() -> None:
    """Two values at one cell add up, as scipy's COO constructor does -- but not bit for bit.

    Both sum the same numbers; they do not sum them in the same order, and float addition is not
    associative. Asserting equality here would be asserting scipy's accumulation order, which is
    not a property of this format. The *count*, the indices and the offsets are exact, and those
    are what a mis-ravel would break.
    """
    rng = np.random.default_rng(3)
    shape = (200, 30, 4)
    coords = tuple(rng.integers(0, size, 5_000) for size in shape)
    values = rng.random(5_000).astype(np.float32)

    built = SparseArray.from_coords(shape, coords, values).layouts[0]
    flat = sp.csr_matrix((values, (coords[0], coords[1] * shape[2] + coords[2])), shape=(shape[0], shape[1] * shape[2]))
    assert np.array_equal(built.indices, flat.indices)
    assert np.array_equal(built.indptr, flat.indptr)
    assert np.allclose(built.data, flat.data)


def test_duplicates_can_be_refused_instead_of_summed() -> None:
    """For a caller who believes its coordinates are unique -- because a summed pair looks like a larger value.

    Silently doubling one cell is exactly the kind of mistake this package refuses elsewhere; the
    default is `sum` only because that is what the construction being replaced already did.
    """
    with pytest.raises(LayoutError, match="already names"):
        SparseArray.from_coords((4, 3), (np.array([1, 1]), np.array([2, 2])), np.array([1.0, 2.0]), duplicates="refuse")


def test_as_sparse_array_dispatches_on_what_a_value_carries(matrix: Any, cube: np.ndarray) -> None:
    """One entry point for callers holding "whatever it came in as", duck-typed like `layout_of`.

    Dispatching on attributes rather than classes is what keeps scipy and anndata optional: the
    branch that is not taken never imports anything.
    """
    assert as_sparse_array(matrix).indexed_axes == (0, 1)
    assert as_sparse_array([matrix, matrix.tocsc()]).indexed_axes == (0, 1)
    assert as_sparse_array(cube).shape == CUBE
    array = SparseArray.from_matrix(matrix)
    assert as_sparse_array(array) is array
    assert as_sparse_array(next(iter(array))).indexed_axes == (0,)


def test_something_with_no_sparse_array_in_it_is_named_rather_than_guessed_at() -> None:
    """A converter that guesses is a converter that writes the wrong store quietly."""
    with pytest.raises(TypeError, match="from_coords"):
        as_sparse_array("a path, probably")


# --------------------------------------------------------------------------- #
# Re-compression, at every rank
# --------------------------------------------------------------------------- #


def test_with_axis_rebuilds_a_layout_above_rank_two(cube: np.ndarray) -> None:
    """`.tocsc()` spells one answer and only at rank two; the coordinates spell all of them.

    Without this, a rank-three array built along one axis could never gain another, and the
    generalisation the format is *for* would stop at the constructor.
    """
    coords = np.nonzero(cube)
    values = cube[coords]
    one = SparseArray.from_coords(cube.shape, coords, values, axes=(0,))
    grown = one.with_axis(2)

    assert one.indexed_axes == (0,), "with_axis returns a new array; nothing here mutates"
    assert grown.indexed_axes == (0, 2)
    _assert_same(grown.layouts[2], SparseArray.from_coords(cube.shape, coords, values, axes=(2,)).layouts[2])


def test_to_coords_is_the_inverse_of_the_compression(cube: np.ndarray) -> None:
    """The exchange format between layouts: unravel through `index_order`, repeat through `indptr`.

    If either half is wrong every re-compression is wrong, and each one is wrong in a way that
    still writes, still reads and still returns the right *number* of values.
    """
    built = SparseArray.from_dense(cube, axes=(1,))
    coords, values = built.to_coords()
    assert len(coords) == 3
    assert np.array_equal(values, cube[coords])


def test_a_sparse_array_is_accepted_wherever_a_list_of_matrices_is(tmp_path: Path, matrix: Any) -> None:
    """One branch in `layouts_of` is what makes every consumer take the new object with no change.

    The writer, and anything that funnels through the same function to reach a granted prefix.
    """
    array = SparseArray.from_matrix(matrix)
    assert list(layouts_of(array)) == [0, 1]

    write_store(tmp_path / "viaobject.zarr", array)
    write_store(tmp_path / "vialist.zarr", [matrix, matrix.tocsc()])
    assert describe(tmp_path / "viaobject.zarr") == describe(tmp_path / "vialist.zarr")
    assert np.array_equal(read_layout(tmp_path / "viaobject.zarr", 0).toarray(), matrix.toarray())


def test_a_layout_written_through_the_api_is_still_read_by_anndata(tmp_path: Path, matrix: Any) -> None:
    """The interop claim, restated over the convenience layer rather than assumed to carry."""
    anndata = pytest.importorskip("anndata")
    SparseArray.from_matrix(matrix).write(tmp_path / "store.zarr")
    group = zarr.open_group(str(tmp_path / "store.zarr"), mode="r")
    assert np.array_equal(anndata.io.read_elem(group["layouts/axis0"]).toarray(), matrix.toarray())


# --------------------------------------------------------------------------- #
# The store: one description, one reader per axis, and the refusal intact
# --------------------------------------------------------------------------- #


@pytest.fixture
def written(tmp_path: Path, cube: np.ndarray) -> Path:
    """A rank-three store compressed along every one of its axes."""
    return SparseArray.from_dense(cube).write(tmp_path / "cube.zarr")


def test_a_store_answers_along_every_axis_it_compresses(written: Path, cube: np.ndarray) -> None:
    """The point of holding a reader per axis: two questions, one opened prefix.

    `coords_at` comes back in `index_order`, not axis order, which is the only honest way to read
    it -- so the assertion indexes the cube through that order rather than through `range(3)`.
    """
    with open_array(written) as store:
        assert store.shape == CUBE
        assert store.indexed_axes == (0, 1, 2)
        (first, second), values = store.coords_at(1, axis=1)
        assert np.array_equal(values, cube[first, 1, second])
        assert store.along(1) is store.along(1), "the reader is opened once and kept"


def test_reading_without_naming_an_axis_is_refused_when_there_is_a_choice(written: Path) -> None:
    """The reader's refusal, which wrapping it must not turn into a default.

    Which layout to read is the whole of what two layouts differ in: picking one for the caller
    answers a different question at the same speed, and nothing about the result says so.
    """
    with open_array(written) as store, pytest.raises(LayoutError, match="a decision"):
        store.slice_at(1)


def test_one_layout_is_not_a_choice(tmp_path: Path, matrix: Any) -> None:
    """With a single layout there is nothing to decide, so `axis=` is not required to say it."""
    path = SparseArray.from_matrix(matrix, axes=(0,)).write(tmp_path / "one.zarr")
    with open_array(path) as store:
        assert store.indexed_axes == (0,)
        positions, values = store.slice_at(0)
        assert len(positions) == len(values)


def test_asking_along_an_axis_no_layout_compresses_says_what_that_would_cost(tmp_path: Path, matrix: Any) -> None:
    """Not slower -- a scan of every byte, which is the measurement the second layout exists for."""
    path = SparseArray.from_matrix(matrix, axes=(0,)).write(tmp_path / "one.zarr")
    with open_array(path) as store, pytest.raises(LayoutError, match="scan of every byte"):
        store.slice_at(0, axis=1)


def test_to_array_reads_every_layout_rather_than_resolving_one(written: Path, cube: np.ndarray) -> None:
    """It asks for all of them, so it must not trip the refusal that exists for asking for one.

    Routing this through the single-axis resolver would make a two-layout store impossible to read
    back whole, which is the one thing this method is for.
    """
    with open_array(written) as store:
        back = store.to_array()
    assert back.indexed_axes == (0, 1, 2)
    for axis in range(3):
        _assert_same(back.layouts[axis], SparseArray.from_dense(cube, axes=(axis,)).layouts[axis])


def test_a_store_round_trips_through_the_object_it_was_written_from(tmp_path: Path, cube: np.ndarray) -> None:
    """Write, read whole, write again -- the same store, so nothing is lost in either direction."""
    first = SparseArray.from_dense(cube).write(tmp_path / "first.zarr")
    with open_array(first) as store:
        store.to_array().write(tmp_path / "second.zarr")
    assert describe(tmp_path / "first.zarr") == describe(tmp_path / "second.zarr")


def test_the_store_can_be_used_without_a_with_block(written: Path) -> None:
    """Opening reads only the block, so there is nothing that must be closed to be correct."""
    store = SparseStore(written)
    assert store.rank == 3
    store.close()
    assert "axis0+axis1+axis2" in repr(store)


def test_a_coo_matrix_is_converted_rather_than_refused(matrix: Any, hand: dict[int, Layout]) -> None:
    """`layout_of` refuses COO, correctly; the converter does not have to.

    That refusal is about *which axis* one layout compresses, which COO does not carry. Here the
    axes are the argument, so there is nothing left to guess -- and the ingest scripts that used to
    write `matrix.tocsr() if not isspmatrix_csr(matrix)` had a COO to handle for exactly this reason.
    """
    built = SparseArray.from_matrix(matrix.tocoo())
    for axis in (0, 1):
        _assert_same(built.layouts[axis], hand[axis])
    assert as_sparse_array(matrix.tocoo()).indexed_axes == (0, 1)


# --------------------------------------------------------------------------- #
# What the layer refuses
#
# The converters were the claim this file was written to check. These are the other half: the
# states a `SparseArray` could be *put* in that it documents as impossible, and the arguments it
# accepted and then ignored. Each of these was reachable before it was closed.
# --------------------------------------------------------------------------- #


def test_coordinates_off_the_end_name_the_axis_and_the_value(cube: Any) -> None:
    """An out-of-range coordinate is refused here, where the coordinate is still in hand.

    Left to what happens downstream it is two different silences: on a raveled axis numpy says
    "invalid entry in coordinates array", naming neither the axis nor the value; on the *compressed*
    axis nothing catches it at all -- `bincount` returns a longer array and the result is a
    SparseArray whose `shape` lies, refused much later by the writer with a message about `indptr`
    length that says nothing about the coordinate that caused it.
    """
    coords = (np.array([0, 1, 5]), np.array([0, 1, 2]), np.array([0, 1, 2]))
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(LayoutError, match="along axis 0 include 5"):
        SparseArray.from_coords((3, 4, 5), coords, values, axes=(0,))
    with pytest.raises(LayoutError, match="along axis 0 include -1"):
        SparseArray.from_coords((3, 4, 5), (np.array([-1, 1, 2]), coords[1], coords[2]), values)


def test_an_array_holding_no_layouts_is_refused_at_construction(matrix: Any) -> None:
    """The class says an instance of it is a set that could be written; now that is true.

    Empty, it used to construct happily and then raise `StopIteration` from `shape` -- the worst
    available failure, because inside a generator that does not propagate, it silently ends it.
    """
    with pytest.raises(LayoutError, match="is its layouts"):
        SparseArray({})
    with pytest.raises(LayoutError, match="different shapes"):
        SparseArray.from_matrices([matrix, matrix[:10].tocsc()])


def test_two_arrays_of_the_same_values_compare_equal(matrix: Any) -> None:
    """`store.to_array() == array` is the first assertion anyone writes, and it used to raise.

    The dataclass compared its fields as a tuple, which compares numpy arrays with `==`, which
    returns an array the tuple comparison then asks for a truth value.
    """
    assert SparseArray.from_matrix(matrix) == SparseArray.from_matrix(matrix)
    assert SparseArray.from_matrix(matrix, axes=(0,)) != SparseArray.from_matrix(matrix)
    doubled = SparseArray.from_matrix((matrix * 2).tocsr())
    assert SparseArray.from_matrix(matrix) != doubled
    assert SparseArray.from_matrix(matrix).layout(0) == SparseArray.from_matrix(matrix).layout(0)


def test_the_layouts_cannot_be_reached_around_and_mutated(matrix: Any) -> None:
    """`frozen=True` blocks rebinding, not mutation, so the refusals were a formality."""
    array = SparseArray.from_matrix(matrix)
    with pytest.raises(TypeError):
        array.layouts[99] = array.layout(0)  # pyright: ignore[reportIndexIssue]
    with pytest.raises(TypeError):
        hash(array)


def test_asking_for_fewer_axes_through_the_dispatcher_buys_fewer(matrix: Any) -> None:
    """`axes=` used to be dropped without a word by the branches built from existing layouts."""
    layouts = list(SparseArray.from_matrix(matrix))
    assert as_sparse_array(layouts, axes=(1,)).indexed_axes == (1,)
    assert as_sparse_array(layouts).indexed_axes == (0, 1)
    assert as_sparse_array(SparseArray.from_matrix(matrix), axes=(0,)).indexed_axes == (0,)
    with pytest.raises(LayoutError, match="already built"):
        as_sparse_array(layouts, axes=(5,))


def test_duplicates_can_be_refused_through_the_dispatcher(matrix: Any) -> None:
    """A COO reaching the matrix branch had its duplicates summed by scipy, silently.

    Which is the one thing `Duplicates` exists to document, because a summed duplicate is
    indistinguishable from a larger value.
    """
    coo = sp.coo_matrix(
        (np.array([1.0, 2.0], dtype=np.float32), (np.array([0, 0]), np.array([1, 1]))), shape=(3, 4)
    )
    assert as_sparse_array(coo).layout(0).data.tolist() == [3.0]
    with pytest.raises(LayoutError, match="already names"):
        as_sparse_array(coo, duplicates="refuse")


def test_a_store_reads_a_batch_along_every_axis_it_compresses(written: Path, cube: Any) -> None:
    """The batch methods forward exactly as the single-position ones do, axis and all."""
    with open_array(written) as store:
        for axis in range(cube.ndim):
            selection = store.slices_at([2, 0, 2], axis=axis)
            assert len(selection) == 3
            for place, position in enumerate((2, 0, 2)):
                assert np.array_equal(selection.slice_at(place)[1], store.slice_at(position, axis=axis)[1])
            assert np.allclose(store.dense_slices([1, 0], axis=axis), store.slices_at([1, 0], axis=axis).to_dense())


def test_a_batch_without_an_axis_is_refused_when_there_is_a_choice(written: Path) -> None:
    """Wrapping the reader's refusal must not become defaulting it, batch methods included."""
    with open_array(written) as store:
        with pytest.raises(LayoutError, match="a decision"):
            store.slices_at([0, 1])
        with pytest.raises(LayoutError, match="a decision"):
            store.slices_over(0, 2)


# --------------------------------------------------------------------------- #
# The surface no test reached
#
# Written after `as_sparse_array(axes=...)` turned out to have been dropped on two of its branches
# for as long as it had existed. Nothing had caught it because nothing had called it: the converters
# were covered, and the accessors around them were not. These are the ones that were not.
# --------------------------------------------------------------------------- #


def test_from_layouts_and_from_matrices_build_what_the_hand_written_form_built(
    matrix: Any, hand: dict[int, Layout]
) -> None:
    """The two converters that add nothing, asserted against the construction they replace."""
    for built in (
        SparseArray.from_matrices([matrix, matrix.tocsc()]),
        SparseArray.from_layouts(list(hand.values())),
        SparseArray.from_layouts(hand[0]),
    ):
        for axis in built.indexed_axes:
            _assert_same(built.layout(axis), hand[axis])
    assert SparseArray.from_layouts(hand[0]).indexed_axes == (0,)


def test_asking_a_layout_for_an_axis_it_does_not_hold_names_the_ones_it_does(matrix: Any) -> None:
    """The refusal says what it costs to fix, rather than only that it cannot."""
    array = SparseArray.from_matrix(matrix, axes=(0,))
    assert array.layout(0) is array.layouts[0]
    with pytest.raises(LayoutError, match=r"compresses \(0,\), not axis 1"):
        array.layout(1)


def test_to_scipy_returns_the_matrix_the_layout_came_from(matrix: Any) -> None:
    """Rank two only, and the axis is keyword-only like every other read in the package."""
    array = SparseArray.from_matrix(matrix)
    assert np.allclose(array.to_scipy(axis=0).toarray(), matrix.toarray())
    assert array.to_scipy(axis=1).format == "csc"
    cube = SparseArray.from_dense(np.eye(3)[None, :, :] * np.arange(1, 4)[:, None, None])
    with pytest.raises(SporadikError, match="no rank-3 matrix"):
        cube.to_scipy(axis=0)


def test_to_dense_is_the_inverse_of_from_dense(cube: Any) -> None:
    """`from_dense` had no counterpart, so a round trip could not be stated in one line."""
    assert np.allclose(SparseArray.from_dense(cube).to_dense(), cube)


def test_write_into_puts_the_layouts_in_a_group_that_is_already_open(tmp_path: Path, matrix: Any) -> None:
    """The split that lets one writer serve a directory and a granted object-store prefix alike."""
    import zarr

    group = zarr.open_group(str(tmp_path / "into.zarr"), mode="w")
    SparseArray.from_matrix(matrix).write_into(group)
    assert describe(tmp_path / "into.zarr").shape == matrix.shape


def test_a_store_answers_the_things_it_knows_without_reading_any_values(written: Path, cube: Any) -> None:
    """`nnz`, `dtype` and `len` come off the description, which is why they cost nothing."""
    with open_array(written) as store:
        assert store.nnz == int(np.count_nonzero(cube))
        assert store.dtype == str(cube.dtype)
        assert len(store) == cube.ndim
        assert store.rank == cube.ndim


def test_the_single_position_reads_agree_with_the_matrix_they_were_written_from(
    tmp_path: Path, matrix: Any
) -> None:
    """`bounds_at`, `dense_slice`, `maxima` and `to_scipy` on a store -- none had been called."""
    path = SparseArray.from_matrix(matrix).write(tmp_path / "reads.zarr")
    with open_array(path) as store:
        low, high = store.bounds_at(7, axis=1)
        assert high - low == matrix.getcol(7).nnz
        assert np.allclose(store.dense_slice(7, axis=1), matrix.getcol(7).toarray().ravel())
        assert np.allclose(store.maxima(axis=1), np.abs(matrix.toarray()).max(axis=0))
        assert np.allclose(store.to_scipy(axis=0).toarray(), matrix.toarray())


def test_add_axis_writes_one_layout_without_rewriting_the_others(tmp_path: Path, matrix: Any) -> None:
    """The point is what it does *not* touch: the layouts already there keep their bytes."""
    path = SparseArray.from_matrix(matrix, axes=(0,)).write(tmp_path / "grow.zarr")
    def fingerprint() -> dict[str, bytes]:
        root = path / "layouts" / "axis0"
        return {str(f.relative_to(root)): f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}

    before = fingerprint()

    with open_array(path) as store:
        assert store.indexed_axes == (0,)
        store.add_axis(1)
        assert store.indexed_axes == (0, 1)
        assert np.allclose(store.dense_slice(7, axis=1), matrix.getcol(7).toarray().ravel())

    assert fingerprint() == before, "adding an axis rewrote a layout that was already correct"
    assert describe(path).shape == matrix.shape
    assert np.allclose(read_layout(path, 1).toarray(), matrix.toarray())


def test_add_axis_is_a_no_op_for_an_axis_already_there(tmp_path: Path, matrix: Any) -> None:
    """And a refusal for one the array does not have, in the same words `with_axis` uses."""
    path = SparseArray.from_matrix(matrix).write(tmp_path / "full.zarr")
    with open_array(path) as store:
        assert store.add_axis(1) is store
        with pytest.raises(LayoutError, match="not an axis of shape"):
            store.add_axis(5)


def test_to_layout_reads_one_whole_layout_back(tmp_path: Path, matrix: Any) -> None:
    """The read both `to_array` and `add_axis` needed, rather than each reaching past the reader."""
    path = SparseArray.from_matrix(matrix).write(tmp_path / "whole.zarr")
    with open_array(path) as store:
        assert store.along(0).to_layout() == SparseArray.from_matrix(matrix).layout(0)
        assert store.to_array() == SparseArray.from_matrix(matrix)
