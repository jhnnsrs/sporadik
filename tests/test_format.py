"""The format: what it writes, and everything it refuses to read back.

These are the conformance half of the specification in `README.md`: the prose says what a store is,
and this says it in a form that fails. They live with the format rather than with any consumer of
it, because a second implementation is entitled to be checked against the same claims.

The refusals are most of what is here, and one of them is the reason the format has a root block
at all. **zarr writes an array's ``zarr.json`` before its chunks**, and fills a missing chunk with
the fill value rather than failing, so an interrupted upload leaves a tree whose declarations are
all intact and whose values are silently zero. `test_a_store_missing_its_chunks_is_refused` is
that failure, reproduced by deleting chunk objects from a store that was written correctly.

No network and no server: everything here is a directory on disk, which is the same tree the
upload path writes into a granted S3 prefix through the one shared writer.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import zarr

from sporadik import (
    BLOCK_KEY,
    LAYOUTS_GROUP,
    SPEC_VERSION,
    SparseReader,
    chunk_for,
    describe,
    layout_over,
    layouts_of,
    open_store,
    read_layout,
    validate_layout,
    write_store,
)

#: Small, and deliberately not square: a square matrix would let an axis mix-up pass every
#: shape check in the module, which is exactly the class of bug these tests are for.
ROWS, COLS, DENSITY = 400, 90, 0.05


@pytest.fixture
def matrix() -> Any:
    """One reproducible CSC matrix. Objects on axis 0, features on axis 1."""
    return sp.random(ROWS, COLS, density=DENSITY, format="csc", dtype=np.float32, random_state=0)


@pytest.fixture
def both(tmp_path: Path, matrix: Any) -> Path:
    """A store holding both layouts of `matrix` -- the shape a real ingest writes."""
    return write_store(tmp_path / "store.zarr", [matrix, matrix.tocsr()])


def _block(path: Path) -> dict:
    """The root block, read straight out of the group's own metadata."""
    return json.loads((path / "zarr.json").read_text())["attributes"][BLOCK_KEY]


def _rewrite_block(path: Path, **changes: Any) -> None:
    """Edit the root block in place -- how a malformed or half-written store is fabricated."""
    metadata = json.loads((path / "zarr.json").read_text())
    metadata["attributes"][BLOCK_KEY].update(changes)
    (path / "zarr.json").write_text(json.dumps(metadata))


# --------------------------------------------------------------------------- #
# What it writes
# --------------------------------------------------------------------------- #
def test_both_layouts_round_trip(both: Path, matrix: Any) -> None:
    """The values come back identical, from each layout independently."""
    for axis in (0, 1):
        back = read_layout(both, axis)
        assert back.shape == (ROWS, COLS)
        assert (back != matrix).nnz == 0


def test_the_block_names_what_was_written(both: Path) -> None:
    """The block is the store's own account of itself, and it is complete."""
    block = _block(both)
    assert block == {
        "spec": SPEC_VERSION,
        "complete": True,
        "shape": [ROWS, COLS],
        "layouts": [
            {"path": f"{LAYOUTS_GROUP}/axis0", "indexed_axis": 0, "index_order": [1]},
            {"path": f"{LAYOUTS_GROUP}/axis1", "indexed_axis": 1, "index_order": [0]},
        ],
    }


def test_each_layout_is_a_plain_anndata_group(both: Path) -> None:
    """At rank two a layout carries anndata's spelling exactly, and nothing of ours.

    The declared shape is the matrix's own -- not the compressed axis first -- because that is
    what anndata means by a `csc_matrix`, and being a real one is the whole interop claim.
    """
    for axis, encoding in ((0, "csr_matrix"), (1, "csc_matrix")):
        attrs = json.loads((both / LAYOUTS_GROUP / f"axis{axis}" / "zarr.json").read_text())["attributes"]
        assert attrs["encoding-type"] == encoding
        assert attrs["encoding-version"] == "0.1.0"
        assert attrs["shape"] == [ROWS, COLS]
        assert BLOCK_KEY not in attrs


def test_anndata_reads_a_layout_unchanged(both: Path, matrix: Any) -> None:
    """The whole reason this is not a bespoke format: anndata reads it with no cooperation.

    Asserted rather than remembered, because it is the single claim the decision not to invent a
    file format rests on -- if it stops being true, the trade-off that was priced changes.
    """
    anndata_io = pytest.importorskip("anndata.io")
    for axis in (0, 1):
        group = zarr.open_group(str(both), mode="r")[f"{LAYOUTS_GROUP}/axis{axis}"]
        back = anndata_io.read_elem(group)
        assert back.shape == (ROWS, COLS)
        assert (back != matrix).nnz == 0, f"axis{axis} must be the matrix, not its transpose"


def test_every_array_is_chunked_at_the_request_granularity(both: Path, matrix: Any) -> None:
    """Chunked and compressed, sized for one S3 request -- including `indptr`.

    `indptr` matters most and is the thing that changed: written whole it is 152 KB over 19 059
    features and 43 MB over 5.4 M bins, and the second is a transfer nobody wants for two entries.
    Chunked, the 128 KB holding those two also serves the next sixteen thousand positions.
    """
    info = describe(both).layouts[1]
    assert not info.range_readable, "the default trades bytes for cache reuse, so it is not byte-addressable"
    for name in ("data", "indices", "indptr"):
        assert info.chunks[name] == chunk_for(np.dtype(np.float32) if name == "data" else np.int32, info.chunks[name])

    blobs = [p for p in (both / LAYOUTS_GROUP / "axis1" / "data").rglob("*") if p.is_file() and p.name != "zarr.json"]
    assert sum(p.stat().st_size for p in blobs) < matrix.data.nbytes, (
        "compressed, because a request pays for bytes but not per byte"
    )


def test_a_byte_addressable_store_reads_only_its_own_bytes(tmp_path: Path, matrix: Any) -> None:
    """The other trade, kept because it is the right one for a cold reader that caches nothing.

    Counted at the store, which is the only place the claim is real: `slice_at` issues range reads
    whose byte totals are exactly the slice, plus 2 x itemsize for the `indptr` bracket.
    """
    from zarr.abc.store import RangeByteRequest

    store_path = write_store(tmp_path / "exact.zarr", matrix, byte_addressable=True)
    info = describe(store_path).layouts[1]
    assert info.range_readable

    reader = SparseReader(store_path, 1)
    fetched: list[int] = []
    store = reader._group["data"].store_path.store
    original = store.get_partial_values

    async def counting(prototype: Any, key_ranges: Any) -> Any:
        pairs = list(key_ranges)
        for _, request in pairs:
            if isinstance(request, RangeByteRequest):
                fetched.append(request.end - request.start)
        return await original(prototype, pairs)

    store.get_partial_values = counting
    try:
        _, values = reader.slice_at(7)
    finally:
        store.get_partial_values = original

    column = matrix.getcol(7)
    wanted = column.nnz * (matrix.data.dtype.itemsize + matrix.indices.dtype.itemsize)
    bracket = 2 * matrix.indptr.dtype.itemsize
    assert sum(fetched) == wanted + bracket, f"fetched {sum(fetched)} bytes for a {column.nnz}-nonzero slice"
    assert len(values) == column.nnz
    assert np.allclose(reader.dense_slice(7), column.toarray().ravel())


def test_both_write_modes_read_back_the_same_values(tmp_path: Path, matrix: Any) -> None:
    """The trade is where the bytes go, never what they say."""
    chunked = write_store(tmp_path / "a.zarr", matrix)
    exact = write_store(tmp_path / "b.zarr", matrix, byte_addressable=True)
    with open_store(chunked, 1) as one, open_store(exact, 1) as two:
        for position in (0, 7, COLS - 1):
            assert np.allclose(one.dense_slice(position), two.dense_slice(position))


def test_one_layout_is_a_legal_store(tmp_path: Path, matrix: Any) -> None:
    """A store answering one question is not half a store; it offers one capability."""
    store = write_store(tmp_path / "one.zarr", matrix)
    info = describe(store)
    assert sorted(info.layouts) == [1]
    assert info.indexing(1) is not None
    assert info.indexing(0) is None


def test_trailing_empty_slices_are_written(tmp_path: Path) -> None:
    """A slice with nothing in it after the last value -- a raster's samples after its last spike.

    Every such run starts at ``len(data)``, which ``np.maximum.reduceat`` refuses outright; the
    maxima of those slices are zero, like any other empty run.
    """
    raster = sp.csr_matrix(
        (np.ones(3, dtype=np.float32), ([0, 1, 2], [0, 5, 10])), shape=(3, 50)
    ).tocsc()
    store = write_store(tmp_path / "raster.zarr", raster)
    assert describe(store).layouts[1].nnz == 3
    np.testing.assert_array_equal(read_layout(store, 1).toarray(), raster.toarray())


# --------------------------------------------------------------------------- #
# What it refuses to write
# --------------------------------------------------------------------------- #
def test_two_of_the_same_layout_are_refused(matrix: Any) -> None:
    """One capability twice, with nothing to say which a reader should use."""
    with pytest.raises(ValueError, match="one capability twice"):
        layouts_of([matrix, matrix.copy()])


def test_two_shapes_are_two_stores(matrix: Any) -> None:
    """A transpose that was never re-encoded is the way this actually happens."""
    with pytest.raises(ValueError, match="different shapes"):
        layouts_of([matrix, matrix.T.tocsr()])


def test_a_coo_matrix_says_how_to_fix_it(matrix: Any) -> None:
    """COO carries its nonzeros differently, so it fails on the missing array, not the format.

    A `TypeError` rather than a `ValueError`, and deliberately so: there is no `indices` to read,
    which is a different fact from having one in an encoding this cannot write. Both refusals name
    the conversion, because that is what the caller has to do either way.
    """
    with pytest.raises(TypeError, match="tocsr"):
        layouts_of(matrix.tocoo())


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"indexed_axis": 5}, "not an axis of"),
        ({"shape": (400,)}, "at least 2 axes"),
        ({"indptr": np.arange(7)}, "entries"),
        ({"indices": np.arange(3)}, "parallel"),
    ],
)
def test_validate_sparse_refuses_arrays_that_contradict_the_declaration(
    matrix: Any, changes: dict, expected: str
) -> None:
    """Checked before the upload, because the server's copy of this check costs a round trip."""
    fields = {
        "data": matrix.data,
        "indices": matrix.indices,
        "indptr": matrix.indptr,
        "shape": matrix.shape,
        "indexed_axis": 1,
    }
    with pytest.raises(ValueError, match=expected):
        validate_layout(**{**fields, **changes})


# --------------------------------------------------------------------------- #
# What it refuses to read -- the half-written store, and its relatives
# --------------------------------------------------------------------------- #
def test_a_store_missing_its_chunks_is_refused(tmp_path: Path, matrix: Any) -> None:
    """The failure the block exists for, and the one that used to be completely silent.

    Deleting a layout's chunk objects leaves every ``zarr.json`` intact, so before the block this
    store passed every check, reported the right ``nnz``, and returned the right *number* of
    values for a slice -- all of them zero, because zarr substitutes the fill value for a chunk
    it cannot fetch. Nothing raised anywhere.
    """
    store = write_store(tmp_path / "torn.zarr", matrix)
    chunks = [p for p in (store / LAYOUTS_GROUP / "axis1" / "data").rglob("*") if p.is_file() and p.name != "zarr.json"]
    assert chunks, "the fixture must have written chunk objects for this test to mean anything"
    for chunk in chunks:
        chunk.unlink()

    # The store still *claims* everything it claimed before: this is why the claim is not enough.
    assert json.loads((store / LAYOUTS_GROUP / "axis1" / "data" / "zarr.json").read_text())["shape"] == [matrix.nnz]
    _rewrite_block(store, complete=False)

    with pytest.raises(ValueError, match="only a finished store is readable"):
        describe(store)


def test_a_store_with_no_block_is_refused(tmp_path: Path, matrix: Any) -> None:
    """An upload that died before its last write, which is what a missing block means."""
    store = write_store(tmp_path / "nb.zarr", matrix)
    metadata = json.loads((store / "zarr.json").read_text())
    del metadata["attributes"][BLOCK_KEY]
    (store / "zarr.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="did not finish"):
        describe(store)


def test_an_unknown_spec_is_refused_rather_than_guessed_at(both: Path) -> None:
    """A spec selects how every byte is read, so reading an unknown one is not conservative."""
    _rewrite_block(both, spec="2")
    with pytest.raises(ValueError, match="spec"):
        describe(both)


def test_a_layout_named_but_absent_is_refused(both: Path) -> None:
    """The block lists what the writer finished; a name with nothing behind it is a torn upload."""
    _rewrite_block(both, layouts=[
        {"path": f"{LAYOUTS_GROUP}/axis0", "indexed_axis": 0, "index_order": [1]},
        {"path": f"{LAYOUTS_GROUP}/nothing_here", "indexed_axis": 1, "index_order": [0]},
    ])
    with pytest.raises(ValueError, match="not in the prefix"):
        describe(both)


def test_a_layout_filed_under_the_wrong_name_is_refused(tmp_path: Path, matrix: Any) -> None:
    """The path and the group's own `encoding-type` are two statements of one fact.

    Worth checking because getting it wrong is silent in the worst way: the store would be read
    along the *other* axis, and every lookup would return a real, wrong slice.
    """
    store = write_store(tmp_path / "swapped.zarr", matrix)
    (store / LAYOUTS_GROUP / "axis1").rename(store / LAYOUTS_GROUP / "axis0")
    _rewrite_block(store, layouts=[{"path": f"{LAYOUTS_GROUP}/axis0", "indexed_axis": 1, "index_order": [0]}])

    with pytest.raises(ValueError, match="filed under the wrong name"):
        describe(store)


def test_a_layout_disagreeing_with_the_store_shape_is_refused(both: Path) -> None:
    """The check `_assert_stores_agree` used to make across two rows, made once against bytes."""
    path = both / LAYOUTS_GROUP / "axis0" / "zarr.json"
    metadata = json.loads(path.read_text())
    metadata["attributes"]["shape"] = [ROWS + 1, COLS]
    path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="declares shape"):
        describe(both)


def test_a_store_naming_no_layouts_is_refused(both: Path) -> None:
    """A store is its layouts."""
    _rewrite_block(both, layouts=[])
    with pytest.raises(ValueError, match="names no layouts"):
        describe(both)


# --------------------------------------------------------------------------- #
# Reading one slice
# --------------------------------------------------------------------------- #
def test_a_slice_is_the_column_it_should_be(both: Path, matrix: Any) -> None:
    """The read a colouring makes: one feature, every object's value for it."""
    with open_store(both, 1) as reader:
        for position in (0, 7, COLS - 1):
            positions, values = reader.slice_at(position)
            column = matrix.getcol(position).toarray().ravel()
            assert np.array_equal(positions, np.nonzero(column)[0])
            assert np.allclose(values, column[column != 0])
            assert np.allclose(reader.dense_slice(position), column)


def test_a_slice_is_the_row_it_should_be(both: Path, matrix: Any) -> None:
    """The read a hover makes, out of the other layout: one object, everything in it."""
    rows = matrix.tocsr()
    with open_store(both, 0) as reader:
        row = rows.getrow(11).toarray().ravel()
        assert np.allclose(reader.dense_slice(11), row)


def test_maxima_matches_a_dense_reduction(both: Path, matrix: Any) -> None:
    """The window an ingest stores, because the server serves no statistics."""
    with open_store(both, 1) as reader:
        assert np.allclose(reader.maxima(), np.abs(matrix.toarray()).max(axis=0))


def test_a_position_off_the_end_is_an_index_error(both: Path) -> None:
    """Named against the indexed axis, since that is the one the reader is talking about."""
    with open_store(both, 1) as reader, pytest.raises(IndexError, match=f"{COLS} slices"):
        reader.slice_at(COLS)


def test_reading_a_layout_the_store_does_not_hold_is_refused(tmp_path: Path, matrix: Any) -> None:
    """Not a slower read -- a scan of every byte, which is why it is refused instead."""
    store = write_store(tmp_path / "csc_only.zarr", matrix)
    with pytest.raises(ValueError, match="scan of every byte"):
        SparseReader(store, 0)


def test_two_layouts_make_the_choice_explicit(both: Path) -> None:
    """Which layout to read is the whole of what they differ in, so it is not defaulted."""
    with pytest.raises(ValueError, match="a decision"):
        SparseReader(both)


def test_a_layout_built_from_raw_arrays_needs_no_scipy(tmp_path: Path, matrix: Any) -> None:
    """`scipy` is not a dependency of this package, and the writer's input says so."""
    layout = layout_over(
        (ROWS, COLS), 1,
        data=np.asarray(matrix.data), indices=np.asarray(matrix.indices), indptr=np.asarray(matrix.indptr),
    )
    store = write_store(tmp_path / "raw.zarr", layout)
    assert describe(store).layouts[1].nnz == matrix.nnz


# --------------------------------------------------------------------------- #
# Rank three and up -- two axes is one case, not the definition
# --------------------------------------------------------------------------- #
#: Deliberately three different extents: with any two equal, an axis mix-up survives every
#: shape check in the module, which is the class of bug rank makes easier to write.
CUBE = (6, 4, 5)


@pytest.fixture
def cube() -> np.ndarray:
    """A reproducible rank-three array that is mostly zeros."""
    rng = np.random.default_rng(0)
    return (rng.random(CUBE) * (rng.random(CUBE) < 0.3)).astype(np.float32)


def _layout_from_dense(dense: np.ndarray, indexed_axis: int) -> Any:
    """The three arrays for one layout of a dense array, at any rank.

    Literally "move the compressed axis to the front, flatten the rest, store CSR" -- which is
    what a layout *is*. Lives in the test rather than the module because a caller writing a
    rank-three store already holds its values in whatever form they came in; this is only how
    the fixture gets there.
    """
    order = tuple(axis for axis in range(dense.ndim) if axis != indexed_axis)
    flat = sp.csr_matrix(np.transpose(dense, (indexed_axis, *order)).reshape(dense.shape[indexed_axis], -1))
    return layout_over(
        dense.shape, indexed_axis, data=flat.data, indices=flat.indices, indptr=flat.indptr, index_order=order
    )


@pytest.fixture
def cube_store(tmp_path: Path, cube: np.ndarray) -> Path:
    """A rank-three store compressed along every one of its axes -- all three questions."""
    return write_store(tmp_path / "cube.zarr", [_layout_from_dense(cube, axis) for axis in range(3)])


def test_a_rank_three_store_answers_along_every_axis(cube_store: Path, cube: np.ndarray) -> None:
    """The point of the generalisation: one layout per axis something selects along.

    A (object, feature, timepoint) matrix answers "this object", "this feature" and "this
    timepoint" in one contiguous read each, and costs one stored layout per question.
    """
    info = describe(cube_store)
    assert info.rank == 3
    assert sorted(info.layouts) == [0, 1, 2]

    for axis in range(3):
        with open_store(cube_store, axis) as reader:
            assert reader.info.slices == CUBE[axis]
            for position in (0, CUBE[axis] - 1):
                assert np.allclose(reader.dense_slice(position), np.take(cube, position, axis=axis))


def test_a_rank_three_slice_unravels_to_real_coordinates(cube_store: Path, cube: np.ndarray) -> None:
    """`indices` is raveled over the other axes, so a position means nothing without the order."""
    with open_store(cube_store, 1) as reader:
        coords, values = reader.coords_at(2)
        assert reader.info.index_order == (0, 2)
        assert len(coords) == 2
        assert np.allclose(values, cube[coords[0], 2, coords[1]])


def test_the_indptr_invariant_holds_at_every_rank(cube_store: Path) -> None:
    """`len(indptr) == shape[indexed_axis] + 1` is the spine of the format, not a 2-D fact."""
    for axis, info in describe(cube_store).layouts.items():
        assert info.chunks["indptr"] == CUBE[axis] + 1


def test_a_rank_three_layout_declares_the_raveled_shape_it_holds(cube_store: Path) -> None:
    """Above rank two the child is a genuine csr over the raveled view, and says so.

    Not a lie about the data: what the group holds really is that two-axis matrix. The real
    shape lives in the block, which is the only thing that knows the array was ever rank three.
    """
    attrs = json.loads((cube_store / LAYOUTS_GROUP / "axis1" / "zarr.json").read_text())["attributes"]
    assert attrs["encoding-type"] == "csr_matrix"
    assert attrs["shape"] == [CUBE[1], CUBE[0] * CUBE[2]]
    assert _block(cube_store)["shape"] == list(CUBE)


def test_anndata_reads_a_rank_three_layout_as_its_raveled_view(cube_store: Path, cube: np.ndarray) -> None:
    """Interop degrades honestly rather than breaking: anndata gets the matrix that is there."""
    anndata_io = pytest.importorskip("anndata.io")
    back = anndata_io.read_elem(zarr.open_group(str(cube_store), mode="r")[f"{LAYOUTS_GROUP}/axis0"])
    assert back.shape == (CUBE[0], CUBE[1] * CUBE[2])
    assert np.allclose(back.toarray(), cube.reshape(CUBE[0], -1))


def test_scipy_cannot_be_asked_for_a_rank_three_matrix(cube_store: Path) -> None:
    """`read_sparse` is rank two only, and the refusal says what to use instead."""
    with pytest.raises(ValueError, match="slice at a time"):
        read_layout(cube_store, 0)


def test_an_index_order_that_is_not_a_permutation_is_refused(cube: np.ndarray) -> None:
    """The one fact in the format that cannot be recovered from the bytes, so it is checked."""
    layout = _layout_from_dense(cube, 1)
    with pytest.raises(ValueError, match="permutation"):
        layout_over(cube.shape, 1, data=layout.data, indices=layout.indices, indptr=layout.indptr, index_order=(0, 1))


def test_a_wrong_index_order_in_the_block_is_refused(cube_store: Path) -> None:
    """Because a wrong one does not fail -- it puts every value in a different cell."""
    block = _block(cube_store)
    block["layouts"][1]["index_order"] = [0, 1]
    _rewrite_block(cube_store, layouts=block["layouts"])
    with pytest.raises(ValueError, match="not a permutation"):
        describe(cube_store)


def test_more_layouts_than_axes_is_refused(cube_store: Path) -> None:
    """There is one axis to compress per axis the array has; a fourth would be a copy."""
    block = _block(cube_store)
    _rewrite_block(cube_store, layouts=[*block["layouts"], dict(block["layouts"][0])])
    with pytest.raises(ValueError, match="rank-3 array"):
        describe(cube_store)


# --------------------------------------------------------------------------- #
# Reading many slices
#
# The claim these make is not "the values are right" -- `slice_at` already had that -- it is **how
# many times we waited for them**. A format whose whole justification is one contiguous range read
# was offering one position at a time, and handing zarr's plural `get_partial_values` a list of
# exactly one, three times, to read a single slice. So the cost tests count *calls*, and they count
# them at the store, which is the only place the claim is real.
#
# The two write modes are counted through different hooks, and that is not an accident of the test:
# they are different mechanisms. A byte-addressable store issues range reads through
# `get_partial_values`; a chunked one goes through zarr's indexing to `store.get`, once per chunk,
# and makes **zero** `get_partial_values` calls. A cost test pointed at the wrong hook sees zero of
# everything and passes no matter what the code does.
# --------------------------------------------------------------------------- #


def _count_range_calls(reader: SparseReader) -> tuple[list[list[int]], Any]:
    """Record one entry per `get_partial_values` call, holding that call's range sizes."""
    store = reader._group["data"].store_path.store
    original = store.get_partial_values
    calls: list[list[int]] = []

    async def counting(prototype: Any, key_ranges: Any) -> Any:
        pairs = list(key_ranges)
        calls.append([getattr(request, "end", 0) - getattr(request, "start", 0) for _, request in pairs])
        return await original(prototype, pairs)

    store.get_partial_values = counting
    return calls, lambda: setattr(store, "get_partial_values", original)


def _count_chunk_gets(reader: SparseReader) -> tuple[list[str], Any]:
    """Record every chunk object a chunked store is asked for, by key."""
    store = reader._group["data"].store_path.store
    original = store.get
    keys: list[str] = []

    async def counting(key: str, *args: Any, **kwargs: Any) -> Any:
        keys.append(key)
        return await original(key, *args, **kwargs)

    store.get = counting
    return keys, lambda: setattr(store, "get", original)


@pytest.fixture
def exact(tmp_path: Path, matrix: Any) -> Path:
    """Both layouts, written byte-addressably -- the variant whose reads are range reads."""
    return write_store(tmp_path / "exact.zarr", [matrix, matrix.tocsr()], byte_addressable=True)


def test_one_slice_costs_two_round_trips_rather_than_three(exact: Path) -> None:
    """`indices` and `data` are the same range over sibling arrays, so they are one request.

    Three waves was never a property of the format -- `indptr` genuinely has to answer before there
    is a run to ask for, but the two arrays of that run do not have to answer one after the other.
    """
    reader = SparseReader(exact, 1)
    calls, restore = _count_range_calls(reader)
    try:
        reader.slice_at(7)
    finally:
        restore()
    assert len(calls) == 2, f"one slice took {len(calls)} waves"
    assert len(calls[0]) == 1, "the first wave is the indptr bracket"
    assert len(calls[1]) == 2, "the second wave is indices and data together"


def test_a_scattered_batch_costs_two_round_trips_whatever_its_size(exact: Path) -> None:
    """The headline: the number of waves does not depend on how many slices were asked for."""
    reader = SparseReader(exact, 1)
    for count in (3, 20, 60):
        calls, restore = _count_range_calls(reader)
        try:
            reader.slices_at(np.linspace(0, reader.info.slices - 1, count, dtype=int))
        finally:
            restore()
        assert len(calls) == 2, f"{count} slices took {len(calls)} waves"


def test_a_contiguous_range_of_slices_is_one_range_read(exact: Path) -> None:
    """Contiguous positions are already one byte range, so nothing is coalesced and none over-read."""
    reader = SparseReader(exact, 1)
    calls, restore = _count_range_calls(reader)
    try:
        selection = reader.slices_over(10, 40)
    finally:
        restore()
    assert len(calls) == 2
    assert len(calls[0]) == 1, "overlapping indptr windows fold into one range"
    assert len(calls[1]) == 2, "one indices range and one data range, not one pair per slice"
    low, _ = reader.bounds_at(10)
    _, high = reader.bounds_at(39)
    assert calls[1][0] == (high - low) * reader._group["indices"].dtype.itemsize
    assert selection.nnz == high - low


def test_a_batch_on_a_chunked_store_fetches_each_chunk_once(both: Path) -> None:
    """The chunked variant's win is the other one: a chunk two runs share is fetched once, not twice."""
    reader = SparseReader(both, 1)
    positions = list(range(0, reader.info.slices, 3))

    keys, restore = _count_chunk_gets(reader)
    try:
        reader.slices_at(positions)
    finally:
        restore()
    batched = list(keys)

    keys, restore = _count_chunk_gets(reader)
    try:
        for position in positions:
            reader.slice_at(position)
    finally:
        restore()
    one_at_a_time = list(keys)

    assert len(batched) == len(set(batched)), "a batch fetched some chunk twice"
    assert len(batched) < len(one_at_a_time)


def test_a_batch_of_one_costs_what_a_single_slice_costs(both: Path, exact: Path) -> None:
    """The one-run dispatch, pinned.

    Without it a batch of one is a fancy index over a contiguous run -- an eight-byte offset per
    nonzero to read what a plain slice reads directly -- and `slices_at([p])` would be *slower* than
    the `slice_at(p)` it is meant to replace.
    """
    chunked = SparseReader(both, 1)
    keys, restore = _count_chunk_gets(chunked)
    try:
        chunked.slices_at([5])
        batched = len(keys)
        keys.clear()
        chunked.slice_at(5)
    finally:
        restore()
    assert batched == len(keys)

    ranged = SparseReader(exact, 1)
    calls, restore = _count_range_calls(ranged)
    try:
        ranged.slices_at([5])
        batch_ranges = [list(call) for call in calls]
        calls.clear()
        ranged.slice_at(5)
        single_ranges = [list(call) for call in calls]
    finally:
        restore()
    assert batch_ranges == single_ranges


@pytest.mark.parametrize("axis", [0, 1])
def test_batched_and_one_at_a_time_read_the_same_values(both: Path, exact: Path, axis: int) -> None:
    """However few times we waited, the bytes are the ones reading them singly would have given."""
    for path in (both, exact):
        reader = SparseReader(path, axis)
        positions = np.array([4, 1, 4, 0, 9, 2])
        selection = reader.slices_at(positions)
        for place, position in enumerate(positions):
            wanted_indices, wanted_values = reader.slice_at(int(position))
            got_indices, got_values = selection.slice_at(place)
            assert np.array_equal(got_indices, wanted_indices)
            assert np.array_equal(got_values, wanted_values)


def test_slices_come_back_in_the_order_they_were_asked_for(both: Path) -> None:
    """Sorted is the order the bytes are *fetched* in; it is never the order they are returned in."""
    reader = SparseReader(both, 1)
    descending = np.arange(12)[::-1]
    selection = reader.slices_at(descending)
    assert np.array_equal(selection.positions, descending)
    for place, position in enumerate(descending):
        assert np.array_equal(selection.slice_at(place)[1], reader.slice_at(int(position))[1])


def test_a_repeated_position_is_read_once_and_returned_twice(exact: Path) -> None:
    """A batch of ids from a join legitimately repeats, so repeats are answered rather than refused."""
    reader = SparseReader(exact, 1)
    calls, restore = _count_range_calls(reader)
    try:
        selection = reader.slices_at([3, 3, 3])
    finally:
        restore()
    assert len(selection) == 3
    assert len(calls[1]) == 2, "one run fetched, not three"
    first = selection.slice_at(0)
    for place in (1, 2):
        assert np.array_equal(selection.slice_at(place)[1], first[1])


def test_an_empty_selection_is_a_selection_over_no_slices(both: Path) -> None:
    """What an empty filter returns, rather than a refusal of it."""
    selection = SparseReader(both, 1).slices_at([])
    assert len(selection) == 0
    assert selection.nnz == 0
    assert np.array_equal(selection.indptr, np.array([0]))


def test_a_position_off_the_end_is_an_index_error_in_a_batch_too(both: Path) -> None:
    """The same refusal as the scalar case, in the same words."""
    reader = SparseReader(both, 1)
    with pytest.raises(IndexError, match="not a position along axis 1"):
        reader.slices_at([0, reader.info.slices])


def test_a_selection_unravels_to_coordinates_in_the_original_frame(both: Path, matrix: Any) -> None:
    """Row *i* is ``positions[i]``, and `coords` is what says so.

    This is the trap the type exists for: a caller who reads a coordinate straight off the
    selection's own `indptr` gets a real, wrong one.
    """
    reader = SparseReader(both, 1)
    positions = np.array([7, 2, 7])
    (coords, values) = reader.slices_at(positions).coords()
    assert set(np.unique(coords[1]).tolist()) <= {2, 7}
    dense = matrix.toarray()
    assert np.allclose(dense[coords[0], coords[1]], values)


def test_a_selection_can_be_written_as_a_store_of_its_own(tmp_path: Path, both: Path, matrix: Any) -> None:
    """`as_layout` is the caller saying the selection *is* the array now -- and then it writes."""
    selection = SparseReader(both, 1).slices_over(10, 40)
    written = write_store(tmp_path / "subset.zarr", selection.as_layout())
    assert describe(written).shape == (matrix.shape[0], 30)
    assert np.allclose(read_layout(written, 1).toarray(), matrix.tocsc()[:, 10:40].toarray())


def test_dense_slices_is_dense_slice_stacked(both: Path) -> None:
    """The batch shape a caller feeding a model wants, and the same numbers as one at a time."""
    reader = SparseReader(both, 1)
    positions = [5, 1, 5]
    assert np.allclose(
        reader.dense_slices(positions), np.stack([reader.dense_slice(position) for position in positions])
    )


def test_reading_inside_an_event_loop_says_to_use_a_worker_thread(exact: Path) -> None:
    """The stance was always "read in a worker thread"; the code never said it.

    Inside a running loop `asyncio.run` raised asyncio's own error about asyncio, naming neither
    this package nor the way out of it.
    """
    import asyncio

    from sporadik import SporadikError

    reader = SparseReader(exact, 1)

    async def read() -> Any:
        return reader.slice_at(7)

    with pytest.raises(SporadikError, match="to_thread"):
        asyncio.run(read())


def test_reading_from_a_worker_thread_inside_an_event_loop_works(exact: Path) -> None:
    """The advice in that message, asserted rather than remembered."""
    import asyncio

    reader = SparseReader(exact, 1)

    async def read() -> Any:
        return await asyncio.to_thread(reader.slices_at, [7, 8])

    selection = asyncio.run(read())
    assert len(selection) == 2


def test_stored_maxima_match_the_reduction_they_replace(both: Path, matrix: Any) -> None:
    """The writer now does what this module's own advice always said to do at ingest."""
    reader = SparseReader(both, 1)
    assert describe(both).layouts[1].has_maxima
    assert np.allclose(reader.maxima(), np.abs(matrix.toarray()).max(axis=0))


def test_a_store_written_without_maxima_still_reads(tmp_path: Path, both: Path) -> None:
    """The array is additive, so a store written before it existed is still a legal store.

    Which is why nothing announces it in the block: `describe` looks up the three names it needs and
    never enumerates a layout group for unknown children, so the spec version does not move and a
    reader that predates this ignores the extra array rather than refusing the store.
    """
    import shutil

    legacy = tmp_path / "legacy.zarr"
    shutil.copytree(both, legacy)
    for axis in (0, 1):
        shutil.rmtree(legacy / LAYOUTS_GROUP / f"axis{axis}" / "maxima")

    assert describe(legacy).spec == SPEC_VERSION
    assert not describe(legacy).layouts[1].has_maxima
    assert np.allclose(SparseReader(legacy, 1).maxima(), SparseReader(both, 1).maxima())


def test_the_block_says_nothing_about_maxima(both: Path) -> None:
    """A derived fact is read off the artifact, never declared -- as `range_readable` already is.

    Stated as a test because the alternative was tempting and would have been a spec change:
    announcing the array in the block changes `block_for`'s output for *every* store, including the
    ones that do not carry it, and `tests/test_spec_document.py` asserts that output against the
    README.
    """
    block = _block(both)
    assert "maxima" not in json.dumps(block)


def test_a_closed_reader_says_so_rather_than_indexing_an_empty_cache(both: Path) -> None:
    """`close()` empties the cached `indptr`; without a flag the next read was numpy's IndexError."""
    from sporadik import SporadikError

    reader = SparseReader(both, 1)
    reader.close()
    for read in (lambda: reader.bounds_at(0), lambda: reader.slices_at([0, 1])):
        with pytest.raises(SporadikError, match="closed"):
            read()
