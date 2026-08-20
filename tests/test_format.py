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
    assert sum(p.stat().st_size for p in blobs) < matrix.data.nbytes, "compressed, because a request pays for bytes but not per byte"


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
def test_validate_sparse_refuses_arrays_that_contradict_the_declaration(matrix: Any, changes: dict, expected: str) -> None:
    """Checked before the upload, because the server's copy of this check costs a round trip."""
    fields = {"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr, "shape": matrix.shape, "indexed_axis": 1}
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
    return layout_over(dense.shape, indexed_axis, data=flat.data, indices=flat.indices, indptr=flat.indptr, index_order=order)


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
