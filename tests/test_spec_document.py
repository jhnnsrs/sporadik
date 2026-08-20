"""The README, checked against the code it specifies.

`README.md` is the normative document: a second implementation -- in another language, or in a
server forbidden from importing this package -- is written from it and from nothing else. A
specification nothing checks drifts from its reference implementation within a release, and the
drift is invisible precisely to the people relying on the document rather than the code.

So the block example is parsed out of the prose and asserted against `sporadik.spec`, and then fed
through the real reader. A README that describes a store this package would refuse fails here.
"""

import json
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

import sporadik

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
MARKER = "<!-- spec-block:"


def _documented_block() -> dict:
    """The example block out of `README.md`, found by its marker rather than by line number."""
    assert README.exists(), f"{README} is this package's specification; the suite asserts against it"
    text = README.read_text()
    assert MARKER in text, f"{README} has lost the marker that anchors its example block"
    return json.loads(text.split(MARKER, 1)[1].split("```json", 1)[1].split("```", 1)[0])


def test_the_documented_block_has_the_keys_the_code_writes():
    """The document and `block_for` describe the same object."""
    documented = _documented_block()
    shape = tuple(documented["shape"])
    # Layouts built directly rather than from a 91 033 x 19 059 matrix: the block is derived from
    # the axes and the ravel order, so materialising an array to obtain one would be asserting
    # about scipy's constructor at the cost of a minute.
    empty = np.empty(0, dtype=np.float32)
    written = sporadik.block_for(
        shape,
        {
            axis: sporadik.layout_over(
                shape, axis, data=empty, indices=empty.astype(np.int32), indptr=np.zeros(shape[axis] + 1, dtype=np.int32)
            )
            for axis in range(len(shape))
        },
    )

    assert sorted(documented) == sorted(written), "the block's keys, as documented"
    assert documented["spec"] == written["spec"] == sporadik.SPEC_VERSION
    assert documented["complete"] is written["complete"] is True
    assert documented["layouts"] == written["layouts"], "paths, axes and ravel orders, as documented"


def test_the_documented_paths_follow_the_rule_the_document_states():
    """`layouts/axis{k}` -- recomputed, not copied, so the table and the function cannot diverge."""
    documented = _documented_block()
    rank = len(documented["shape"])
    for entry in documented["layouts"]:
        assert entry["path"] == sporadik.layout_path(entry["indexed_axis"])
        others = [axis for axis in range(rank) if axis != entry["indexed_axis"]]
        assert sorted(entry["index_order"]) == others, "index_order is the axes it did not compress"


def test_a_store_written_here_reads_back_through_the_documented_block(tmp_path):
    """Not just shaped right -- what the writer actually lands matches the document, key for key."""
    matrix = sp.random(400, 90, density=0.05, format="csc", dtype=np.float32, random_state=0)
    store = sporadik.write_store(tmp_path / "s.zarr", [matrix, matrix.tocsr()])
    landed = json.loads((store / "zarr.json").read_text())["attributes"][sporadik.BLOCK_KEY]

    documented = _documented_block()
    assert sorted(landed) == sorted(documented)
    assert landed["spec"] == documented["spec"]
    assert [entry["path"] for entry in landed["layouts"]] == [entry["path"] for entry in documented["layouts"]]


@pytest.mark.parametrize("claim", [
    "len(indptr) == shape[indexed_axis] + 1",
    "layouts/axis{indexed_axis}",
    '"sporadik"',
])
def test_the_document_states_the_load_bearing_claims(claim):
    """The three sentences a second implementer cannot get right by guessing.

    Not a style check: the invariant, the naming rule and the attribute key are the parts of this
    format that are not derivable from looking at a store, and a README that stopped saying one of
    them would leave the next implementation to invent it.
    """
    assert claim in README.read_text(), f"the specification no longer states: {claim}"


def test_the_block_key_is_not_borrowed_from_a_consumer():
    """The namespace is the format's own.

    It was `mikro-sparse` while the format lived inside one client. A format that names a consumer
    in its own bytes is one that cannot honestly be used by a second one.
    """
    assert sporadik.BLOCK_KEY == "sporadik"
