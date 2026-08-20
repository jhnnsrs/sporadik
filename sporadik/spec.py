"""The format itself: the names, the versions and the rules that are *normative*.

Everything in this module is the specification rather than an implementation of it. A second
implementation -- in another language, in a server that must not import this package -- reproduces
what is here and nothing else; :mod:`sporadik.writer` and :mod:`sporadik.reader` are then one
reference implementation among however many exist.

That separation is not tidiness. A reader that imports its writer inherits the writer's
dependencies and its release cycle, and a version skew between them becomes an outage rather than
a refusal. Two independent implementations of a written-down format is the only arrangement in
which "the format is specified" is a testable claim instead of a shared object file.

The format in one sentence:

> A sparse array is a zarr group holding one child per **axis made contiguous**, and a block in
> its own attributes -- written last -- saying which children it finished.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BLOCK_KEY",
    "ENCODINGS",
    "LAYOUTS_GROUP",
    "MIN_RANK",
    "SPEC_VERSION",
    "SUPPORTED_SPECS",
    "anndata_encoding",
    "block_for",
    "layout_path",
    "raveled_shape",
]

#: The spec version this package writes.
#:
#: A version selects how every byte in the prefix is read, so a reader that meets one it does not
#: know **refuses**. That is the same rule any format with a version needs and the reason to carry
#: one at all: registering a store nothing can decode fails later, somewhere else, and by then the
#: bytes have been paid for.
SPEC_VERSION = "1"

#: The versions this package reads. A superset of :data:`SPEC_VERSION` once there is a second.
SUPPORTED_SPECS = frozenset({"1"})

#: Where the block lives, in the root group's own attributes.
#:
#: Namespaced because the attributes are shared ground: anndata's ``encoding-type`` lives one level
#: down in each layout, and every other reader is entitled to ignore what it does not recognise --
#: as zarr, anndata and `zarrita` all do.
BLOCK_KEY = "sporadik"

#: The group the layouts hang under, one level below the root. `AnnData` puts whole sparse groups
#: under ``layers/`` the same way.
LAYOUTS_GROUP = "layouts"

#: The lowest rank this format describes: a compressed axis needs at least one other axis to hold
#: the positions. **There is no highest.** A layout is one axis made contiguous, so an array of
#: rank *n* has up to *n* of them, and rank two is the case where that coincides with CSR and CSC.
MIN_RANK = 2

#: anndata's two rank-two spellings, and which axis each one's ``indptr`` walks. A dict rather than
#: two constants because it is also the validation: an ``encoding-type`` outside it is a group this
#: cannot honestly claim to understand.
ENCODINGS: dict[str, int] = {"csr_matrix": 0, "csc_matrix": 1}


def layout_path(indexed_axis: int) -> str:
    """Where the layout compressing ``indexed_axis`` lives inside the prefix.

    Named for the axis rather than for an encoding, because *making one axis contiguous* is what a
    layout is at every rank -- ``csr``/``csc`` name the two cases of rank two and say nothing at
    rank three. A reader recomputes this and compares, so a layout filed under the wrong name is
    refused rather than silently indexed along the wrong axis.
    """
    return f"{LAYOUTS_GROUP}/axis{int(indexed_axis)}"


def anndata_encoding(rank: int, indexed_axis: int) -> str:
    """The ``encoding-type`` a layout's own group declares, at a given rank.

    At rank two this is anndata's exactly, and the three arrays sporadik writes are **byte-identical**
    to what anndata writes for the same matrix -- so a rank-two layout is a real anndata sparse
    group and `scanpy` reads it as the array it is.

    Above rank two there is no anndata spelling of the thing. The child holds the array raveled to
    two axes, which genuinely *is* a ``csr_matrix``, and says so; the block carries the real shape
    and the ravel order.
    """
    if rank == MIN_RANK:
        return "csr_matrix" if indexed_axis == 0 else "csc_matrix"
    return "csr_matrix"


def raveled_shape(shape: tuple[int, ...], indexed_axis: int, index_order: tuple[int, ...]) -> tuple[int, ...]:
    """The shape a layout's own group declares, which is not always the array's.

    At rank two it is the array's, so the group is anndata-faithful. Above rank two it is the
    raveled pair the child literally holds: the compressed axis, and the product of everything else
    in ``index_order``.
    """
    if len(shape) == MIN_RANK:
        return shape
    remainder = 1
    for axis in index_order:
        remainder *= int(shape[axis])
    return (int(shape[indexed_axis]), remainder)


def block_for(shape: tuple[int, ...], layouts: dict[int, Any]) -> dict[str, Any]:
    """The block a writer lands **last**, describing what it finished.

    Last is the whole of why it is worth having. Everything else in a prefix declares something
    before it is true -- zarr writes an array's ``zarr.json`` ahead of its chunks -- so the block
    is the only statement in the tree made *after* the thing it describes exists.

    ``index_order`` is the one entry here that is not a restatement of something the child already
    carries: it is the order ``indices`` was raveled over the uncompressed axes, and it cannot be
    recovered from the bytes. At rank two it has one member and says nothing; above it, a wrong one
    does not fail -- it reads a different cell.
    """
    return {
        "spec": SPEC_VERSION,
        "complete": True,
        "shape": [int(size) for size in shape],
        "layouts": [
            {
                "path": layout_path(axis),
                "indexed_axis": int(axis),
                "index_order": [int(other) for other in layouts[axis].index_order],
            }
            for axis in sorted(layouts)
        ],
    }
