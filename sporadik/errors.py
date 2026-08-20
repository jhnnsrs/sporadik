"""What sporadik raises, and the one thing every message here is for.

A refusal names *what would have gone wrong silently*, because that is the only reason any of
these checks exist. A sparse store is a pile of integers: read it along the wrong axis, unravel it
in the wrong order, or read one whose upload stopped halfway, and nothing crashes -- you get real
numbers from the wrong cells, or the right count of zeros. Every error below marks a place where
that was possible and is now not.
"""


class SporadikError(ValueError):
    """Base for everything sporadik refuses. A `ValueError`, because it always is one."""


class SpecError(SporadikError):
    """The store declares a spec version this cannot read.

    Refused rather than attempted: a version selects how every byte in the prefix is interpreted,
    so reading an unknown one is not the cautious option, it is the reckless one.
    """


class IncompleteError(SporadikError):
    """The store has no completion marker, so its upload did not finish.

    The one failure the format exists to make visible. Zarr writes an array's metadata *before*
    its chunks and substitutes the fill value for a chunk it cannot fetch, so a torn upload leaves
    a tree whose every declaration is intact and whose values are silently zero.
    """


class LayoutError(SporadikError):
    """A layout contradicts the store around it, or itself.

    Its path disagrees with the axis it compresses, its `index_order` is not a permutation, its
    declared shape is not the one its rank implies, or its `indptr` does not have one entry per
    slice plus the end.
    """
