"""What ``import sporadik`` puts in front of a caller.

The package root is the API. Everything below it is a module a caller may reach into, but
``__all__`` is the part that is promised, and a name that drifts out of it is a public symbol
nobody can find or a documented one that no longer exists. Neither shows up in any other test,
because every other test imports what it needs directly.
"""

import importlib
import pkgutil

import sporadik

#: Submodules are importable and not part of the surface `__all__` describes.
_SUBMODULES = {module.name for module in pkgutil.iter_modules(sporadik.__path__)}


def test_everything_promised_is_actually_there():
    """Every name in `__all__` resolves. A promise that does not import is worse than no promise."""
    missing = [name for name in sporadik.__all__ if not hasattr(sporadik, name)]
    assert not missing, f"promised but absent: {missing}"


def test_nothing_public_is_left_out_of_the_promise():
    """Nothing reachable at the root is left undocumented and accidentally supported."""
    public = {name for name in dir(sporadik) if not name.startswith("_") and name not in _SUBMODULES}
    assert not public - set(sporadik.__all__)


def test_every_submodule_name_is_promised_at_the_root():
    """The other direction, and the one the check above structurally cannot see.

    `dir(sporadik)` only holds what `__init__` imported, so a public name in a submodule that was
    never imported is invisible to it -- and that is precisely the case worth catching, because such
    a name is real, annotated on public signatures, and importable only by reaching past the promise
    into `sporadik.layout`. `MatrixLike` and `Duplicates` were both in that state: `MatrixLike`
    annotates `write_store`, `read_layout` and four converters, so a caller could not type its own
    signatures without importing from a module the package never advertised.
    """
    unpromised: dict[str, list[str]] = {}
    for name in sorted(_SUBMODULES):
        module = importlib.import_module(f"sporadik.{name}")
        missing = [entry for entry in getattr(module, "__all__", ()) if entry not in sporadik.__all__]
        if missing:
            unpromised[name] = missing
    assert not unpromised, f"public in a submodule but not promised at the root: {unpromised}"


def test_the_promise_lists_each_name_once():
    """A name twice is a merge that went wrong, and `__all__` is where it would show."""
    duplicated = sorted({name for name in sporadik.__all__ if sporadik.__all__.count(name) > 1})
    assert not duplicated, f"listed more than once: {duplicated}"
