"""__version__ must derive from distribution metadata, never a source literal.

Release 3.31.0 shipped with a stale hardcoded __version__ (3.30.1); these tests
keep that class of drift impossible.
"""
import ast
import importlib.metadata
from pathlib import Path

import skell_e_router


def test_version_matches_distribution_metadata():
    try:
        expected = importlib.metadata.version("skell-e-router")
    except importlib.metadata.PackageNotFoundError:
        expected = "0.0.0+uninstalled"
    assert skell_e_router.__version__ == expected


def test_no_top_level_version_literal_in_source():
    """A top-level `__version__ = "x.y.z"` literal is exactly what drifted;
    only the metadata-derived assignment (inside try/except) is allowed."""
    tree = ast.parse(Path(skell_e_router.__file__).read_text(encoding="utf-8-sig"))
    literals = [n for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)]
    assert not literals, "__version__ must come from importlib.metadata, not a hardcoded literal"


def test_fallback_version_when_distribution_missing(monkeypatch):
    import importlib
    real_version = importlib.metadata.version

    def fake_version(name):
        if name == "skell-e-router":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    module = importlib.reload(skell_e_router)
    try:
        assert module.__version__ == "0.0.0+uninstalled"
    finally:
        monkeypatch.undo()
        importlib.reload(skell_e_router)
