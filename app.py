"""Deployment entrypoint.

Vercel's Python runtime looks for a top-level `app` in one of a few known
filenames -- this is one of them -- and routes every request to it. Nothing
here is needed to run locally; use `rag-serve` for that.

The sys.path line is deliberate. This project uses a src/ layout, so
`rag_project` is only importable once the project itself is installed
(which `uv sync` does). Prepending src/ makes the import work either way,
rather than failing at cold start if the build only installed dependencies.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_project.api import app  # noqa: E402

__all__ = ["app"]
