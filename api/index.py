"""Vercel serverless entry point.

Vercel treats each module under `api/` as a function and looks for an ASGI
`app`, so this exists only to make `rag_project` importable and re-export the
real application from rag_project.api.

`src` is put on the path rather than installing the package: Vercel builds from
requirements.txt, and asking pip to build a `uv_build`-backed project inside the
build container is a second thing that can fail for no benefit here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_project.api import app  # noqa: E402

__all__ = ["app"]
