"""Cached expensive objects.

Streamlit re-runs the whole script on every interaction. Assistant.build()
reads every chunk off disk and constructs the BM25 index over ~3,500 documents,
so without cache_resource that work would repeat on each keystroke. This is not
an optimisation -- uncached, the app is unusable.
"""

from __future__ import annotations

import json

import streamlit as st

from ..assistant import Assistant
from ..config import get_settings


@st.cache_resource(show_spinner="Loading corpus and building the lexical index…")
def get_assistant() -> Assistant:
    return Assistant.build()


@st.cache_data(ttl=30)
def index_info() -> dict:
    """Index manifest + live row count. Short TTL so a re-index shows up."""
    s = get_settings()
    info: dict = {
        "table": s.table,
        "embedding_model": s.openai_embed_model,
        "generation_model": s.openai_model,
        "guard_model": s.openai_guard_model,
        "threshold": s.confidence_threshold,
        "documents": [],
        "skipped_documents": [],
        "counts": {},
        "built_at": None,
    }
    if s.index_manifest_path.exists():
        manifest = json.loads(s.index_manifest_path.read_text())
        info["documents"] = manifest.get("documents", [])
        info["skipped_documents"] = manifest.get("skipped_documents", [])
        info["counts"] = manifest.get("counts", {})
        info["built_at"] = manifest.get("built_at")
        info["chunking"] = manifest.get("chunking", {})
    return info


@st.cache_data(ttl=10)
def eval_results() -> list[dict] | None:
    path = get_settings().eval_dir / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
