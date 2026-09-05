"""Streamlit entry point. Run with `uv run rag-ui`.

Uses st.navigation rather than a pages/ directory so the nav labels are chosen
explicitly instead of being derived from filenames.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Medical Guideline Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

_VIEWS = Path(__file__).parent / "views"

navigation = st.navigation(
    [
        st.Page(_VIEWS / "assistant.py", title="Assistant", icon="🩺", default=True),
        st.Page(_VIEWS / "evaluation.py", title="Evaluation", icon="📊"),
    ]
)
navigation.run()
