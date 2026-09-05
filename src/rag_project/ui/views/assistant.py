"""Assistant page.

Imports are absolute on purpose: Streamlit executes a page script as a
top-level script rather than as a member of its package, so relative imports
have no parent package to resolve against and raise ImportError at load.

Each question is independent: the assistant has no conversational memory, so the
UI is a question box with history beneath rather than a chat. Presenting it as a
chat would imply follow-ups like "and in children?" carry context, when in fact
they are sent as standalone queries and usually refused.
"""

from __future__ import annotations

import time

import streamlit as st

from rag_project.guardrails.policy import DISCLAIMER
from rag_project.models import Response
from rag_project.ui.components import render_answer, render_pipeline, render_sources
from rag_project.ui.resources import get_assistant, index_info

EXAMPLES = [
    "How is tuberculosis diagnosed?",
    "What are the referral criteria for severe acute malnutrition?",
    "What does the guideline say about management of snake bite?",
    "Should I take rifampicin for my cough?",
    "What is the treatment for bipolar disorder?",
]

if "history" not in st.session_state:
    st.session_state.history = []          # list[tuple[Response, float]]
if "pending" not in st.session_state:
    st.session_state.pending = None


# --- sidebar -------------------------------------------------------------
info = index_info()

with st.sidebar:
    st.markdown("### Corpus")
    counts = info.get("counts", {})
    col_a, col_b = st.columns(2)
    col_a.metric("documents", counts.get("documents", "—"))
    col_b.metric("chunks", counts.get("chunks", "—"))
    st.caption(
        f"table `{info['table']}` · threshold {info['threshold']}\n\n"
        f"answers: `{info['generation_model']}` · guards: `{info['guard_model']}`\n\n"
        f"embeddings: `{info['embedding_model']}`"
    )
    if info.get("built_at"):
        st.caption(f"indexed {info['built_at']}")

    with st.expander(f"Documents ({len(info.get('documents', []))})"):
        for d in info.get("documents", []):
            st.markdown(
                f"<div style='font-size:.82em;padding:.15rem 0;'>{d.get('title', d['doc_id'])}</div>",
                unsafe_allow_html=True,
            )

    skipped = info.get("skipped_documents") or []
    if skipped:
        with st.expander(f"Not indexed ({len(skipped)})"):
            st.caption(
                "In the corpus but contributing no chunks — questions relying "
                "solely on these will be refused for low confidence."
            )
            for d in skipped:
                st.markdown(
                    f"<div style='font-size:.82em;padding:.15rem 0;opacity:.7;'>"
                    f"{d.get('title', d['doc_id'])} <code>{d.get('filename','')}</code></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Scope"):
        st.markdown(
            "Answers come **only** from the indexed government guidelines, with "
            "citations.\n\nRefused: personalized advice, diagnosis, dosing for a "
            "person, emergencies, and anything the corpus does not cover."
        )

    if st.session_state.history and st.button("Clear history", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# --- header --------------------------------------------------------------
st.title("🩺 Medical Guideline Assistant")
st.caption(
    "Grounded answers from official MOHFW Standard Treatment Guidelines. "
    "Every claim is cited; unsupported questions are refused."
)


def _ask(query: str) -> None:
    st.session_state.pending = query


with st.form("ask", clear_on_submit=False):
    query = st.text_area(
        "Your question",
        placeholder="What do the guidelines say about the diagnosis of tuberculosis?",
        height=90,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
    if submitted and query.strip():
        _ask(query.strip())

st.caption("Try one:")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES, strict=True):
    with col:
        st.button(
            example if len(example) < 26 else example[:23] + "…",
            key=f"ex-{example}",
            help=example,
            use_container_width=True,
            on_click=_ask,
            args=(example,),
        )


# --- run -----------------------------------------------------------------
if st.session_state.pending:
    pending = st.session_state.pending
    st.session_state.pending = None
    with st.spinner("Running the pipeline…"):
        started = time.perf_counter()
        try:
            response: Response = get_assistant().ask(pending)
        except Exception as exc:  # surfaced, not swallowed
            st.exception(exc)
            st.stop()
        elapsed = time.perf_counter() - started
    st.session_state.history.insert(0, (response, elapsed))


# --- results -------------------------------------------------------------
for i, (response, elapsed) in enumerate(st.session_state.history):
    st.divider()
    st.markdown(f"##### {response.query}")
    render_answer(response)
    st.write("")
    render_sources(response)
    st.write("")
    render_pipeline(response, elapsed)
    if i == 0:
        st.caption(f"⚠️ {DISCLAIMER}")

if not st.session_state.history:
    st.info(
        "Ask a question above. The **Pipeline** panel under each answer shows "
        "which gate made the decision — including when a question is refused "
        "before any retrieval happens.",
        icon="💡",
    )
