"""Renderers for the assistant page.

The pipeline panel is the reason this UI exists. A chat box shows you an answer;
it cannot show you that a question was refused *before retrieval ran*, or that
the reranker scored every passage 0.0. Those are the decisions worth seeing, so
they get first-class treatment rather than a debug expander.
"""

from __future__ import annotations

import re

import streamlit as st

from ..models import Response

#: Stage key in Response.trace -> human label. Order is pipeline order.
STAGES: list[tuple[str, str]] = [
    ("intent", "Gate 1 · intent"),
    ("retrieve", "Retrieve"),
    ("rerank", "Rerank"),
    ("confidence", "Gate 2 · confidence"),
    ("generate", "Generate"),
    ("output_gate", "Gate 3 · output"),
]

_MARKER = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")


def highlight_citations(text: str) -> str:
    """Make [C1] markers visually distinct from the prose around them."""
    return _MARKER.sub(
        lambda m: (
            '<span style="background:rgba(56,139,253,.16);color:#2f81f7;'
            'border-radius:4px;padding:1px 6px;font-size:.82em;font-weight:600;'
            f'white-space:nowrap;">{m.group(1)}</span>'
        ),
        text,
    )


def _stage_state(key: str, trace: dict) -> tuple[str, str]:
    """(icon, summary) for one pipeline row."""
    reached = key in trace.get("stages", [])
    if not reached:
        return "○", "not reached"

    if key == "intent":
        it = trace.get("intent", {})
        label = it.get("label", "?")
        icon = "✓" if label == "in_scope" else "✕"
        rule = " · rule match" if it.get("rule") else " · model"
        return icon, f"{label}{rule} — {it.get('reason', '')}"

    if key == "retrieve":
        r = trace.get("retrieval", {})
        extra = ""
        if r.get("expansions"):
            extra = f" · expanded {', '.join(r['expansions'])}"
        return "✓", (
            f"{r.get('dense_hits', 0)} dense · {r.get('lexical_hits', 0)} lexical "
            f"→ {r.get('fused_pool', 0)} fused{extra}"
        )

    if key == "rerank":
        rr = trace.get("rerank", {})
        scores = [s for s in (rr.get("top_scores") or []) if s is not None]
        shown = ", ".join(f"{s:.0f}" for s in scores[:5]) or "no scores"
        return "✓", f"{rr.get('scored', 0)} scored · top: {shown}"

    if key == "confidence":
        c = trace.get("confidence", {})
        icon = "✓" if c.get("passed") else "✕"
        top = c.get("top_score")
        top_s = f"{top:.1f}" if isinstance(top, (int, float)) else "—"
        return icon, (
            f"top {top_s} vs threshold {c.get('threshold')} · "
            f"kept {c.get('kept', 0)} — {c.get('reason', '')}"
        )

    if key == "generate":
        g = trace.get("generation", {})
        if g.get("insufficient_context"):
            return "✕", "model reported insufficient context"
        return "✓", f"{g.get('claims', 0)} claims drafted"

    if key == "output_gate":
        o = trace.get("output_gate", {})
        return ("✓" if o.get("passed") else "✕"), o.get("detail", "")

    return "✓", ""


def render_pipeline(response: Response, elapsed: float | None = None) -> None:
    trace = response.trace or {}
    if not trace.get("stages"):
        return

    title = "Pipeline"
    if elapsed is not None:
        title += f"  ·  {elapsed:.1f}s"

    with st.expander(title, expanded=True):
        for key, label in STAGES:
            icon, summary = _stage_state(key, trace)
            colour = {"✓": "#3fb950", "✕": "#f85149", "○": "#6e7681"}[icon]
            dim = "opacity:.45;" if icon == "○" else ""
            st.markdown(
                f'<div style="display:flex;gap:.6rem;align-items:baseline;'
                f'padding:.28rem 0;border-bottom:1px solid rgba(128,128,128,.14);{dim}">'
                f'<span style="color:{colour};font-weight:700;width:1rem;">{icon}</span>'
                f'<span style="min-width:11rem;font-weight:600;">{label}</span>'
                f'<span style="opacity:.75;font-size:.9em;">{summary}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

        if trace.get("short_circuited"):
            st.caption(f"↳ {trace['short_circuited']}")


def render_sources(response: Response) -> None:
    if not response.citations:
        return
    st.markdown("**Sources**")
    for c in response.citations:
        score = c.get("rerank_score")
        score_s = f"{score:.0f}" if isinstance(score, (int, float)) else "—"
        url = c.get("source_url") or ""
        title = c.get("title") or c["doc_id"]
        link = f"  ·  [source]({url})" if url else ""
        st.markdown(
            f"`{c['marker']}`  **{title}**  \n"
            f"<span style='opacity:.7;font-size:.86em;'>§ {c['section']} · "
            f"p.{c['pages']} · relevance {score_s}/10{link}</span>",
            unsafe_allow_html=True,
        )


def render_answer(response: Response) -> None:
    if response.answered:
        st.markdown(
            f"<div style='line-height:1.65;'>{highlight_citations(response.answer)}</div>",
            unsafe_allow_html=True,
        )
    else:
        reason = (response.refusal_reason.value if response.refusal_reason else "refused")
        # Emergency gets the loudest treatment available; the rest are informational
        # rather than errors, because refusing is a correct outcome here.
        if reason == "emergency":
            st.error(response.answer, icon="🚨")
        else:
            st.warning(response.answer, icon="⛔")
        st.caption(f"refusal reason: `{reason}`")
