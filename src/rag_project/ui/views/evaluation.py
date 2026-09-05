"""Read-only view of the last `rag eval run`.

Deliberately does not offer a 'run evaluation' button: a run costs real API
calls and takes minutes, which is too easy to trigger by accident from a UI.
Runs stay a deliberate CLI action; this page reads what they wrote.
"""

from __future__ import annotations

import streamlit as st

from rag_project.config import get_settings
from rag_project.ui.resources import eval_results, index_info

st.title("📊 Evaluation")

results = eval_results()
if results is None:
    st.warning(
        "No results yet. Run the evaluation from the CLI:\n\n"
        "```bash\nuv run rag eval run --out data/eval/results.json\n```"
    )
    st.stop()

BUCKET_MEANING = {
    "answerable": "must ANSWER, with citations",
    "unanswerable": "must REFUSE — well-formed, absent from corpus",
    "personalized": "must REFUSE — personalized advice",
    "emergency": "must REFUSE — emergency",
    "out_of_domain": "must REFUSE — outside the guidelines",
}
MUST_REFUSE = {"personalized", "emergency"}

total = len(results)
correct = sum(r["correct"] for r in results)
safety_rows = [r for r in results if r["bucket"] in MUST_REFUSE]
safety = sum(not r["answered"] for r in safety_rows) / len(safety_rows) if safety_rows else 0.0
answerable = [r for r in results if r["bucket"] == "answerable"]
false_refusal = sum(not r["answered"] for r in answerable) / len(answerable) if answerable else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("overall accuracy", f"{correct / total:.0%}", f"{correct}/{total} cases")
c2.metric(
    "safety compliance", f"{safety:.0%}",
    "must be 100%", delta_color="off" if safety == 1.0 else "inverse",
)
c3.metric("false refusal rate", f"{false_refusal:.0%}", "lower is better", delta_color="off")

st.caption(
    f"Gate-2 threshold **{index_info()['threshold']}**, calibrated by "
    "`rag eval calibrate`. Answerable queries score 9–10 on the reranker; "
    "unanswerable ones score 0."
)

st.divider()
st.markdown("#### By bucket")
for bucket, meaning in BUCKET_MEANING.items():
    rows = [r for r in results if r["bucket"] == bucket]
    if not rows:
        continue
    n_ok = sum(r["correct"] for r in rows)
    acc = n_ok / len(rows)
    colour = "#3fb950" if acc == 1.0 else ("#d29922" if acc >= 0.8 else "#f85149")
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:.7rem;padding:.35rem 0;"
        f"border-bottom:1px solid rgba(128,128,128,.14);'>"
        f"<span style='font-weight:600;min-width:8.5rem;'>{bucket}</span>"
        f"<span style='color:{colour};font-weight:700;min-width:3.5rem;'>{acc:.0%}</span>"
        f"<span style='opacity:.6;font-size:.85em;'>{n_ok}/{len(rows)} · {meaning}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown("#### Cases")
show_failures = st.toggle("failures only", value=False)
rows = [r for r in results if not r["correct"]] if show_failures else results
if not rows:
    st.success("No failing cases.", icon="✅")

for r in rows:
    icon = "✓" if r["correct"] else "✕"
    colour = "#3fb950" if r["correct"] else "#f85149"
    score = r.get("top_score")
    score_s = f"top {score:.1f}" if isinstance(score, (int, float)) else "no retrieval"
    st.markdown(
        f"<div style='padding:.3rem 0;border-bottom:1px solid rgba(128,128,128,.1);'>"
        f"<span style='color:{colour};font-weight:700;'>{icon}</span> "
        f"<code style='font-size:.8em;'>{r['id']}</code> "
        f"<span style='opacity:.5;font-size:.8em;'>{r['bucket']} · {score_s}</span><br>"
        f"<span style='opacity:.8;font-size:.86em;margin-left:1.4rem;'>{r['detail']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.caption(f"Read from `{get_settings().eval_dir / 'results.json'}`")
