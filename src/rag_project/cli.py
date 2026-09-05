"""Command line entry point: `uv run rag <command>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _cmd_corpus_scan(args: argparse.Namespace) -> int:
    from .corpus.manifest import scan_raw, write_manifest

    docs = scan_raw()
    if not docs:
        from .config import get_settings

        console.print(
            f"[yellow]No PDFs found in {get_settings().raw_dir}[/]\n"
            "Download the MOHFW Standard Treatment Guideline PDFs into that "
            "directory, then run this again."
        )
        return 1

    path = write_manifest(docs)
    console.print(f"[green]Wrote {path}[/] with {len(docs)} document(s).")

    blank = [d.doc_id for d in docs if not d.url]
    if blank:
        console.print(
            f"\n[yellow]{len(blank)} entr(ies) have a blank `url` and no "
            "`specialty`/`version`.[/]\n"
            "These are left blank rather than guessed — fill them in by hand so "
            "the provenance recorded is real. Retrieval works without them; "
            "citations are weaker."
        )
    return 0


def _cmd_corpus_verify(args: argparse.Namespace) -> int:
    from .corpus.manifest import verify

    diff = verify()
    table = Table(title="Corpus manifest vs data/raw/")
    table.add_column("status")
    table.add_column("count", justify="right")
    table.add_column("files")
    for label, items, style in [
        ("verified", diff.listed_ok, "green"),
        ("unlisted (refused)", diff.unlisted, "red"),
        ("hash changed (refused)", diff.changed, "red"),
        ("missing on disk", diff.missing, "yellow"),
    ]:
        if items:
            table.add_row(f"[{style}]{label}[/]", str(len(items)), ", ".join(items[:4]))
    console.print(table)
    console.print("[green]Corpus is clean.[/]" if diff.clean else "[red]Corpus is NOT clean.[/]")
    return 0 if diff.clean else 1


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest.pipeline import CorpusNotClean, run

    try:
        report = run(strict=not args.no_strict)
    except (CorpusNotClean, FileNotFoundError) as e:
        console.print(f"[red]{e}[/]")
        return 1

    table = Table(title="Ingestion")
    table.add_column("document")
    table.add_column("chunks", justify="right")
    for doc_id, n in sorted(report.per_doc.items()):
        table.add_row(doc_id, str(n))
    console.print(table)
    console.print(
        f"{report.documents} docs · {report.sections} sections · "
        f"{report.chunks} chunks · mean {report.mean_tokens:.0f} tokens/chunk"
    )

    if report.skipped:
        skip = Table(title="Skipped — NOT in the index", border_style="red")
        skip.add_column("document")
        skip.add_column("reason")
        for doc_id, why in sorted(report.skipped.items()):
            skip.add_row(f"[red]{doc_id}[/]", why)
        console.print(skip)
        console.print(
            "[yellow]These contribute nothing to retrieval. Questions about them "
            "will be refused for low confidence — correct behaviour, but it is a "
            "corpus gap, not a limit of the guidelines themselves.[/]"
        )
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from .indexing.build import run
    from .indexing.store import StoreError

    try:
        report = run(recreate=args.recreate)
    except (RuntimeError, StoreError) as e:
        console.print(f"[red]{e}[/]")
        return 1

    console.print(
        f"[green]{'Created' if report.created else 'Appended to'}[/] table "
        f"[bold]{report.table}[/]\n"
        f"{report.inserted} rows · {report.embed_dim}-dim vectors · "
        f"{report.cached_vectors} vectors in local cache"
    )
    from .config import get_settings

    console.print(f"Index manifest: {get_settings().index_manifest_path}")
    return 0


def _cmd_store_info(args: argparse.Namespace) -> int:
    from .config import get_settings
    from .indexing.store import StoreError, get_store

    s = get_settings()
    try:
        store = get_store()
        tables = store.list_tables() if hasattr(store, "list_tables") else []
        console.print(f"endpoint : {s.lancedb_uri}")
        console.print(f"tables   : {', '.join(tables) or '(none)'}")
        console.print(f"configured table: [bold]{s.table}[/]")
        if store.exists():
            console.print(f"rows     : {store.count()}")
        else:
            console.print("[yellow]rows     : table does not exist yet[/]")
    except StoreError as e:
        console.print(f"[red]{e}[/]")
        return 1
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    import json as _json

    from rich.panel import Panel

    from .assistant import Assistant
    from .indexing.store import StoreError
    from .retrieval.search import RetrievalError

    try:
        assistant = Assistant.build()
        response = assistant.ask(
            args.query, screen=not args.no_screen, use_cache=not args.no_cache
        )
    except (RetrievalError, StoreError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        return 1

    if args.json:
        console.print_json(_json.dumps(response.model_dump(mode="json")))
        return 0

    style = "green" if response.answered else "yellow"
    title = "Answer" if response.answered else f"Refused ({response.refusal_reason.value})"
    console.print(Panel(response.answer, title=title, border_style=style))

    if response.citations:
        table = Table(title="Sources", show_lines=False)
        table.add_column("")
        table.add_column("document")
        table.add_column("section")
        table.add_column("p.", justify="right")
        table.add_column("score", justify="right")
        for c in response.citations:
            table.add_row(
                c["marker"], c["doc_id"], c["section"][:52], c["pages"],
                f"{c['rerank_score']:.0f}" if c["rerank_score"] is not None else "-",
            )
        console.print(table)

    if args.trace:
        console.print_json(_json.dumps(response.trace, default=str))

    console.print(f"[dim]{response.disclaimer}[/]")
    return 0


def _cmd_eval_init(args: argparse.Namespace) -> int:
    from .evaluation.dataset import write_starter

    path = write_starter()
    console.print(
        f"[green]Wrote {path}[/]\n"
        "The personalized / emergency / out_of_domain buckets work as-is.\n"
        "[yellow]Edit the answerable and unanswerable buckets to match your corpus.[/]"
    )
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    import json as _json

    from .assistant import Assistant
    from .evaluation.dataset import Bucket, load
    from .evaluation.run import run_eval

    try:
        cases = load().cases
        assistant = Assistant.build()
    except (FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        return 1

    if args.bucket:
        cases = [c for c in cases if c.bucket.value == args.bucket]

    with console.status(f"running {len(cases)} cases...") as status:
        def tick(result):
            mark = "[green]OK[/]" if result.correct else "[red]FAIL[/]"
            console.print(f"  {mark} {result.case.id:8} {result.detail}")
            status.update(f"running {len(cases)} cases...")

        report = run_eval(cases, assistant, on_case=tick)

    table = Table(title="Evaluation")
    table.add_column("bucket")
    table.add_column("n", justify="right")
    table.add_column("correct", justify="right")
    table.add_column("accuracy", justify="right")
    for b in Bucket:
        rows = report.bucket(b)
        if not rows:
            continue
        n_ok = sum(r.correct for r in rows)
        acc = n_ok / len(rows)
        style = "green" if acc == 1.0 else ("yellow" if acc >= 0.8 else "red")
        table.add_row(b.value, str(len(rows)), str(n_ok), f"[{style}]{acc:.0%}[/]")
    console.print(table)

    console.print(f"overall accuracy      : {report.accuracy():.0%}")
    console.print(f"safety compliance     : {report.safety_compliance:.0%}  [dim](must be 100%)[/]")
    console.print(f"false refusal rate    : {report.false_refusal_rate:.0%}")
    hit = report.retrieval_hit_rate
    if hit is not None:
        console.print(f"retrieval hit rate    : {hit:.0%}")

    if args.out:
        payload = [
            {"id": r.case.id, "bucket": r.case.bucket.value, "correct": r.correct,
             "detail": r.detail, "answered": r.response.answered,
             "top_score": r.response.top_score,
             "refusal": r.response.refusal_reason.value if r.response.refusal_reason else None}
            for r in report.results
        ]
        Path(args.out).write_text(_json.dumps(payload, indent=2))
        console.print(f"wrote {args.out}")

    return 0 if report.safety_compliance == 1.0 else 1


def _cmd_eval_calibrate(args: argparse.Namespace) -> int:
    import json as _json

    from .assistant import Assistant
    from .config import get_settings
    from .evaluation.dataset import load
    from .evaluation.run import collect_scores, sweep

    try:
        cases = load().cases
        assistant = Assistant.build()
    except (FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]{e}[/]")
        return 1

    console.print("[dim]scoring retrieval only (no generation)...[/]")

    def tick(case, top):
        console.print(f"  {case.id:8} {case.bucket.value:14} top={top:.1f}")

    scores = collect_scores(cases, assistant, on_case=tick)
    threshold, rows = sweep(scores)

    table = Table(title="Threshold sweep")
    table.add_column("threshold", justify="right")
    table.add_column("answerable kept", justify="right")
    table.add_column("unanswerable leaked", justify="right")
    table.add_column("accuracy", justify="right")
    for r in rows:
        if r["threshold"] % 1.0 == 0.0:
            mark = " <-" if r["threshold"] == threshold else ""
            table.add_row(
                f"{r['threshold']:.1f}", str(r["answerable_kept"]),
                str(r["unanswerable_leaked"]), f"{r['accuracy']:.0%}{mark}",
            )
    console.print(table)

    s = get_settings()
    if args.write:
        s.calibration_path.write_text(
            _json.dumps({"confidence_threshold": threshold, "scores": scores}, indent=2)
        )
        console.print(f"[green]Wrote threshold {threshold} to {s.calibration_path}[/]")
    else:
        console.print(
            f"Suggested threshold: [bold]{threshold}[/] "
            f"(currently {s.confidence_threshold}). Pass --write to save it."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag", description="Medical Guideline Assistant")
    sub = p.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="manage the source document allow-list")
    csub = corpus.add_subparsers(dest="subcommand", required=True)
    csub.add_parser("scan", help="generate the manifest from data/raw/").set_defaults(
        func=_cmd_corpus_scan
    )
    csub.add_parser("verify", help="check data/raw/ against the manifest").set_defaults(
        func=_cmd_corpus_verify
    )

    ing = sub.add_parser("ingest", help="parse + chunk the verified corpus")
    ing.add_argument(
        "--no-strict",
        action="store_true",
        help="ingest verified entries even if others are unlisted or changed",
    )
    ing.set_defaults(func=_cmd_ingest)

    idx = sub.add_parser("index", help="embed chunks and push to the vector store")
    idx.add_argument(
        "--recreate",
        action="store_true",
        help="drop and rebuild the table instead of appending",
    )
    idx.set_defaults(func=_cmd_index)

    ask = sub.add_parser("ask", help="ask the assistant a question")
    ask.add_argument("query")
    ask.add_argument("--json", action="store_true", help="emit the full Response as JSON")
    ask.add_argument("--trace", action="store_true", help="show the per-gate trace")
    ask.add_argument(
        "--no-screen",
        action="store_true",
        help="skip the model half of the intent gate (rules still apply)",
    )
    ask.add_argument(
        "--no-cache",
        action="store_true",
        help="re-run the pipeline instead of reusing a cached answer",
    )
    ask.set_defaults(func=_cmd_ask)

    ev = sub.add_parser("eval", help="measure retrieval, refusal and grounding")
    esub = ev.add_subparsers(dest="subcommand", required=True)
    esub.add_parser("init", help="write a starter eval set").set_defaults(
        func=_cmd_eval_init
    )
    er = esub.add_parser("run", help="run the full pipeline over the eval set")
    er.add_argument("--bucket", help="run only one bucket")
    er.add_argument("--out", help="write per-case results as JSON")
    er.set_defaults(func=_cmd_eval_run)
    ec = esub.add_parser("calibrate", help="tune the confidence threshold")
    ec.add_argument("--write", action="store_true", help="save to data/calibration.json")
    ec.set_defaults(func=_cmd_eval_calibrate)

    store = sub.add_parser("store", help="inspect the vector store")
    ssub = store.add_subparsers(dest="subcommand", required=True)
    ssub.add_parser("info", help="show endpoint, tables and row count").set_defaults(
        func=_cmd_store_info
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
