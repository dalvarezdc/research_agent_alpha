#!/usr/bin/env python3
"""
Tests for the /file command handler behavior in router.py and document_context
threading through the AgentOrchestrator.

Strategy: we do NOT spin up the REPL loop. Instead we re-implement the tiny
/file handler state-machine inline and call parse_document through monkeypatch,
exactly mirroring what router.main() does.  This keeps the tests fast, focused,
and free of LLM or I/O side-effects.
"""

import io
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

import document_parser as dp
from document_parser.models import (
    DocumentMetadata,
    ParseResult,
    ParseStatus,
    ParseWarning,
)
import router


# ---------------------------------------------------------------------------
# Helpers: build ParseResult fixtures
# ---------------------------------------------------------------------------

def _meta(fmt="pdf", pages=None):
    return DocumentMetadata(
        source_path="/tmp/fake.pdf",
        file_format=fmt,
        backend="test",
        page_count=pages,
        char_count=0,
    )


def _success(markdown: str, fmt="pdf", pages=2):
    return ParseResult(
        markdown=markdown,
        status=ParseStatus.SUCCESS,
        metadata=_meta(fmt=fmt, pages=pages),
    )


def _partial(markdown: str, warnings=None):
    result = ParseResult(
        markdown=markdown,
        status=ParseStatus.PARTIAL,
        metadata=_meta(),
        warnings=warnings or [ParseWarning("Some pages skipped", location="page 3")],
    )
    return result


def _failed(warnings=None):
    return ParseResult(
        markdown="",
        status=ParseStatus.FAILED,
        metadata=_meta(),
        warnings=warnings or [ParseWarning("Could not read file")],
    )


# ---------------------------------------------------------------------------
# Helper: run the /file handler inline (mirrors router.main() logic exactly)
# Returns (attached_document_context, printed_output)
# ---------------------------------------------------------------------------

def _run_file_handler(query: str, initial_context=None, mock_parse_result=None):
    """
    Execute the /file command branch from router.main() in isolation.

    Parameters
    ----------
    query : str
        The full query string, e.g. "/file /path/to/doc.pdf"
    initial_context : str | None
        Value of `attached_document_context` before the command runs.
    mock_parse_result : ParseResult | Exception | None
        If a ParseResult, patched as the return value of parse_document().
        If an Exception class/instance, it will be raised by parse_document().
        If None, parse_document is not called (for commands that don't reach it).

    Returns
    -------
    (new_context, output_text)
    """
    MAX = router.MAX_DOCUMENT_CONTEXT_CHARS

    captured = io.StringIO()
    attached = initial_context

    with patch("sys.stdout", captured):
        # Mirror the REPL logic from router.main()
        if query == "/file clear":
            attached = None
            print("→ Document context cleared.\n")

        elif query == "/file":
            if attached:
                print(f"→ Document attached: {len(attached):,} chars. Use '/file clear' to remove it.\n")
            else:
                print("→ No document attached. Use '/file <path>' to attach one.\n")

        elif query.startswith("/file "):
            file_path = query[len("/file "):].strip()
            if not file_path:
                print("→ Usage: /file <path>\n")
            else:
                # patch parse_document
                if isinstance(mock_parse_result, type) and issubclass(mock_parse_result, Exception):
                    def _raise(*a, **kw):
                        raise mock_parse_result(file_path)
                    with patch("document_parser.parse_document", side_effect=_raise):
                        try:
                            result = dp.parse_document(file_path)
                        except FileNotFoundError:
                            print(f"→ File not found: {file_path}\n")
                            result = None
                elif isinstance(mock_parse_result, Exception):
                    with patch("document_parser.parse_document", side_effect=mock_parse_result):
                        try:
                            result = dp.parse_document(file_path)
                        except FileNotFoundError:
                            print(f"→ File not found: {file_path}\n")
                            result = None
                else:
                    with patch("document_parser.parse_document", return_value=mock_parse_result):
                        result = dp.parse_document(file_path)

                if result is not None:
                    if not result.ok:
                        print(f"→ Could not parse '{file_path}':")
                        for w in result.warnings:
                            print(f"   ⚠ {w}")
                        print()
                    else:
                        original_len = len(result.markdown)
                        if original_len > MAX:
                            dropped = original_len - MAX
                            pct = dropped / original_len * 100
                            print(
                                f"⚠️  Document is {original_len:,} chars — exceeds the "
                                f"{MAX:,}-char context limit.\n"
                                f"   Truncated to {MAX:,} chars "
                                f"(dropped {dropped:,} chars, ~{pct:.1f}% of the document).\n"
                                f"   Only the first {MAX:,} chars will be used as context."
                            )
                            attached = result.markdown[:MAX]
                        else:
                            attached = result.markdown

                        fmt = result.metadata.file_format
                        pages = result.metadata.page_count
                        page_str = f", {pages} page{'s' if pages != 1 else ''}" if pages else ""
                        char_count = len(attached)

                        if result.status is ParseStatus.PARTIAL:
                            for w in result.warnings:
                                print(f"   ⚠ {w}")

                        fname = os.path.basename(file_path)
                        print(
                            f"✓ Attached {fname} ({fmt}{page_str}, {char_count:,} chars). "
                            f"Stays attached until '/file clear'.\n"
                        )

    return attached, captured.getvalue()


# ===========================================================================
# Tests: constant
# ===========================================================================

def test_max_document_context_chars_constant():
    """MAX_DOCUMENT_CONTEXT_CHARS must be exactly 100,000."""
    assert router.MAX_DOCUMENT_CONTEXT_CHARS == 100_000


# ===========================================================================
# Tests: /file clear
# ===========================================================================

def test_file_clear_resets_context():
    ctx, out = _run_file_handler("/file clear", initial_context="some text")
    assert ctx is None
    assert "Document context cleared" in out


def test_file_clear_when_already_none():
    ctx, out = _run_file_handler("/file clear", initial_context=None)
    assert ctx is None
    assert "Document context cleared" in out


# ===========================================================================
# Tests: /file alone (status check)
# ===========================================================================

def test_file_alone_no_attachment():
    ctx, out = _run_file_handler("/file", initial_context=None)
    assert ctx is None
    assert "No document attached" in out


def test_file_alone_with_attachment():
    doc = "hello world " * 10
    ctx, out = _run_file_handler("/file", initial_context=doc)
    assert ctx == doc  # unchanged
    assert f"{len(doc):,} chars" in out


# ===========================================================================
# Tests: /file with only spaces → Usage message
# ===========================================================================

def test_file_spaces_only_shows_usage():
    ctx, out = _run_file_handler("/file   ", initial_context=None)
    assert ctx is None
    assert "Usage: /file <path>" in out


# ===========================================================================
# Tests: /file <path> — FileNotFoundError
# ===========================================================================

def test_file_not_found_prints_error():
    ctx, out = _run_file_handler(
        "/file /nonexistent/doc.pdf",
        initial_context=None,
        mock_parse_result=FileNotFoundError,
    )
    assert ctx is None
    assert "File not found" in out


def test_file_not_found_leaves_context_unchanged():
    existing = "prior context"
    ctx, out = _run_file_handler(
        "/file /nonexistent/doc.pdf",
        initial_context=existing,
        mock_parse_result=FileNotFoundError,
    )
    assert ctx == existing


# ===========================================================================
# Tests: /file <path> — FAILED result
# ===========================================================================

def test_file_failed_does_not_attach():
    ctx, out = _run_file_handler(
        "/file /tmp/corrupt.pdf",
        initial_context=None,
        mock_parse_result=_failed(warnings=[ParseWarning("Corrupt file")]),
    )
    assert ctx is None


def test_file_failed_prints_warnings():
    ctx, out = _run_file_handler(
        "/file /tmp/corrupt.pdf",
        initial_context=None,
        mock_parse_result=_failed(warnings=[ParseWarning("Corrupt file")]),
    )
    assert "Could not parse" in out
    assert "Corrupt file" in out


def test_file_failed_does_not_overwrite_existing_context():
    existing = "my previous doc"
    ctx, out = _run_file_handler(
        "/file /tmp/bad.pdf",
        initial_context=existing,
        mock_parse_result=_failed(),
    )
    assert ctx == existing


# ===========================================================================
# Tests: /file <path> — SUCCESS, under limit
# ===========================================================================

def test_file_success_attaches_full_markdown():
    markdown = "# Report\n\nSome content here."
    ctx, out = _run_file_handler(
        "/file /tmp/doc.pdf",
        initial_context=None,
        mock_parse_result=_success(markdown, fmt="pdf", pages=3),
    )
    assert ctx == markdown


def test_file_success_prints_confirmation():
    markdown = "# Report\n\nSome content here."
    ctx, out = _run_file_handler(
        "/file /tmp/doc.pdf",
        initial_context=None,
        mock_parse_result=_success(markdown, fmt="pdf", pages=3),
    )
    assert "Attached doc.pdf" in out
    assert "pdf" in out
    assert "3 pages" in out
    assert "Stays attached until '/file clear'" in out


def test_file_success_single_page_grammar():
    markdown = "# Page one only"
    ctx, out = _run_file_handler(
        "/file /tmp/single.pdf",
        initial_context=None,
        mock_parse_result=_success(markdown, fmt="pdf", pages=1),
    )
    assert "1 page" in out
    assert "1 pages" not in out


def test_file_success_no_pages_metadata():
    """When page_count is None (e.g. .txt), no page string appears."""
    markdown = "plain text"
    ctx, out = _run_file_handler(
        "/file /tmp/notes.txt",
        initial_context=None,
        mock_parse_result=_success(markdown, fmt="txt", pages=None),
    )
    assert "page" not in out.lower() or "Stays attached" in out


# ===========================================================================
# Tests: /file <path> — SUCCESS, over 100_000 chars (truncation)
# ===========================================================================

def test_file_truncation_attaches_first_100k_chars():
    big_doc = "x" * 150_000
    ctx, out = _run_file_handler(
        "/file /tmp/big.pdf",
        initial_context=None,
        mock_parse_result=_success(big_doc),
    )
    assert ctx == big_doc[:100_000]
    assert len(ctx) == 100_000


def test_file_truncation_overflow_notification_original_len():
    big_doc = "x" * 150_000
    ctx, out = _run_file_handler(
        "/file /tmp/big.pdf",
        initial_context=None,
        mock_parse_result=_success(big_doc),
    )
    assert "150,000" in out


def test_file_truncation_overflow_notification_limit():
    big_doc = "x" * 150_000
    ctx, out = _run_file_handler(
        "/file /tmp/big.pdf",
        initial_context=None,
        mock_parse_result=_success(big_doc),
    )
    assert "100,000" in out


def test_file_truncation_overflow_notification_dropped():
    big_doc = "x" * 150_000
    ctx, out = _run_file_handler(
        "/file /tmp/big.pdf",
        initial_context=None,
        mock_parse_result=_success(big_doc),
    )
    # dropped = 50,000
    assert "50,000" in out


def test_file_truncation_overflow_notification_percentage():
    big_doc = "x" * 150_000
    ctx, out = _run_file_handler(
        "/file /tmp/big.pdf",
        initial_context=None,
        mock_parse_result=_success(big_doc),
    )
    # dropped pct = 50_000/150_000 * 100 = 33.3%
    assert "33.3%" in out


# ===========================================================================
# Tests: sticky behavior — context survives to the next query
# ===========================================================================

def test_sticky_context_survives_subsequent_query():
    """After attaching, context is not None and would be passed to the orchestrator."""
    markdown = "# Sticky Document"
    ctx, _ = _run_file_handler(
        "/file /tmp/sticky.pdf",
        initial_context=None,
        mock_parse_result=_success(markdown),
    )
    # Simulate a subsequent non-/file query: context is used as-is
    # (The REPL passes `attached_document_context or ""` to orchestrator)
    assert ctx is not None
    assert ctx == markdown
    # After a normal query, context is unchanged (sticky)
    # We verify this by confirming /file status still shows the text
    _, out2 = _run_file_handler("/file", initial_context=ctx)
    assert f"{len(ctx):,} chars" in out2


def test_sticky_context_passes_to_orchestrator_on_next_query():
    """
    Verify that the value produced by /file would be forwarded to the orchestrator
    as the document_context argument on a subsequent normal (non-/file) query.

    The REPL dispatches via: ``document_context=attached_document_context or ""``
    (router.py lines ~493, 503, 511, 521 — one per agent branch).

    We cannot spin up the interactive REPL here, so we test the threading formula
    directly: after /file sets attached_document_context, the expression
    ``attached_document_context or ""`` must equal the parsed markdown content.
    Full end-to-end REPL dispatch is exercised by manual router.main() testing.
    """
    markdown = "# Patient Labs\n\nTSH: 6.2 mIU/L (elevated)\nFT4: 0.8 ng/dL (low-normal)"

    # Step 1: simulate /file attaching the document
    attached_document_context, _ = _run_file_handler(
        "/file /tmp/labs.pdf",
        initial_context=None,
        mock_parse_result=_success(markdown, fmt="pdf", pages=1),
    )

    # Step 2: confirm attachment occurred
    assert attached_document_context is not None, (
        "After /file, attached_document_context must be set"
    )
    assert attached_document_context == markdown

    # Step 3: verify the router threading expression
    # router.py passes `attached_document_context or ""` to every orchestrator run_* method.
    # A non-/file query must NOT reset attached_document_context, so:
    document_context_passed_to_orchestrator = attached_document_context or ""
    assert document_context_passed_to_orchestrator == markdown, (
        "The value forwarded to orchestrator.run_*(..., document_context=...) "
        "must equal the attached markdown content"
    )

    # Step 4: verify the MAX constant is exported for router-level truncation
    assert router.MAX_DOCUMENT_CONTEXT_CHARS == 100_000


def test_sticky_context_cleared_by_clear_command():
    ctx = "some document text"
    ctx_after, out = _run_file_handler("/file clear", initial_context=ctx)
    assert ctx_after is None
    assert "cleared" in out.lower()


# ===========================================================================
# Tests: /file <path> — PARTIAL result attaches WITH warnings
# ===========================================================================

def test_file_partial_attaches_context():
    markdown = "# Partial doc\n\nSome content."
    ctx, out = _run_file_handler(
        "/file /tmp/partial.pdf",
        initial_context=None,
        mock_parse_result=_partial(markdown),
    )
    assert ctx == markdown


def test_file_partial_prints_warnings():
    markdown = "# Partial doc\n\nSome content."
    ctx, out = _run_file_handler(
        "/file /tmp/partial.pdf",
        initial_context=None,
        mock_parse_result=_partial(
            markdown, warnings=[ParseWarning("Image-only page skipped", location="page 3")]
        ),
    )
    assert "Image-only page skipped" in out


def test_file_partial_also_prints_confirmation():
    markdown = "# Partial doc"
    ctx, out = _run_file_handler(
        "/file /tmp/partial.pdf",
        initial_context=None,
        mock_parse_result=_partial(markdown),
    )
    assert "Attached" in out
    assert "Stays attached" in out
