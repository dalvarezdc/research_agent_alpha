"""CLI: convert a document to markdown.

Usage:
    uv run python -m document_parser <file> [-o out.md] [--format pdf]

Prints markdown to stdout (or writes to ``-o``). Warnings go to stderr. Exit
code is 0 on SUCCESS, 1 on PARTIAL or FAILED.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import ParseStatus
from .service import parse_document, supported_extensions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="document_parser",
        description="Convert a document (PDF/Word/text/rtf) to markdown.",
    )
    parser.add_argument("file", help="Path to the document to convert.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write markdown to this file instead of stdout.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help=(
            "Override the format (e.g. 'pdf'). Supported: "
            f"{', '.join(supported_extensions())}."
        ),
    )
    args = parser.parse_args(argv)

    try:
        result = parse_document(args.file, content_type=args.format)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.output:
        Path(args.output).write_text(result.markdown, encoding="utf-8")
        print(
            f"{result.status.value}: wrote {len(result.markdown)} chars to {args.output}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(result.markdown)
        if result.markdown and not result.markdown.endswith("\n"):
            sys.stdout.write("\n")

    return 0 if result.status is ParseStatus.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
