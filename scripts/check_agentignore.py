#!/usr/bin/env python3
"""Fail when a path matches the repo-root ``.agentignore`` file.

Gitignore syntax: comments, ``*``, ``?``, ``**``, character classes, ``!``
negation, and a trailing slash. A matching directory covers everything inside
it. Negation cannot bring back a file whose parent directory is ignored.

The pre-commit hook runs this against the index. A path is blocked when either
``HEAD:.agentignore`` or the staged ``.agentignore`` ignores it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_AGENTIGNORE = ".agentignore"
_MAX_SHOWN = 40
_DOUBLE_STAR_DIR = "(?:[^/]+/)*"
_MISSING = "agentignore: .agentignore is missing; refusing to continue.\n"
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Rule:
    """One compiled ``.agentignore`` pattern."""

    source: str
    regex: re.Pattern[str]
    negation: bool


def parse_agentignore(
    text: str,
    *,
    flags: re.RegexFlag = re.NOFLAG,
) -> tuple[Rule, ...]:
    """Compile gitignore-style ``text`` into rules, in file order."""
    rules: list[Rule] = []
    for raw_line in text.lstrip("\ufeff").splitlines():
        parsed = _parse_line(raw_line)
        if parsed is None:
            continue
        pattern, negation = parsed
        rules.append(_compile_rule(pattern, negation=negation, flags=flags))
    return tuple(rules)


def is_ignored(path: str, rules: Sequence[Rule]) -> bool:
    """Return whether ``path`` (repo-relative, slash-separated) is ignored."""
    return _evaluator(rules)(_normalize_relative(path))


def main(argv: Sequence[str] | None = None) -> int:
    """Check explicit paths, staged paths, or both. Return a process status."""
    parser = argparse.ArgumentParser(
        description="Block paths that .agentignore tells agents not to touch.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Repo-relative paths to check. With none, staged changes are checked.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check paths staged for commit, including renames and deletes.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root. Defaults to the parent of this script.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    code, message = check(args.root, staged=args.staged, paths=args.paths)
    if message:
        sys.stderr.write(message)
    return code


def check(
    root: Path,
    *,
    staged: bool,
    paths: Sequence[str],
) -> tuple[int, str]:
    """Return ``(status, stderr message)`` for ``paths`` under ``root``."""
    rule_sets = load_rule_sets(root)
    if rule_sets is None:
        return 1, _MISSING
    selected = [str(path) for path in paths]
    if staged or not selected:
        staged_paths = read_staged_paths(root)
        if staged_paths is None:
            return 1, "agentignore: could not read staged paths from git.\n"
        selected = [*staged_paths, *selected]
    normalized = _dedupe(_normalize_user_path(path, root) for path in selected)
    blocked = blocked_paths(normalized, rule_sets)
    if not blocked:
        return 0, ""
    return 1, _format_blocked(blocked)


def load_rule_sets(root: Path) -> tuple[tuple[Rule, ...], ...] | None:
    """Load HEAD, index, and worktree copies. ``None`` when none exist."""
    flags = re.IGNORECASE if _ignore_case(root) else re.NOFLAG
    texts: list[str] = []
    for spec in (f"HEAD:{_AGENTIGNORE}", f":{_AGENTIGNORE}"):
        shown = _git_show(root, spec)
        if shown is not None and shown not in texts:
            texts.append(shown)
    worktree = _read_worktree(root)
    if worktree is not None and worktree not in texts:
        texts.append(worktree)
    if not texts:
        return None
    return tuple(parse_agentignore(text, flags=flags) for text in texts)


def read_staged_paths(root: Path) -> list[str] | None:
    """Return staged add/modify/delete/rename paths, or ``None`` on git failure."""
    if not _inside_work_tree(root):
        return None
    result = _run_git(root, "diff", "--cached", "--name-status", "-z")
    if result.returncode != 0:
        return None
    return _parse_name_status(result.stdout)


def blocked_paths(
    paths: Sequence[str],
    rule_sets: Sequence[Sequence[Rule]],
) -> list[str]:
    """Return ``paths`` ignored by any ruleset, preserving order."""
    evaluators = [_evaluator(rules) for rules in rule_sets]
    return [
        path
        for path in paths
        if any(evaluate(path) for evaluate in evaluators)
    ]


def _evaluator(rules: Sequence[Rule]) -> Callable[[str], bool]:
    cache: dict[str, bool] = {}

    def evaluate(candidate: str) -> bool:
        if candidate in cache:
            return cache[candidate]
        ignored = False
        parts = candidate.split("/")
        parents = ["/".join(parts[:index]) for index in range(1, len(parts))]
        for rule in rules:
            if rule.regex.search(candidate) is None:
                continue
            if rule.negation:
                # Git will not re-include a file under an excluded directory.
                if any(evaluate(parent) for parent in parents):
                    continue
                ignored = False
            else:
                ignored = True
        cache[candidate] = ignored
        return ignored

    return evaluate


def _parse_line(raw_line: str) -> tuple[str, bool] | None:
    line = _strip_trailing_space(raw_line.rstrip("\r\n"))
    if not line or line.startswith("#"):
        return None
    negation = False
    if line.startswith(("\\#", "\\!")):
        line = line[1:]
    elif line.startswith("!"):
        negation = True
        line = line[1:]
    if not line:
        return None
    line = line.removesuffix("/")
    if not line:
        return None
    return line, negation


def _strip_trailing_space(line: str) -> str:
    end = len(line)
    while end > 0 and line[end - 1] == " ":
        if end > 1 and line[end - 2] == "\\":
            break
        end -= 1
    trimmed = line[:end]
    if trimmed.endswith("\\ "):
        return trimmed[:-2] + " "
    return trimmed


def _compile_rule(
    pattern: str,
    *,
    negation: bool,
    flags: re.RegexFlag,
) -> Rule:
    anchored = pattern.startswith("/") or "/" in pattern
    body = _translate(pattern.removeprefix("/"))
    prefix = "^" if anchored else "(?:^|/)"
    expression = f"{prefix}{body}(?:/.*)?$"
    return Rule(
        source=pattern,
        regex=re.compile(expression, flags),
        negation=negation,
    )


def _translate(pattern: str) -> str:
    parts = pattern.split("/")
    pieces: list[str] = []
    for index, part in enumerate(parts):
        if part == "**":
            if index == len(parts) - 1:
                pieces.append(".*")
            else:
                pieces.append(_DOUBLE_STAR_DIR)
            continue
        pieces.append(_translate_segment(part))
    rendered: list[str] = []
    for index, piece in enumerate(pieces):
        if index > 0 and pieces[index - 1] != _DOUBLE_STAR_DIR:
            rendered.append("/")
        rendered.append(piece)
    return "".join(rendered)


def _translate_segment(segment: str) -> str:
    rendered: list[str] = []
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if char == "*":
            rendered.append("[^/]*")
            while index < length and segment[index] == "*":
                index += 1
            continue
        if char == "?":
            rendered.append("[^/]")
            index += 1
            continue
        if char == "[":
            character_class = _read_class(segment, index)
            if character_class is None:
                rendered.append(re.escape(char))
                index += 1
                continue
            expression, index = character_class
            rendered.append(expression)
            continue
        if char == "\\" and index + 1 < length:
            rendered.append(re.escape(segment[index + 1]))
            index += 2
            continue
        rendered.append(re.escape(char))
        index += 1
    return "".join(rendered)


def _read_class(segment: str, start: int) -> tuple[str, int] | None:
    index = start + 1
    length = len(segment)
    if index >= length:
        return None
    inner: list[str] = []
    if segment[index] in "!^":
        inner.append("^")
        index += 1
    if index < length and segment[index] == "]":
        inner.append("]")
        index += 1
    closed = False
    while index < length:
        char = segment[index]
        if char == "]":
            closed = True
            index += 1
            break
        if char == "\\" and index + 1 < length:
            inner.append(re.escape(segment[index + 1]))
            index += 2
            continue
        inner.append(char)
        index += 1
    if not closed:
        return None
    return "[" + "".join(inner) + "]", index


def _format_blocked(paths: Sequence[str]) -> str:
    shown = list(paths[:_MAX_SHOWN])
    lines = [
        "agentignore: commit blocked. These paths match .agentignore "
        "and must not be changed:",
        *[f"  {path}" for path in shown],
    ]
    extra = len(paths) - len(shown)
    if extra:
        lines.append(f"  ... and {extra} more")
    return "\n".join(lines) + "\n"


def _normalize_user_path(path: str, root: Path) -> str:
    text = path.replace("\\", "/").strip().removeprefix("./")
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            text = candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            text = candidate.as_posix()
    return _normalize_relative(text)


def _normalize_relative(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")


def _dedupe(paths: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _read_worktree(root: Path) -> str | None:
    path = root / _AGENTIGNORE
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _git_show(root: Path, spec: str) -> str | None:
    if not _inside_work_tree(root):
        return None
    result = _run_git(root, "show", spec)
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", "surrogateescape")


def _ignore_case(root: Path) -> bool:
    if not _inside_work_tree(root):
        return False
    result = _run_git(root, "config", "--bool", "core.ignorecase")
    if result.returncode != 0:
        return False
    return result.stdout.strip() == b"true"


def _inside_work_tree(root: Path) -> bool:
    result = _run_git(root, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == b"true"


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    git = shutil.which("git")
    if git is None:
        return subprocess.CompletedProcess(args=["git", *args], returncode=1, stdout=b"", stderr=b"")
    return subprocess.run(  # noqa: S603 - git path comes from shutil.which; args are fixed flags
        [git, "-C", str(root), *args],
        check=False,
        capture_output=True,
    )


def _parse_name_status(payload: bytes) -> list[str]:
    records = payload.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        status = records[index]
        if not status:
            break
        code = status[:1]
        index += 1
        if index >= len(records) or not records[index]:
            break
        paths.append(records[index].decode("utf-8", "surrogateescape"))
        index += 1
        if code in b"RC" and index < len(records) and records[index]:
            paths.append(records[index].decode("utf-8", "surrogateescape"))
            index += 1
    return paths


if __name__ == "__main__":
    sys.exit(main())
