"""Parser and pre-commit gate for the repo-root .agentignore file."""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_agentignore",
    _ROOT / "scripts" / "check_agentignore.py",
)
if _SPEC is None or _SPEC.loader is None:
    msg = "Could not load scripts/check_agentignore.py"
    raise ImportError(msg)
check_agentignore = importlib.util.module_from_spec(_SPEC)
sys.modules["check_agentignore"] = check_agentignore
_SPEC.loader.exec_module(check_agentignore)


def _rules(text: str):
    return check_agentignore.parse_agentignore(text)


def _ignored(text: str, path: str) -> bool:
    return check_agentignore.is_ignored(path, _rules(text))


def test_repo_agentignore_matches_this_layout() -> None:
    text = (_ROOT / ".agentignore").read_text(encoding="utf-8")
    rules = check_agentignore.parse_agentignore(text)
    ignored = [
        ".env",
        ".env.dev",
        "subdir/.env.local",
        "credentials",
        "credentials.json",
        "secrets/my_credentials.txt",
        "credentials/token.txt",
        "data/app_config.json",
        "data/app_config.json.tmp",
        "pkg/__pycache__/mod.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".mypy_cache/3.12/pkg.meta",
        "cache/reference_validation.db",
        "cache/reference_validation.db-wal",
        ".venv/lib/python/site.py",
        "design_system/node_modules/react/index.js",
        "dist/pkg.whl",
        "design_system/dist/index.html",
        "build/lib/foo.py",
        "pkg/foo.egg-info/PKG-INFO",
        "outputs/run/patient_report.md",
    ]
    allowed = [
        "AGENTS.md",
        "app_config.py",
        "tests/test_app_config.py",
        "scripts/check_agentignore.py",
        ".githooks/pre-commit",
        "router.py",
        "langchain_agents/base.py",
        "outputs_test/report.md",
        "cache/readme.txt",
        "reference_validation/cache/cache_manager.py",
        "docs/diagrams/build_diagrams.py",
        "design_system/src/App.tsx",
        "frontend/app.js",
        "data/app.db",
    ]
    missed = [path for path in ignored if not check_agentignore.is_ignored(path, rules)]
    blocked = [path for path in allowed if check_agentignore.is_ignored(path, rules)]
    assert missed == []
    assert blocked == []


def test_comments_blanks_and_trailing_space_are_ignored() -> None:
    text = "\n# comment\noutputs/   \n"
    assert _ignored(text, "outputs/a.md")
    assert not _ignored(text, "outputs_test/a.md")


def test_anchored_cache_db_does_not_match_nested_or_other_trees() -> None:
    text = "cache/*.db\n"
    assert _ignored(text, "cache/reference_validation.db")
    assert not _ignored(text, "cache/nested/reference_validation.db")
    assert not _ignored(text, "reference_validation/cache/foo.db")
    assert not _ignored(text, "cache/readme.txt")


def test_directory_pattern_matches_anywhere_and_not_a_prefix() -> None:
    text = "dist/\nbuild/\n"
    assert _ignored(text, "dist/wheel.whl")
    assert _ignored(text, "design_system/dist/assets/index.js")
    assert _ignored(text, "build/lib/mod.py")
    assert not _ignored(text, "docs/diagrams/build_diagrams.py")
    assert not _ignored(text, "distance/readme.md")


def test_negation_reincludes_a_file_but_not_under_an_excluded_directory() -> None:
    assert not _ignored("*.log\n!keep.log\n", "keep.log")
    assert _ignored("*.log\n!keep.log\n", "other.log")
    assert _ignored("!keep.log\n*.log\n", "keep.log")
    assert _ignored("outputs/\n!outputs/README.md\n", "outputs/README.md")
    assert not _ignored("cache/*.db\n!cache/keep.db\n", "cache/keep.db")


def test_double_star_and_character_class() -> None:
    assert _ignored("**/secret.txt\n", "a/b/secret.txt")
    assert _ignored("docs/**/draft.md\n", "docs/draft.md")
    assert _ignored("docs/**/draft.md\n", "docs/a/b/draft.md")
    assert not _ignored("docs/**/draft.md\n", "other/draft.md")
    assert _ignored("file.[ab]\n", "file.a")
    assert not _ignored("file.[ab]\n", "file.c")
    assert _ignored("file.[!a]\n", "file.b")
    assert not _ignored("file.[!a]\n", "file.a")


def test_escaped_hash_and_question_mark() -> None:
    assert _ignored("\\#hash\n", "#hash")
    assert not _ignored("\\#hash\n", "hash")
    assert _ignored("a?c\n", "abc")
    assert not _ignored("a?c\n", "ac")


def test_casefold_flag_matches_env_variants() -> None:
    rules = check_agentignore.parse_agentignore(".env*\n", flags=re.IGNORECASE)
    assert check_agentignore.is_ignored(".ENV", rules)
    assert check_agentignore.is_ignored(".Env.Local", rules)


def test_cli_blocks_matching_path_and_allows_source(tmp_path: Path, capsys) -> None:
    (tmp_path / ".agentignore").write_text("outputs/\n.env*\n", encoding="utf-8")
    blocked = check_agentignore.main(
        ["--root", str(tmp_path), "outputs/run/report.md"],
    )
    blocked_err = capsys.readouterr().err
    assert blocked == 1
    assert "outputs/run/report.md" in blocked_err

    allowed = check_agentignore.main(["--root", str(tmp_path), "router.py"])
    assert allowed == 0
    assert capsys.readouterr().err == ""


def test_cli_fails_closed_when_agentignore_is_missing(tmp_path: Path, capsys) -> None:
    code = check_agentignore.main(["--root", str(tmp_path), "router.py"])
    assert code == 1
    assert "missing" in capsys.readouterr().err


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    command = [
        git,
        "-C",
        str(repo),
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=Agent Ignore Test",
        "-c",
        "commit.gpgsign=false",
        *args,
    ]
    return subprocess.run(  # noqa: S603 - git path comes from shutil.which
        command,
        check=check,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".githooks").mkdir()
    shutil.copy2(
        _ROOT / "scripts" / "check_agentignore.py",
        repo / "scripts" / "check_agentignore.py",
    )
    shutil.copy2(_ROOT / ".githooks" / "pre-commit", repo / ".githooks" / "pre-commit")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "core.hooksPath", ".githooks")
    (repo / ".agentignore").write_text(".env*\noutputs/\n", encoding="utf-8")
    (repo / "ok.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "credentials").write_text("token\n", encoding="utf-8")
    _git(
        repo,
        "add",
        ".agentignore",
        "ok.py",
        "credentials",
        "scripts/check_agentignore.py",
        ".githooks/pre-commit",
    )
    _git(repo, "commit", "-m", "init")
    return repo


def test_pre_commit_blocks_ignored_path_and_allows_source(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    _git(repo, "add", "-f", ".env")
    blocked = _git(repo, "commit", "-m", "secret", check=False)
    assert blocked.returncode != 0
    assert ".env" in blocked.stderr
    _git(repo, "restore", "--staged", ".env")

    (repo / "notes.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "notes.md")
    allowed = _git(repo, "commit", "-m", "notes")
    assert allowed.returncode == 0


def test_pre_commit_blocks_when_pattern_is_removed_in_the_same_commit(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / ".agentignore").write_text("# weakened\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    _git(repo, "add", "-f", ".agentignore", ".env")
    blocked = _git(repo, "commit", "-m", "weaken", check=False)
    assert blocked.returncode != 0
    assert ".env" in blocked.stderr


def test_pre_commit_blocks_delete_when_pattern_is_added_in_the_same_commit(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    current = (repo / ".agentignore").read_text(encoding="utf-8")
    (repo / ".agentignore").write_text(current + "*credential*\n", encoding="utf-8")
    _git(repo, "add", ".agentignore")
    _git(repo, "rm", "credentials")
    blocked = _git(repo, "commit", "-m", "drop credentials", check=False)
    assert blocked.returncode != 0
    assert "credentials" in blocked.stderr


def test_pre_commit_blocks_rename_into_an_ignored_name(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "mv", "ok.py", ".env.backup")
    blocked = _git(repo, "commit", "-m", "rename", check=False)
    assert blocked.returncode != 0
    assert ".env.backup" in blocked.stderr
