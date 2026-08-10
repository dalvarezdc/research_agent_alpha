"""Unit tests for app_config (Slack webhooks + LLM API key store)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import app_config


@pytest.fixture()
def isolated_config(tmp_path, monkeypatch):
    """Point app_config at a temp file and reset managed env state."""
    path = tmp_path / "app_config.json"
    app_config.set_config_path(path)
    # Clear managed keys and known env vars so tests are hermetic
    for entry in app_config.KNOWN_API_KEY_VARS:
        monkeypatch.delenv(entry["env_var"], raising=False)
    app_config._managed_env_vars = set()  # noqa: SLF001 — test isolation
    yield path
    app_config.set_config_path(app_config._DEFAULT_CONFIG_PATH)  # noqa: SLF001
    app_config._managed_env_vars = set()  # noqa: SLF001


def test_empty_config_on_missing_file(isolated_config):
    cfg = app_config.load_and_apply()
    assert cfg["slack_webhooks"] == []
    assert cfg["api_keys"] == {}
    assert app_config.list_slack_webhooks() == []


def test_add_list_delete_slack_webhook(isolated_config):
    entry = app_config.add_slack_webhook("Clinical", "https://hooks.slack.com/services/T/B/X")
    assert entry["id"]
    assert entry["name"] == "Clinical"
    assert entry["url"].startswith("https://")

    listed = app_config.list_slack_webhooks()
    assert len(listed) == 1
    assert listed[0]["id"] == entry["id"]

    got = app_config.get_slack_webhook(entry["id"])
    assert got is not None
    assert got["name"] == "Clinical"

    updated = app_config.update_slack_webhook(entry["id"], name="Ops")
    assert updated["name"] == "Ops"

    assert app_config.delete_slack_webhook(entry["id"]) is True
    assert app_config.list_slack_webhooks() == []
    assert app_config.delete_slack_webhook(entry["id"]) is False


def test_reject_non_https_webhook(isolated_config):
    with pytest.raises(ValueError, match="https://"):
        app_config.add_slack_webhook("Bad", "http://example.com/hook")


def test_set_and_apply_api_key(isolated_config, monkeypatch):
    status = app_config.set_api_key("GROK_API_KEY", "xai-test-secret-key-12345")
    assert status["configured"] is True
    assert status["source"] == "config"
    assert "xai-" in status["preview"] or "…" in status["preview"]
    assert os.getenv("GROK_API_KEY") == "xai-test-secret-key-12345"

    # Public config never leaks full secret
    public = app_config.get_public_config()
    blob = str(public)
    assert "xai-test-secret-key-12345" not in blob

    assert app_config.delete_api_key("GROK_API_KEY") is True
    assert os.getenv("GROK_API_KEY") is None


def test_unknown_api_key_rejected(isolated_config):
    with pytest.raises(ValueError, match="Unknown"):
        app_config.set_api_key("NOT_A_REAL_KEY", "value")


def test_empty_api_key_rejected(isolated_config):
    with pytest.raises(ValueError, match="empty"):
        app_config.set_api_key("OPENAI_API_KEY", "   ")


def test_env_only_key_shows_environment_source(isolated_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-only")
    statuses = app_config.list_api_key_status()
    anthropic = next(s for s in statuses if s["env_var"] == "ANTHROPIC_API_KEY")
    assert anthropic["configured"] is True
    assert anthropic["source"] == "environment"
    # delete should not remove pure-env keys from the store (nothing stored)
    assert app_config.delete_api_key("ANTHROPIC_API_KEY") is False


def test_mask_secret():
    assert app_config.mask_secret("") == ""
    assert "…" in app_config.mask_secret("abcdefghijklmnop")
    assert app_config.mask_secret("short")  # still returns something masked


def test_persist_roundtrip(isolated_config):
    app_config.add_slack_webhook("A", "https://hooks.slack.com/services/A")
    app_config.set_api_key("DEEPSEEK_API_KEY", "sk-deepseek-roundtrip")
    # Re-read from disk without in-memory cache (module always reads file)
    wh = app_config.list_slack_webhooks()
    assert len(wh) == 1
    assert Path(isolated_config).exists()
    raw = isolated_config.read_text(encoding="utf-8")
    assert "hooks.slack.com" in raw
    assert "sk-deepseek-roundtrip" in raw  # on-disk plaintext for local use
