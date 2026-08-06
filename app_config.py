"""
Application configuration store for UI-managed secrets and integrations.

Persists Slack webhooks and LLM API keys to a local JSON file under ``data/``
(gitignored). API keys are applied to ``os.environ`` so existing providers in
``llm_integrations`` pick them up without code changes.

This is intended for local/single-user deployments. Keys are stored in plain
text on disk; do not use this module as a multi-tenant secrets manager.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "data" / "app_config.json"

# Known credential env vars that power LLM providers (see check_llms.py).
KNOWN_API_KEY_VARS: list[dict[str, str]] = [
    {
        "env_var": "GROK_API_KEY",
        "label": "xAI Grok",
        "description": "Powers grok-4.5, grok-4.3",
        "placeholder": "xai-...",
    },
    {
        "env_var": "ANTHROPIC_API_KEY",
        "label": "Anthropic Claude",
        "description": "Powers claude-sonnet / claude-opus models",
        "placeholder": "sk-ant-...",
    },
    {
        "env_var": "OPENAI_API_KEY",
        "label": "OpenAI",
        "description": "Powers gpt-4o and related models",
        "placeholder": "sk-...",
    },
    {
        "env_var": "DEEPSEEK_API_KEY",
        "label": "DeepSeek",
        "description": "Powers deepseek-v4-flash / deepseek-v4-pro",
        "placeholder": "sk-...",
    },
    {
        "env_var": "VERTEX_PROJECT",
        "label": "Google Vertex Project ID",
        "description": "GCP project for Gemini / Claude-on-Vertex (not a secret key)",
        "placeholder": "my-gcp-project",
    },
]

_KNOWN_ENV_VAR_SET = {entry["env_var"] for entry in KNOWN_API_KEY_VARS}

_lock = threading.RLock()
_config_path: Path = _DEFAULT_CONFIG_PATH
# Env vars that were applied from the config file (so we can clear them on delete).
_managed_env_vars: set[str] = set()


def _empty_config() -> dict[str, Any]:
    return {
        "slack_webhooks": [],
        "api_keys": {},
    }


def set_config_path(path: Path | str) -> None:
    """Override the config file path (primarily for tests)."""
    global _config_path
    with _lock:
        _config_path = Path(path)


def get_config_path() -> Path:
    return _config_path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_raw() -> dict[str, Any]:
    path = _config_path
    if not path.exists():
        return _empty_config()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read app config at %s: %s — using empty config", path, e)
        return _empty_config()

    if not isinstance(data, dict):
        return _empty_config()

    config = _empty_config()
    webhooks = data.get("slack_webhooks")
    if isinstance(webhooks, list):
        config["slack_webhooks"] = [
            wh
            for wh in webhooks
            if isinstance(wh, dict) and wh.get("id") and wh.get("url")
        ]
    keys = data.get("api_keys")
    if isinstance(keys, dict):
        config["api_keys"] = {
            str(k): str(v)
            for k, v in keys.items()
            if k and isinstance(v, str) and v.strip()
        }
    return config


def _write_raw(config: dict[str, Any]) -> None:
    path = _config_path
    _ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "slack_webhooks": config.get("slack_webhooks") or [],
        "api_keys": config.get("api_keys") or {},
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def mask_secret(value: Optional[str], *, visible: int = 4) -> str:
    """Return a masked preview of a secret (never the full value)."""
    if not value:
        return ""
    value = str(value)
    if len(value) <= visible * 2:
        return "•" * min(len(value), 8)
    return f"{value[:visible]}…{value[-visible:]}"


def apply_api_keys_to_environ(config: Optional[dict[str, Any]] = None) -> None:
    """Push stored API keys into ``os.environ`` so LLM providers can use them."""
    global _managed_env_vars
    with _lock:
        data = config if config is not None else _read_raw()
        keys = data.get("api_keys") or {}
        applied: set[str] = set()
        for env_var, value in keys.items():
            if not env_var or not isinstance(value, str) or not value.strip():
                continue
            os.environ[str(env_var)] = value.strip()
            applied.add(str(env_var))
        # Drop managed keys that were removed from config
        for env_var in list(_managed_env_vars - applied):
            os.environ.pop(env_var, None)
        _managed_env_vars = applied


def load_and_apply() -> dict[str, Any]:
    """Load config from disk and apply API keys to the process environment."""
    with _lock:
        config = _read_raw()
        apply_api_keys_to_environ(config)
        logger.info(
            "Loaded app config: %d slack webhook(s), %d api key(s) from %s",
            len(config.get("slack_webhooks") or []),
            len(config.get("api_keys") or {}),
            _config_path,
        )
        return deepcopy(config)


# ── Slack webhooks ──────────────────────────────────────────────────────────


def list_slack_webhooks() -> List[dict[str, Any]]:
    with _lock:
        return deepcopy(_read_raw().get("slack_webhooks") or [])


def get_slack_webhook(webhook_id: str) -> Optional[dict[str, Any]]:
    with _lock:
        for wh in _read_raw().get("slack_webhooks") or []:
            if wh.get("id") == webhook_id:
                return deepcopy(wh)
    return None


def add_slack_webhook(name: str, url: str) -> dict[str, Any]:
    name = (name or "").strip() or "Slack Webhook"
    url = (url or "").strip()
    if not url.startswith("https://"):
        raise ValueError("Webhook URL must start with https://")

    entry = {
        "id": str(uuid.uuid4()),
        "name": name,
        "url": url,
    }
    with _lock:
        config = _read_raw()
        webhooks = list(config.get("slack_webhooks") or [])
        webhooks.append(entry)
        config["slack_webhooks"] = webhooks
        _write_raw(config)
    return deepcopy(entry)


def update_slack_webhook(
    webhook_id: str,
    *,
    name: Optional[str] = None,
    url: Optional[str] = None,
) -> dict[str, Any]:
    with _lock:
        config = _read_raw()
        webhooks = list(config.get("slack_webhooks") or [])
        for i, wh in enumerate(webhooks):
            if wh.get("id") != webhook_id:
                continue
            updated = dict(wh)
            if name is not None:
                updated["name"] = (name or "").strip() or updated.get("name") or "Slack Webhook"
            if url is not None:
                url = url.strip()
                if not url.startswith("https://"):
                    raise ValueError("Webhook URL must start with https://")
                updated["url"] = url
            webhooks[i] = updated
            config["slack_webhooks"] = webhooks
            _write_raw(config)
            return deepcopy(updated)
    raise KeyError(f"Webhook not found: {webhook_id}")


def delete_slack_webhook(webhook_id: str) -> bool:
    with _lock:
        config = _read_raw()
        webhooks = list(config.get("slack_webhooks") or [])
        new_list = [wh for wh in webhooks if wh.get("id") != webhook_id]
        if len(new_list) == len(webhooks):
            return False
        config["slack_webhooks"] = new_list
        _write_raw(config)
        return True


# ── API keys ────────────────────────────────────────────────────────────────


def get_api_keys_raw() -> Dict[str, str]:
    with _lock:
        return dict(_read_raw().get("api_keys") or {})


def set_api_key(env_var: str, value: str) -> dict[str, Any]:
    env_var = (env_var or "").strip()
    if env_var not in _KNOWN_ENV_VAR_SET:
        raise ValueError(
            f"Unknown API key variable '{env_var}'. "
            f"Supported: {', '.join(sorted(_KNOWN_ENV_VAR_SET))}"
        )
    value = (value or "").strip()
    if not value:
        raise ValueError("API key value cannot be empty")

    with _lock:
        config = _read_raw()
        keys = dict(config.get("api_keys") or {})
        keys[env_var] = value
        config["api_keys"] = keys
        _write_raw(config)
        apply_api_keys_to_environ(config)

    return _api_key_status_entry(env_var)


def delete_api_key(env_var: str) -> bool:
    env_var = (env_var or "").strip()
    with _lock:
        config = _read_raw()
        keys = dict(config.get("api_keys") or {})
        if env_var not in keys:
            return False
        del keys[env_var]
        config["api_keys"] = keys
        _write_raw(config)
        apply_api_keys_to_environ(config)
        return True


def _api_key_status_entry(env_var: str) -> dict[str, Any]:
    meta = next((m for m in KNOWN_API_KEY_VARS if m["env_var"] == env_var), None)
    stored = get_api_keys_raw().get(env_var)
    env_value = os.getenv(env_var)
    configured = bool(env_value)
    source = None
    if stored:
        source = "config"
    elif env_value:
        source = "environment"
    return {
        "env_var": env_var,
        "label": meta["label"] if meta else env_var,
        "description": meta["description"] if meta else "",
        "placeholder": meta["placeholder"] if meta else "",
        "configured": configured,
        "source": source,
        "preview": mask_secret(env_value or stored),
    }


def list_api_key_status() -> List[dict[str, Any]]:
    return [_api_key_status_entry(entry["env_var"]) for entry in KNOWN_API_KEY_VARS]


def get_public_config() -> dict[str, Any]:
    """Return a frontend-safe snapshot (secrets masked, never full keys)."""
    return {
        "slack_webhooks": list_slack_webhooks(),
        "api_keys": list_api_key_status(),
        "known_api_key_vars": KNOWN_API_KEY_VARS,
    }
