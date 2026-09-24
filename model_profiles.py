"""Server-side model profiles for independent Claude CLI invocations."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_MODEL_CONFIG = Path(__file__).resolve().parent / "config" / "models.local.json"
_AUTH_FIELDS = ("api_key", "api_key_env", "auth_token", "auth_token_env")
_PROFILE_FIELDS = {"label", "provider", "model", "base_url", "env", *_AUTH_FIELDS}
# Empty values also override credentials/routing in Claude's settings.json.
_ISOLATED_ENV = {
    name: ""
    for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR", "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    )
}


@dataclass(frozen=True)
class ResolvedModelProfile:
    model: str
    env: dict[str, str] = field(repr=False)


@dataclass(frozen=True)
class ModelProfiles:
    profiles: dict[str, dict] = field(default_factory=dict, repr=False)
    defaults: dict[str, str] = field(default_factory=dict)

    def public_data(self) -> dict:
        # Explicit allowlist: credentials and extra env never reach the browser.
        return {
            "models": [
                {
                    "id": profile_id,
                    "label": profile.get("label") or profile_id,
                    "provider": profile.get("provider", ""),
                    "model": profile["model"],
                }
                for profile_id, profile in self.profiles.items()
            ],
            "defaults": dict(self.defaults),
        }

    def resolve(self, profile_id: str) -> ResolvedModelProfile:
        if profile_id not in self.profiles:
            raise ValueError(f"Unknown model profile: {profile_id}")
        profile = self.profiles[profile_id]
        auth_field = next(key for key in _AUTH_FIELDS if key in profile)
        credential = profile[auth_field]
        if auth_field.endswith("_env"):
            credential = os.environ.get(credential, "").strip()
            if not credential:
                raise ValueError(
                    f"Model profile '{profile_id}' requires environment variable "
                    f"{profile[auth_field]}"
                )
        env = {**_ISOLATED_ENV, **profile.get("env", {})}
        env.update({
            "ANTHROPIC_BASE_URL": profile["base_url"],
            "ANTHROPIC_MODEL": profile["model"],
            # Pin aliases and Claude's auxiliary calls to this provider's model.
            "ANTHROPIC_DEFAULT_OPUS_MODEL": profile["model"],
            "ANTHROPIC_DEFAULT_SONNET_MODEL": profile["model"],
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": profile["model"],
            "ANTHROPIC_SMALL_FAST_MODEL": profile["model"],
            "ANTHROPIC_API_KEY" if auth_field.startswith("api_key")
            else "ANTHROPIC_AUTH_TOKEN": credential,
        })
        return ResolvedModelProfile(model=profile["model"], env=env)


def load_model_profiles(path: str | Path | None = None) -> ModelProfiles:
    """Load once at startup; a missing default file keeps legacy CLI behavior."""
    selected = path or os.environ.get("SCIMAS_MODELS_CONFIG")
    source = Path(selected).expanduser() if selected else DEFAULT_MODEL_CONFIG
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if not selected:
            return ModelProfiles()
        raise ValueError(f"Model config file not found: {source}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in model config {source} at line {exc.lineno}, column {exc.colno}"
        ) from None
    except OSError:
        raise ValueError(f"Cannot read model config: {source}") from None

    if not isinstance(data, dict) or set(data) - {"models", "defaults"}:
        raise ValueError("Model config must contain only 'models' and optional 'defaults'.")
    profiles = data.get("models")
    if not isinstance(profiles, dict):
        raise ValueError("Model config 'models' must be an object keyed by profile ID.")
    for profile_id, profile in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile_id):
            raise ValueError("Model profile IDs must use letters, digits, '.', '_' or '-'.")
        prefix = f"Model profile '{profile_id}'"
        if not isinstance(profile, dict) or set(profile) - _PROFILE_FIELDS:
            raise ValueError(f"{prefix} has unsupported fields or is not an object.")
        for key, value in profile.items():
            if key != "env" and (
                not isinstance(value, str) or not value.strip() or "\0" in value
            ):
                raise ValueError(f"{prefix}: '{key}' must be a non-empty string.")
        if not profile.get("model") or not profile.get("base_url"):
            raise ValueError(f"{prefix} requires 'model' and 'base_url'.")
        try:
            url = urlsplit(profile["base_url"])
            valid_url = (
                url.scheme in {"http", "https"} and url.hostname
                and not url.username and not url.password and not url.query and not url.fragment
            )
            url.port
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError(f"{prefix}: base_url must be an HTTP(S) URL without credentials or query.")
        auth_fields = [key for key in _AUTH_FIELDS if key in profile]
        if len(auth_fields) != 1:
            raise ValueError(f"{prefix} requires exactly one of {', '.join(_AUTH_FIELDS)}.")
        for key in auth_fields:
            if key.endswith("_env") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", profile[key]):
                raise ValueError(f"{prefix}: '{key}' must name an environment variable.")
        env = profile.get("env", {})
        if not isinstance(env, dict) or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or not isinstance(value, str) or "\0" in value
            for key, value in env.items()
        ):
            raise ValueError(f"{prefix}: 'env' must map environment variable names to strings.")
        if set(env) & (set(_ISOLATED_ENV) - {"ANTHROPIC_CUSTOM_HEADERS"}):
            raise ValueError(f"{prefix}: use the profile authentication fields, not 'env', for routing/auth.")

    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - {"planner", "agent"}:
        raise ValueError("Model config 'defaults' accepts only 'planner' and 'agent'.")
    for scope, profile_id in defaults.items():
        if not isinstance(profile_id, str) or (profile_id and profile_id not in profiles):
            raise ValueError(f"Default '{scope}' must name an existing model profile.")
    return ModelProfiles(profiles=profiles, defaults=defaults)
