"""CAVE token resolution. Pure: reads the environment / files, never hits the network.

The same token authenticates both the FlyWire (FAFB) and male-CNS datastacks through
the global CAVE auth service. Resolution order (first hit wins):

  1. an explicit token passed by the caller,
  2. ``$CAVE_TOKEN`` or ``$CAVECLIENT_TOKEN`` in the environment,
  3. caveclient's own secret file ``~/.cloudvolume/secrets/cave-secret.json``
     (key ``token`` or ``<server>-token``),
  4. a repo-local ``<repo>/.cave_secret`` (gitignored), either raw token text or
     a JSON object ``{"token": "..."}``.

If none is found we raise with every checked location listed, so the fix is obvious.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..paths import REPO_ROOT

_ENV_VARS = ("CAVE_TOKEN", "CAVECLIENT_TOKEN")
_CLOUDVOLUME_SECRET = Path.home() / ".cloudvolume" / "secrets" / "cave-secret.json"
_REPO_SECRET = REPO_ROOT / ".cave_secret"


def _token_from_cloudvolume(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(obj, dict):
        if obj.get("token"):
            return str(obj["token"]).strip()
        # some installs key the token by server, e.g. "global.daf-apis.com-token"
        for k, v in obj.items():
            if k.endswith("token") and v:
                return str(v).strip()
    return None


def _token_from_repo_secret(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj.get("token"):
                return str(obj["token"]).strip()
        except json.JSONDecodeError:
            return None
    return text


def read_cave_token(explicit: str | None = None) -> str:
    """Return a CAVE token, raising a clear error if none can be found."""
    if explicit:
        return explicit.strip()

    for var in _ENV_VARS:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()

    tok = _token_from_cloudvolume(_CLOUDVOLUME_SECRET)
    if tok:
        return tok

    tok = _token_from_repo_secret(_REPO_SECRET)
    if tok:
        return tok

    raise RuntimeError(
        "No CAVE token found. Provide one of:\n"
        f"  - explicit token argument\n"
        f"  - env var {' or '.join(_ENV_VARS)}\n"
        f"  - {_CLOUDVOLUME_SECRET} (JSON with a 'token' key)\n"
        f"  - {_REPO_SECRET} (raw token text or JSON {{'token': ...}}; gitignored)\n"
        "Get a token from https://global.daf-apis.com/auth/ (FlyWire/CAVE login)."
    )


def have_cave_token() -> bool:
    """True if a token can be resolved without raising (for probes / optional paths)."""
    try:
        read_cave_token()
        return True
    except RuntimeError:
        return False
