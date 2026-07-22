#!/usr/bin/env python3
"""Publish one content-bound canonical run into the static web release.

The simulator writes its immutable run directory and browser projection outside
the web tree.  This command verifies their mutual receipts, copies the run under
its content-derived ID, and updates the public episode index last.  An
interrupted publish can therefore leave only unreferenced files, never a public
manifest that points at a partial run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, TypeVar

from fly_sensor2behavior.canonical_artifacts import (
    CANONICAL_ARTIFACT_SCHEMA_VERSION,
    CANONICAL_WEB_PROJECTION_CANONICALIZATION,
    canonical_web_projection_sha256,
)


ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")
UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)
WEB_REPLAY_PROJECTION_SCHEMA_VERSION = "1.0.0"
LEGACY_WEB_REPLAY_PROJECTION_CANONICALIZATION = (
    "json-sort-keys-compact-utf8-v1"
)
WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS = (
    "source_artifact_manifest_sha256",
    "source_artifact_schema_version",
)


class PublicationError(RuntimeError):
    """Raised when a candidate run cannot be published without ambiguity."""


_ReturnT = TypeVar("_ReturnT")


def _serialize_public_manifest_update(
    function: Callable[..., _ReturnT],
) -> Callable[..., _ReturnT]:
    """Serialize publication for one resolved web root across processes.

    The immutable run and replay are installed before the public index is
    replaced, but the index is still a read-modify-write object.  A lock held
    from before the first read through the final verified write prevents two
    otherwise valid publishers from silently dropping one another's episode.
    The lock lives beside ``public`` so it is not copied into the static site.
    """

    @wraps(function)
    def locked(*args: Any, **kwargs: Any) -> _ReturnT:
        if "web_public_root" not in kwargs:
            raise TypeError("web_public_root must be supplied by keyword")
        public_root = Path(kwargs["web_public_root"]).resolve()
        lock_path = public_root.parent / (
            ".{}.fly-s2b-publish.lock".format(public_root.name)
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    return locked


def _reject_constant(value: str) -> None:
    raise PublicationError("non-finite JSON constant {!r} is forbidden".format(value))


def _reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError("duplicate JSON key {!r} is forbidden".format(key))
        result[key] = value
    return result


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError("{} is not strict JSON: {}".format(label, exc)) from exc
    if not isinstance(value, Mapping):
        raise PublicationError("{} must contain a JSON object".format(label))
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise PublicationError(
            "{} must contain lowercase letters, digits, dots, underscores, or hyphens".format(
                label
            )
        )
    return value


def _tree_receipt(root: Path) -> Mapping[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise PublicationError("artifact source must be a real directory")
    receipt: Dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PublicationError("artifact trees may not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublicationError("artifact trees may contain only regular files")
        relative = path.relative_to(root).as_posix()
        receipt[relative] = _sha256_file(path)
    if not receipt:
        raise PublicationError("artifact tree is empty")
    return receipt


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_projection_bytes(replay: Mapping[str, Any]) -> bytes:
    """Return the exact preimage hashed by both artifact writer generations.

    Python and ECMAScript do not serialize all binary64 values identically
    (``0.0`` and exponent formatting are common differences).  Publishing this
    compact preimage gives the browser bytes whose SHA-256 is already owned by
    the immutable artifact manifest.  The browser can then compare its parsed
    JSON tree with the served replay after removing only the two registered
    circular fields, without attempting an unsafe cross-runtime re-encoding.
    """

    projected = {
        key: value
        for key, value in replay.items()
        if key not in WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    }
    try:
        return json.dumps(
            projected,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PublicationError(
            "canonical web projection contains a non-finite or non-JSON value"
        ) from exc


def _projection_preimage(
    manifest: Mapping[str, Any], replay: Mapping[str, Any]
) -> Tuple[bytes, str]:
    receipt = manifest.get("web_replay_projection")
    if not isinstance(receipt, Mapping):
        raise PublicationError("artifact has no browser projection receipt")
    if receipt.get("schema_version") != WEB_REPLAY_PROJECTION_SCHEMA_VERSION:
        raise PublicationError("artifact browser projection schema is unsupported")
    expected_canonicalization = (
        CANONICAL_WEB_PROJECTION_CANONICALIZATION
        if manifest.get("schema_version") == CANONICAL_ARTIFACT_SCHEMA_VERSION
        else LEGACY_WEB_REPLAY_PROJECTION_CANONICALIZATION
    )
    if receipt.get("canonicalization") != expected_canonicalization:
        raise PublicationError(
            "artifact browser projection canonicalization is unsupported"
        )
    if receipt.get("excluded_top_level_fields") != list(
        WEB_REPLAY_PROJECTION_EXCLUDED_FIELDS
    ):
        raise PublicationError("artifact browser projection exclusions are invalid")
    projection_sha256 = receipt.get("sha256")
    if (
        not isinstance(projection_sha256, str)
        or SHA256_PATTERN.fullmatch(projection_sha256) is None
    ):
        raise PublicationError("artifact browser projection SHA-256 is invalid")
    preimage = _canonical_projection_bytes(replay)
    if hashlib.sha256(preimage).hexdigest() != projection_sha256:
        raise PublicationError("replay does not match the artifact projection receipt")
    return preimage, projection_sha256


def _safe_public_path(root: Path, relative_url: Any, label: str) -> Path:
    if (
        not isinstance(relative_url, str)
        or not relative_url
        or "?" in relative_url
        or "#" in relative_url
        or "\\" in relative_url
    ):
        raise PublicationError("{} must be a plain relative URL".format(label))
    relative = Path(relative_url)
    if relative.is_absolute():
        raise PublicationError("{} must remain under the public root".format(label))
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PublicationError(
            "{} escapes the public root".format(label)
        ) from exc
    return resolved


def _bind_existing_summary(
    summary_value: Any, web_public_root: Path
) -> Tuple[Mapping[str, Any], Path, bytes]:
    """Upgrade/verify one exporter summary with byte and projection receipts."""

    if not isinstance(summary_value, Mapping):
        raise PublicationError("public replay manifest episode must be an object")
    summary = dict(summary_value)
    episode_id = _require_id(summary.get("id"), "public episode id")
    replay_path = _safe_public_path(
        web_public_root, summary.get("data_url"), "public episode data_url"
    )
    artifact_path = _safe_public_path(
        web_public_root,
        summary.get("artifact_manifest_url"),
        "public episode artifact_manifest_url",
    )
    if not replay_path.is_file() or replay_path.is_symlink():
        raise PublicationError("published episode replay must be a regular file")
    if not artifact_path.is_file() or artifact_path.is_symlink():
        raise PublicationError("published artifact manifest must be a regular file")
    replay = _load_json(replay_path, "published episode replay")
    artifact = _load_json(artifact_path, "published artifact manifest")
    replay_sha256 = _sha256_file(replay_path)
    artifact_sha256 = _sha256_file(artifact_path)
    if replay.get("id") != episode_id:
        raise PublicationError("published replay id does not match its summary")
    if replay.get("source_run_id") != artifact.get("run_id"):
        raise PublicationError("published replay run id does not match its artifact")
    if replay.get("source_artifact_manifest_sha256") != artifact_sha256:
        raise PublicationError("published replay does not bind its artifact bytes")
    if replay.get("source_artifact_schema_version") != artifact.get("schema_version"):
        raise PublicationError("published replay artifact schema is inconsistent")
    projection_bytes, _ = _projection_preimage(artifact, replay)

    projection_url = "data/episodes/{}.artifact-projection.json".format(episode_id)
    for field, expected in (
        ("replay_sha256", replay_sha256),
        ("artifact_manifest_sha256", artifact_sha256),
        ("artifact_projection_url", projection_url),
    ):
        declared = summary.get(field)
        if declared is not None and declared != expected:
            raise PublicationError(
                "published episode {} disagrees with its exact bytes".format(field)
            )
        summary[field] = expected
    projection_path = _safe_public_path(
        web_public_root, projection_url, "public episode artifact_projection_url"
    )
    return summary, projection_path, projection_bytes


def _validate_candidate(
    artifact_dir: Path,
    replay_path: Path,
    episode_id: str,
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    Mapping[str, str],
    bytes,
]:
    manifest_path = artifact_dir / "manifest.json"
    digest_path = artifact_dir / "manifest.sha256"
    if not replay_path.is_file() or replay_path.is_symlink():
        raise PublicationError("canonical web replay must be a regular file")
    manifest = _load_json(manifest_path, "canonical artifact manifest")
    replay = _load_json(replay_path, "canonical web replay")
    run_id = _require_id(manifest.get("run_id"), "artifact run_id")
    if manifest.get("schema_version") != CANONICAL_ARTIFACT_SCHEMA_VERSION:
        raise PublicationError("canonical artifact schema is unsupported")
    manifest_sha256 = _sha256_file(manifest_path)
    if not digest_path.is_file() or digest_path.is_symlink():
        raise PublicationError("canonical artifact requires manifest.sha256")
    expected_line = "{}  manifest.json\n".format(manifest_sha256)
    if digest_path.read_text(encoding="ascii", errors="strict") != expected_line:
        raise PublicationError("manifest.sha256 does not bind manifest.json")
    if replay.get("source_run_id") != run_id:
        raise PublicationError("replay source_run_id does not match the artifact")
    if replay.get("source_artifact_manifest_sha256") != manifest_sha256:
        raise PublicationError("replay does not bind the exact artifact manifest bytes")
    if replay.get("source_artifact_schema_version") != manifest.get("schema_version"):
        raise PublicationError("replay artifact schema receipt is inconsistent")
    if replay.get("id") != episode_id:
        raise PublicationError("replay id does not match --episode-id")
    _require_id(replay.get("id"), "replay id")
    projection_bytes, projection_sha256 = _projection_preimage(manifest, replay)
    if canonical_web_projection_sha256(replay) != projection_sha256:
        raise PublicationError("replay does not match the artifact projection receipt")
    label = replay.get("label")
    status = replay.get("status")
    if not isinstance(label, str) or not label.strip():
        raise PublicationError("replay label must be non-empty")
    if status not in ("exploratory", "calibrated", "validated"):
        raise PublicationError("replay status is unsupported")
    tree = _tree_receipt(artifact_dir)
    return manifest, replay, manifest_sha256, tree, projection_bytes


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=".{}-".format(path.name), suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _install_tree(source: Path, target: Path, source_receipt: Mapping[str, str]) -> None:
    if target.exists():
        if _tree_receipt(target) != source_receipt:
            raise PublicationError(
                "content-derived run directory already exists with different bytes"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=str(target.parent), prefix=".{}-".format(target.name))
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True, symlinks=False)
        if _tree_receipt(staging) != source_receipt:
            raise PublicationError("copied artifact failed byte-for-byte verification")
        os.replace(str(staging), str(target))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@_serialize_public_manifest_update
def publish_canonical_run(
    *,
    artifact_dir: Path,
    replay_path: Path,
    web_public_root: Path,
    episode_id: str,
    condition: str,
    description: str,
    color: str,
    replace_episode: bool = False,
    position: Optional[int] = None,
    created_at: Optional[str] = None,
) -> Mapping[str, Any]:
    """Validate and publish one run, making the public index visible last."""

    episode_id = _require_id(episode_id, "episode_id")
    for value, label in ((condition, "condition"), (description, "description")):
        if not isinstance(value, str) or not value.strip():
            raise PublicationError("{} must be non-empty".format(label))
    if not isinstance(color, str) or COLOR_PATTERN.fullmatch(color) is None:
        raise PublicationError("color must be a six-digit CSS hex value")
    if (
        position is not None
        and (isinstance(position, bool) or not isinstance(position, int) or position < 0)
    ):
        raise PublicationError("position must be a non-negative integer")
    if created_at is not None:
        if (
            not isinstance(created_at, str)
            or UTC_TIMESTAMP_PATTERN.fullmatch(created_at) is None
        ):
            raise PublicationError(
                "created_at must use exact UTC YYYY-MM-DDTHH:MM:SSZ form"
            )
        try:
            datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise PublicationError("created_at is not a valid UTC timestamp") from exc
    artifact_dir = Path(artifact_dir).resolve()
    replay_path = Path(replay_path).resolve()
    web_public_root = Path(web_public_root).resolve()
    data_root = web_public_root / "data"
    public_manifest_path = data_root / "manifest.json"
    public_manifest = dict(_load_json(public_manifest_path, "public replay manifest"))
    (
        manifest,
        replay,
        manifest_sha256,
        source_receipt,
        projection_bytes,
    ) = _validate_candidate(
        artifact_dir, replay_path, episode_id
    )
    run_id = str(manifest["run_id"])
    target_artifact = data_root / "runs" / run_id
    target_replay = data_root / "episodes" / (episode_id + ".json")
    replay_bytes = replay_path.read_bytes()
    replay_sha256 = hashlib.sha256(replay_bytes).hexdigest()
    projection_url = "data/episodes/{}.artifact-projection.json".format(episode_id)
    target_projection = _safe_public_path(
        web_public_root, projection_url, "candidate artifact_projection_url"
    )

    episode_values = public_manifest.get("episodes")
    if not isinstance(episode_values, list):
        raise PublicationError("public replay manifest episodes must be an array")
    episodes = []
    projection_writes: Dict[Path, bytes] = {}
    for existing_summary in episode_values:
        bound_summary, sidecar_path, sidecar_bytes = _bind_existing_summary(
            existing_summary, web_public_root
        )
        episodes.append(bound_summary)
        previous = projection_writes.get(sidecar_path)
        if previous is not None and previous != sidecar_bytes:
            raise PublicationError(
                "public episodes ambiguously share an artifact projection path"
            )
        projection_writes[sidecar_path] = sidecar_bytes
    matching_indices = [
        index
        for index, item in enumerate(episodes)
        if isinstance(item, Mapping) and item.get("id") == episode_id
    ]
    if len(matching_indices) > 1:
        raise PublicationError("public replay manifest contains duplicate episode IDs")
    summary = {
        "id": episode_id,
        "label": replay["label"],
        "condition": condition,
        "description": description,
        "color": color.lower(),
        "status": replay["status"],
        "data_url": "data/episodes/{}.json".format(episode_id),
        "replay_sha256": replay_sha256,
        "artifact_manifest_url": "data/runs/{}/manifest.json".format(run_id),
        "artifact_manifest_sha256": manifest_sha256,
        "artifact_projection_url": projection_url,
    }
    if matching_indices:
        index = matching_indices[0]
        existing = episodes[index]
        existing_bytes_equal = target_replay.is_file() and target_replay.read_bytes() == replay_bytes
        if not replace_episode and (existing != summary or not existing_bytes_equal):
            raise PublicationError(
                "episode ID is already published; pass --replace-episode for an intentional replacement"
            )
        episodes[index] = summary
        if position is not None and position != index:
            item = episodes.pop(index)
            episodes.insert(min(position, len(episodes)), item)
    else:
        if target_replay.exists() and target_replay.read_bytes() != replay_bytes:
            raise PublicationError("episode replay path already exists with different bytes")
        if position is None:
            episodes.append(summary)
        else:
            episodes.insert(min(position, len(episodes)), summary)
    public_manifest["episodes"] = episodes
    public_manifest["created_at"] = created_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    public_manifest_bytes = _serialized_json(public_manifest)
    projection_writes[target_projection] = projection_bytes

    # All validation and serialization happens before mutating the web tree.
    _install_tree(artifact_dir, target_artifact, source_receipt)
    if not target_replay.exists() or target_replay.read_bytes() != replay_bytes:
        if target_replay.exists() and not replace_episode:
            raise PublicationError("refusing to replace an existing episode replay")
        _atomic_write(target_replay, replay_bytes)
    for projection_path, payload in sorted(
        projection_writes.items(), key=lambda item: item[0].as_posix()
    ):
        if not projection_path.exists() or projection_path.read_bytes() != payload:
            _atomic_write(projection_path, payload)
    _atomic_write(public_manifest_path, public_manifest_bytes)

    if _sha256_file(target_artifact / "manifest.json") != manifest_sha256:
        raise PublicationError("published artifact manifest changed after installation")
    if target_replay.read_bytes() != replay_bytes:
        raise PublicationError("published replay changed after installation")
    if (
        _sha256_file(target_projection)
        != manifest["web_replay_projection"]["sha256"]
    ):
        raise PublicationError("published artifact projection preimage changed")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--web-public-root", type=Path, default=Path("web/public"))
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--color", required=True)
    parser.add_argument("--replace-episode", action="store_true")
    parser.add_argument(
        "--position",
        type=int,
        help="optional zero-based position in the public episode selector",
    )
    parser.add_argument("--created-at")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = publish_canonical_run(
        artifact_dir=args.artifact_dir,
        replay_path=args.replay,
        web_public_root=args.web_public_root,
        episode_id=args.episode_id,
        condition=args.condition,
        description=args.description,
        color=args.color,
        replace_episode=args.replace_episode,
        position=args.position,
        created_at=args.created_at,
    )
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
