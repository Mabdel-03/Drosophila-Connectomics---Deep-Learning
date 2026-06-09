"""Load and validate the Data Preparation config (configs/data_v783.yaml).

Parses the YAML into a frozen dataclass so the rest of the package consumes typed,
immutable config rather than raw dicts. Expands ${FLYCONN_DATA_ROOT} style env
references in ``data_root``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import DEFAULT_CONFIG, DataPaths


@dataclass(frozen=True)
class FileSpec:
    """One downloadable file (Zenodo or GitHub)."""

    key: str               # filename on disk (and Zenodo key)
    url: str               # fully-resolved download URL
    role: str              # edges | node_set | synapses | annotations | aux
    size: int | None = None
    md5: str | None = None
    source: str = "zenodo"  # zenodo | github


@dataclass(frozen=True)
class Config:
    version: str
    expected_neurons: int
    expected_connections: int
    data_root: str
    adjacency_orientation: str
    node_set: str
    default_nt_policy: str
    synapse_threshold: int
    nt_policies: dict[str, dict[str, int]]
    zenodo_files: list[FileSpec]
    github_files: list[FileSpec]
    citation: dict[str, str]
    raw: dict[str, Any] = field(default_factory=dict)  # original parsed yaml

    # --- convenience ---
    @property
    def all_files(self) -> list[FileSpec]:
        return [*self.zenodo_files, *self.github_files]

    def file(self, key: str) -> FileSpec:
        for f in self.all_files:
            if f.key == key:
                return f
        raise KeyError(f"No file with key {key!r} in config")

    def paths(self) -> DataPaths:
        # data_root in yaml typically embeds ${FLYCONN_DATA_ROOT}/v<version>. Expand
        # env vars; if FLYCONN_DATA_ROOT is unset (so the literal token survives), fall
        # back to the package default base + /v<version>. The resolved string is the
        # FULL versioned dir, used as-is (DataPaths does not re-append the version).
        expanded = os.path.expandvars(self.data_root)
        if "${" in expanded or "$FLYCONN_DATA_ROOT" in expanded:
            from .paths import data_root as default_data_root
            root = default_data_root() / f"v{self.version}"
        else:
            root = Path(expanded)
        return DataPaths.for_version(self.version, root=root)


def _resolve_zenodo(z: dict[str, Any]) -> list[FileSpec]:
    tmpl = z["url_template"]
    out: list[FileSpec] = []
    for key, meta in z["files"].items():
        out.append(
            FileSpec(
                key=key,
                url=tmpl.format(key=key),
                role=meta.get("role", "aux"),
                size=meta.get("size"),
                md5=meta.get("md5"),
                source="zenodo",
            )
        )
    return out


def _resolve_github(g: dict[str, Any]) -> list[FileSpec]:
    tmpl = g["url_template"]
    out: list[FileSpec] = []
    for name, meta in g["files"].items():
        out.append(
            FileSpec(
                key=name,
                url=tmpl.format(name=name),
                role=(meta or {}).get("role", "aux"),
                size=(meta or {}).get("size"),
                md5=(meta or {}).get("md5"),
                source="github",
            )
        )
    return out


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)

    expected = raw.get("expected", {})
    cfg = Config(
        version=str(raw["data_version"]),
        expected_neurons=int(expected["neurons"]),
        expected_connections=int(expected["connections"]),
        data_root=raw["data_root"],
        adjacency_orientation=raw.get("adjacency_orientation", "source_major"),
        node_set=raw.get("node_set", "proofread_root_ids"),
        default_nt_policy=raw["default_nt_policy"],
        synapse_threshold=int(raw.get("synapse_threshold", 5)),
        nt_policies=raw["nt_policies"],
        zenodo_files=_resolve_zenodo(raw["zenodo"]),
        github_files=_resolve_github(raw["github"]),
        citation=raw.get("citation", {}),
        raw=raw,
    )
    return cfg
