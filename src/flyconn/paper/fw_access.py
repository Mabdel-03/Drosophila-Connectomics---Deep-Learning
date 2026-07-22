"""FlyWire data-access layer for the paper verification.

Two interchangeable sources implement the same small protocol so every derive
function is source-agnostic:

  * ``LiveCaveFlyWire`` — queries the live CAVE materialization service
    (``flywire_fafb_public``, version 783, ``synapses_nt_v1``, no cleft threshold).
    This is the **primary** track: it reproduces the paper's published synapse counts
    exactly (validated: VCH 27,576 in / 32,363 out, 1,022 T4/T5 inputs / 12,301 syn,
    912 reciprocal — all to the digit).
  * ``OfflineFlyWire`` — the frozen v783 dump on scratch (raw synapse feather +
    proofread ``edges_full.parquet``). A stricter/older snapshot (~65% of the live
    counts); used as a cross-check and when no token is present.

``make_source()`` picks live when a CAVE token + ``caveclient`` are available, else
offline, so the same verification code runs in both regimes. Live results are cached
to parquet under the motif/paper scratch dir, so a credentialed run leaves a permanent,
re-auditable evidence artifact and later runs need no token.

Neuron metadata (cell_type, side, super_class, neurotransmitter) is taken from the
offline Schlegel annotation table (``neurons.parquet``) in BOTH regimes — it is the
same annotation set the paper cites and does not depend on the live synapse counts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

from ..io import read_parquet, write_parquet
from ..paths import DataPaths, cache_root

VERSION = "783"
DATASTACK = "flywire_fafb_public"
SYNAPSE_TABLE = "synapses_nt_v1"
MAT_VERSION = 783

# Per-synapse columns we standardise on (matches the offline feather schema so the two
# sources are drop-in interchangeable downstream).
SYN_COLS = [
    "pre_pt_root_id", "post_pt_root_id", "neuropil",
    "connection_score", "cleft_score",
    "gaba", "ach", "glut", "oct", "ser", "da",
    "pre_pt_position_x", "pre_pt_position_y", "pre_pt_position_z",
    "post_pt_position_x", "post_pt_position_y", "post_pt_position_z",
]
NT_COLS = ("gaba", "ach", "glut", "oct", "ser", "da")

# The public ``synapses_nt_v1`` ``pt_position`` columns are already in NANOMETRES
# (empirically verified: median nearest-neighbour synapse spacing ~1.45 um and a VCH
# input field spanning ~261x163x181 um — both physically correct only under nm; the
# 4/4/40 nm-per-voxel interpretation would inflate every distance ~4x and the field to
# >1 mm). So the geometry layer converts positions to micrometres by a single /1000.
POSITION_NM_PER_UNIT = 1.0  # raw position units are nanometres


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------
class FlyWireSource(Protocol):
    """The minimal interface every derive function depends on."""

    track: str  # "live" | "offline"

    def synapses(
        self, pre_ids: Iterable[int] | None = None, post_ids: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        """Per-synapse rows where pre in pre_ids OR post in post_ids (SYN_COLS schema)."""
        ...


# ---------------------------------------------------------------------------
# Neuron metadata (shared by both sources)
# ---------------------------------------------------------------------------
class NeuronMeta:
    """root_id-keyed cell_type / side / super_class / neurotransmitter lookup."""

    def __init__(self, neurons: pd.DataFrame):
        cols = ["root_id", "cell_type", "side", "super_class", "nt_canonical"]
        # Optional columns used by Family K (NT-confidence + soma location). Carried when
        # present so older dumps without them still load.
        for extra in ("top_nt", "top_nt_conf", "soma_x", "soma_y", "soma_z"):
            if extra in neurons.columns:
                cols.append(extra)
        self.df = neurons[cols].copy()
        self.by_root = self.df.set_index("root_id")

    @classmethod
    def load(cls, version: str = VERSION) -> "NeuronMeta":
        paths = DataPaths.for_version(version)
        return cls(read_parquet(paths.neurons))

    def attach(self, ids: pd.Series | np.ndarray | Iterable[int]) -> pd.DataFrame:
        """Reindex the metadata onto an arbitrary id sequence (NaN where unannotated)."""
        return self.by_root.reindex(pd.Index(ids))

    def root_ids_of_type(self, cell_types: Iterable[str], side: str | None = None) -> np.ndarray:
        m = self.df["cell_type"].isin(list(cell_types))
        if side is not None:
            m &= self.df["side"] == side
        return self.df.loc[m, "root_id"].to_numpy()


# ---------------------------------------------------------------------------
# Live CAVE source (primary)
# ---------------------------------------------------------------------------
class LiveCaveFlyWire:
    track = "live"

    def __init__(self, cache_dir: Path | None = None, mat_version: int = MAT_VERSION):
        import caveclient  # lazy: only needed for the live track

        self.client = caveclient.CAVEclient(DATASTACK)
        self.mat_version = int(mat_version)
        # Cross-version track (Family K) reaches a non-default materialization (e.g. v630).
        # The track string distinguishes it in the ledger; the cache is version-namespaced so
        # v630 and v783 pulls never collide.
        self.track = "live" if self.mat_version == MAT_VERSION else f"live-v{self.mat_version}"
        self.cache_dir = (Path(cache_dir) if cache_dir
                          else (cache_root() / f"v{VERSION}" / "paper" / f"live_cache_v{self.mat_version}"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """CAVE returns position as a 3-vector column + a 'neuropil'-less schema; expand
        positions into x/y/z and ensure every SYN_COL exists."""
        out = pd.DataFrame()
        out["pre_pt_root_id"] = df["pre_pt_root_id"].astype("int64")
        out["post_pt_root_id"] = df["post_pt_root_id"].astype("int64")
        for side in ("pre", "post"):
            pos = np.vstack(df[f"{side}_pt_position"].to_numpy())
            out[f"{side}_pt_position_x"] = pos[:, 0]
            out[f"{side}_pt_position_y"] = pos[:, 1]
            out[f"{side}_pt_position_z"] = pos[:, 2]
        for c in ("connection_score", "cleft_score", *NT_COLS):
            out[c] = df[c].to_numpy() if c in df.columns else np.nan
        # The live synapse table has no neuropil column; left NaN (geometry uses xyz).
        out["neuropil"] = df["neuropil"].to_numpy() if "neuropil" in df.columns else pd.NA
        return out[SYN_COLS]

    def _query_side(self, ids: list[int], side: str, *, retries: int = 5) -> pd.DataFrame:
        kw = {f"{side}_ids": ids if len(ids) > 1 else ids[0]}
        last = None
        for attempt in range(retries):
            try:
                df = self.client.materialize.synapse_query(
                    materialization_version=self.mat_version, synapse_table=SYNAPSE_TABLE, **kw)
                return self._normalise(df)
            except Exception as e:  # transient 5xx / network blips: back off and retry
                last = e
                msg = str(e)
                transient = any(s in msg for s in ("502", "503", "504", "Bad Gateway",
                                                   "Timeout", "Connection", "Max retries"))
                if attempt == retries - 1 or not transient:
                    raise
                # exponential backoff without Date/random: 2,4,8,16 s
                import time
                time.sleep(2 ** (attempt + 1))
        raise last  # pragma: no cover

    def synapses(self, pre_ids=None, post_ids=None) -> pd.DataFrame:
        cache_key = _cache_key(pre_ids, post_ids)
        cache = self.cache_dir / f"{cache_key}.parquet"
        if cache.exists():
            return read_parquet(cache)
        frames = []
        if pre_ids is not None:
            frames.append(self._query_side(sorted(int(i) for i in pre_ids), "pre"))
        if post_ids is not None:
            frames.append(self._query_side(sorted(int(i) for i in post_ids), "post"))
        out = (pd.concat(frames, ignore_index=True).drop_duplicates()
               if frames else pd.DataFrame(columns=SYN_COLS))
        write_parquet(out, cache)
        return out


# ---------------------------------------------------------------------------
# Offline source (cross-check / no-token fallback)
# ---------------------------------------------------------------------------
class OfflineFlyWire:
    track = "offline"

    def __init__(self, feather_path: Path | None = None):
        paths = DataPaths.for_version(VERSION)
        self.feather = Path(feather_path) if feather_path else paths.raw_file(
            f"flywire_synapses_{VERSION}.feather")

    def synapses(self, pre_ids=None, post_ids=None) -> pd.DataFrame:
        from ..motif.synapse_extract import extract_synapses_for_roots
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tf:
            extract_synapses_for_roots(
                self.feather, tf.name, pre_roots=pre_ids, post_roots=post_ids, log_every=0)
            return read_parquet(tf.name)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def cave_token_present() -> bool:
    if os.environ.get("CAVE_TOKEN"):
        return True
    secret = Path.home() / ".cloudvolume" / "secrets" / "cave-secret.json"
    return secret.exists()


def caveclient_available() -> bool:
    try:
        import caveclient  # noqa: F401
        return True
    except Exception:
        return False


def make_source(prefer: str = "auto", mat_version: int = MAT_VERSION) -> FlyWireSource:
    """Return the FlyWire source to use.

    prefer: "auto" (live if token+lib present, else offline), "live", or "offline".
    mat_version: live materialization version (default 783; Family K reaches 630 for the
    cross-version replication track).
    """
    if prefer == "offline":
        return OfflineFlyWire()
    if prefer == "live" or (prefer == "auto" and cave_token_present() and caveclient_available()):
        try:
            return LiveCaveFlyWire(mat_version=mat_version)
        except Exception as e:  # token bad / network down -> degrade, don't crash
            if prefer == "live":
                raise
            print(f"[fw_access] live CAVE unavailable ({e}); falling back to offline")
    return OfflineFlyWire()


def _cache_key(pre_ids, post_ids) -> str:
    """Stable filename for a (pre_ids, post_ids) query, content-addressed by the id set."""
    import hashlib

    def h(ids) -> str:
        if ids is None:
            return "none"
        ids = sorted(int(i) for i in ids)
        digest = hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()[:16]
        return f"{digest}-n{len(ids)}"

    return f"pre-{h(pre_ids)}__post-{h(post_ids)}"
