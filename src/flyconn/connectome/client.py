"""Thin CAVE client for FAFB + MCNS, with every query cached to scratch.

``caveclient`` (and optionally ``navis``) are imported lazily so the package - and the
whole offline verify path - works in an environment without the ``[cave]`` extra. All
queries return pandas frames whose columns are normalised to the repo vocabulary
(``pre_pt_root_id``, ``post_pt_root_id``, ``cleft_score``, ``nt_canonical``, ...), so the
existing ``flyconn.motif`` / ``flyconn.circuit`` derivations operate on CAVE results
unchanged.

Two usage modes:
  * ``allow_network=True``  (extract stage)  - live queries, populate the cache.
  * ``allow_network=False`` (verify stage)   - cache-only; a miss raises loudly.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..data_prep import schemas
from . import cache as _cache
from .datasets import CaveDataset, get_dataset
from .secrets import read_cave_token

# Column normalisation: CAVE spellings -> repo-canonical names.
_SYN_PRE_ALIASES = ("pre_pt_root_id", "pre_root_id", "pre", "id_pre")
_SYN_POST_ALIASES = ("post_pt_root_id", "post_root_id", "post", "id_post")
_CLEFT_ALIASES = ("cleft_score", "cleftscore", "clf_score")
_NEUROPIL_ALIASES = ("neuropil", "neuropil_region", "roi")
_ROOT_ALIASES = ("pt_root_id", "root_id", "id")
_SIDE_ALIASES = ("somaSide", "soma_side", "side", "hemisphere")
_TYPE_ALIASES = ("cell_type", "type", "cell_type_label")
_SUBCLASS_ALIASES = ("cell_sub_class", "sub_class", "subclass", "super_class", "class")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _caveclient_version() -> str | None:
    try:
        import caveclient

        return getattr(caveclient, "__version__", None)
    except Exception:
        return None


def _rename_first_present(df: pd.DataFrame, aliases, target: str) -> pd.DataFrame:
    """Rename whichever alias column is present to ``target`` (no-op if already named)."""
    if target in df.columns:
        return df
    lower = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return df.rename(columns={lower[a.lower()]: target})
    return df


class ConnectomeClient:
    """A cached CAVE client bound to one dataset (``fafb`` or ``mcns``)."""

    def __init__(
        self,
        dataset: "str | CaveDataset",
        *,
        token: str | None = None,
        version: str = "783",
        allow_network: bool = True,
        cleft_score_threshold: int | None = "__default__",
    ) -> None:
        self.ds: CaveDataset = get_dataset(dataset)
        self.version = version
        self.allow_network = allow_network
        self._token = token  # resolved lazily so cache-only use needs no token
        self._client = None
        self.cleft_score_threshold = (
            self.ds.cleft_threshold_default
            if cleft_score_threshold == "__default__"
            else cleft_score_threshold
        )

    # -- lazy live client ----------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            if not self.allow_network:
                raise RuntimeError(
                    f"{self.ds.key}: live CAVE client requested but allow_network=False."
                )
            import caveclient  # lazy

            tok = read_cave_token(self._token)
            self._client = caveclient.CAVEclient(
                self.ds.datastack, server_address=self.ds.server, auth_token=tok
            )
        return self._client

    # -- discovery -----------------------------------------------------------
    @staticmethod
    def probe(dataset: "str | CaveDataset" = "fafb", *, token: str | None = None,
              timeout: float = 20.0) -> dict:
        """Best-effort reachability check. Never raises; returns a status dict."""
        ds = get_dataset(dataset)
        out = {
            "dataset": ds.key,
            "datastack": ds.datastack,
            "server": ds.server,
            "reachable": False,
            "materialization": None,
            "latency_s": None,
            "error": None,
        }
        t0 = time.time()
        try:
            import caveclient  # lazy

            tok = read_cave_token(token)
            cl = caveclient.CAVEclient(ds.datastack, server_address=ds.server, auth_token=tok)
            info = cl.materialize.get_versions()
            out["reachable"] = True
            out["materialization"] = info
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["latency_s"] = round(time.time() - t0, 3)
        return out

    def list_tables(self) -> list[str]:
        """Live table list (used to discover real MCNS table names)."""
        return list(self.client.materialize.get_tables())

    def materialization_info(self) -> dict:
        return {
            "dataset": self.ds.key,
            "datastack": self.ds.datastack,
            "pinned_materialization": self.ds.materialization,
            "available_versions": list(self.client.materialize.get_versions()),
        }

    # -- internal cache wrapper ---------------------------------------------
    def _cached(self, kind: str, params: dict, fn) -> pd.DataFrame:
        return _cache.cached_or_call(
            self.ds, kind, params, fn,
            version=self.version,
            allow_network=self.allow_network,
            caveclient_version=_caveclient_version(),
            queried_at=_utc_now(),
        )

    # -- core queries --------------------------------------------------------
    def synapse_query(
        self,
        *,
        pre_ids=None,
        post_ids=None,
        cleft_score_threshold: int | None = "__default__",
        columns: list[str] | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Synapse-level query, normalised to repo columns and cached.

        Either ``pre_ids`` or ``post_ids`` (or both) must be given. ``cleft_score_threshold``
        defaults to the dataset default (None for FAFB published totals).
        """
        if pre_ids is None and post_ids is None:
            raise ValueError("synapse_query needs pre_ids and/or post_ids")
        thr = (
            self.cleft_score_threshold
            if cleft_score_threshold == "__default__"
            else cleft_score_threshold
        )
        pre = sorted(int(x) for x in pre_ids) if pre_ids is not None else None
        post = sorted(int(x) for x in post_ids) if post_ids is not None else None
        params = {"pre": pre, "post": post, "cleft_threshold": thr, "columns": columns}

        def _fn() -> pd.DataFrame:
            flt = {}
            if pre is not None:
                flt["pre_pt_root_id"] = pre
            if post is not None:
                flt["post_pt_root_id"] = post
            df = self.client.materialize.synapse_query(
                synapse_table=self.ds.synapse_table, **flt
            )
            return self._normalize_synapses(df, cleft_threshold=thr)

        return self._cached("synapses", params, _fn)

    def _normalize_synapses(self, df: pd.DataFrame, *, cleft_threshold: int | None) -> pd.DataFrame:
        df = _rename_first_present(df, _SYN_PRE_ALIASES, "pre_pt_root_id")
        df = _rename_first_present(df, _SYN_POST_ALIASES, "post_pt_root_id")
        df = _rename_first_present(df, _CLEFT_ALIASES, "cleft_score")
        df = _rename_first_present(df, _NEUROPIL_ALIASES, "neuropil")
        if cleft_threshold is not None and "cleft_score" in df.columns:
            df = df[df["cleft_score"] >= cleft_threshold]
        return df.reset_index(drop=True)

    def connectivity(
        self, root_ids, *, direction: str = "downstream", min_syn: int = 1,
        cleft_score_threshold: int | None = "__default__", refresh: bool = False,
    ) -> pd.DataFrame:
        """Aggregated pre->post synapse-count edge list for a seed set.

        ``direction='downstream'`` restricts the seed to the presynaptic side (edges OUT
        of the seed); ``'upstream'`` restricts it to the postsynaptic side.
        """
        if direction not in ("downstream", "upstream"):
            raise ValueError("direction must be 'downstream' or 'upstream'")
        kw = {"pre_ids": root_ids} if direction == "downstream" else {"post_ids": root_ids}
        syn = self.synapse_query(
            cleft_score_threshold=cleft_score_threshold, refresh=refresh, **kw
        )
        if syn.empty:
            return pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
        edges = (
            syn.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False)
            .size()
            .reset_index(name="syn_count")
        )
        return edges[edges["syn_count"] >= min_syn].reset_index(drop=True)

    def cell_annotations(self, root_ids=None, *, table: str | None = None,
                         refresh: bool = False) -> pd.DataFrame:
        """Cell-type / class / side annotations, normalised to repo column names."""
        table = table or (self.ds.annotation_tables[0] if self.ds.annotation_tables else None)
        if table is None:
            raise ValueError(f"{self.ds.key}: no annotation table configured")
        ids = sorted(int(x) for x in root_ids) if root_ids is not None else None
        params = {"table": table, "root_ids": ids}

        def _fn() -> pd.DataFrame:
            kw = {}
            if ids is not None:
                kw["filter_in_dict"] = {"pt_root_id": ids}
            df = self.client.materialize.query_table(table, **kw)
            return self._normalize_annotations(df)

        return self._cached("annotations", params, _fn)

    def _normalize_annotations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _rename_first_present(df, _ROOT_ALIASES, "root_id")
        df = _rename_first_present(df, _TYPE_ALIASES, "cell_type")
        df = _rename_first_present(df, _SIDE_ALIASES, "somaSide")
        df = _rename_first_present(df, _SUBCLASS_ALIASES, "cell_sub_class")
        if "top_nt" in df.columns and "nt_canonical" not in df.columns:
            df["nt_canonical"] = df["top_nt"].map(schemas.canonical_nt)
        return df.reset_index(drop=True)

    def soma_side(self, root_ids, *, refresh: bool = False) -> pd.DataFrame:
        """MCNS: root_id -> somaSide (the body side a motor neuron innervates)."""
        ann = self.cell_annotations(
            root_ids, table=self.ds.soma_side_table, refresh=refresh
        )
        keep = [c for c in ("root_id", "somaSide", "cell_type", "cell_sub_class") if c in ann.columns]
        return ann[keep].copy()

    def motor_neuron_table(self, *, refresh: bool = False) -> pd.DataFrame:
        """MCNS: motor neurons with subclass + somaSide (and muscle, if annotated)."""
        if self.ds.key != "mcns":
            raise ValueError("motor_neuron_table is MCNS-only")
        return self.cell_annotations(table=self.ds.motor_table, refresh=refresh)
