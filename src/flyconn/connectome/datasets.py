"""Declarative registry of the connectome datasets we query live (no network here).

Two datasets back the figure-ground -> muscular completion:

  * **fafb** - the adult *female* FlyWire FAFB connectome (materialization v783), the
    brain-side dataset the report's optic-lobe / LLPC1 / DN analysis is built on. The
    chemical-synapse table is ``synapses_nt_v1`` and the published totals are
    reproduced with *no* cleft-score threshold (Methods S1/S15).
  * **mcns** - the *male* whole-CNS connectome (``male-cns:v1.0``), which contains both
    brain and ventral nerve cord and therefore links descending-neuron cell types to
    their monosynaptic motor-neuron targets. Each motor neuron carries a ``somaSide``
    annotation = the body side it innervates, which is the key to the wing-laterality
    (ipsi/contra) split.

IMPORTANT: the exact MCNS datastack name, synapse table and annotation/soma-side table
names are *not* knowable offline. The values below are documented DEFAULTS to be
corrected at first live connect (``ConnectomeClient.list_tables()`` discovers the real
names; write the resolved values to ``configs/cave_mcns.yaml``). Treat every ``mcns``
field as overridable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaveDataset:
    """One CAVE-backed connectome dataset and how to query + cache it.

    Mirrors the spirit of ``config.FileSpec``: a typed, immutable description that the
    client reads, so nothing about a dataset is hardcoded in the query code.
    """

    key: str                 # short id used in cache paths + the public API ("fafb" | "mcns")
    datastack: str           # CAVE datastack name (client = caveclient.CAVEclient(datastack))
    server: str              # global server address for the datastack
    materialization: object  # int (783) or str ("v1.0"); pinned, recorded in every cache meta
    synapse_table: str       # chemical-synapse table for synapse_query
    annotation_tables: tuple[str, ...] = ()   # cell-type / class tables to probe
    soma_side_table: str | None = None        # MCNS: motor-neuron somaSide source
    motor_table: str | None = None            # MCNS: motor-neuron list/subclass source
    cleft_threshold_default: int | None = None  # None = no threshold (FAFB published-total)
    cache_subdir: str = ""   # under cache_root()/v<v>/cave_cache/<cache_subdir>

    def __post_init__(self) -> None:
        if not self.cache_subdir:
            object.__setattr__(self, "cache_subdir", self.key)


# --- FlyWire FAFB (female), materialization v783 -----------------------------
# Matches the report Methods: flywire_fafb_public, synapses_nt_v1, no cleft threshold.
FAFB = CaveDataset(
    key="fafb",
    datastack="flywire_fafb_public",
    server="https://global.daf-apis.com",
    materialization=783,
    synapse_table="synapses_nt_v1",
    annotation_tables=("cell_info",),  # Schlegel-et-al annotation table; verify at connect
    cleft_threshold_default=None,      # published totals use all synapses (S1/S15)
    cache_subdir="fafb",
)

# --- Male whole-CNS (MCNS), male-cns:v1.0 ------------------------------------
# DEFAULTS to be confirmed at first connect via list_tables() (see module docstring).
MCNS = CaveDataset(
    key="mcns",
    datastack="male_cns",          # PLACEHOLDER-LIKELY: confirm exact datastack at connect
    server="https://global.daf-apis.com",
    materialization="v1.0",
    synapse_table="synapses",      # PLACEHOLDER-LIKELY: confirm at connect
    annotation_tables=("cell_info", "neuron_annotations"),  # confirm at connect
    soma_side_table="cell_info",   # somaSide lives on the annotation table; confirm column
    motor_table="cell_info",       # motor-neuron subclass ('wm', etc.); confirm at connect
    cleft_threshold_default=None,
    cache_subdir="mcns",
)

# NOTE (deployment finding): the male-CNS connectome is NOT served as a CAVE datastack on
# global.daf-apis.com — it is distributed as public bulk FLAT FILES (Google Cloud Storage,
# downloaded to scratch). Stage 5 therefore reads MCNS offline via ``flyconn.paper.malecns``
# (body-annotations + connectome-weights feather), NOT through this CAVE entry. The MCNS
# CaveDataset below is retained only as a registry placeholder; the live-CAVE path is used
# for the FAFB brain side (figure -> DN edges). See flyconn.muscular.__main__.cmd_extract.
DATASETS: dict[str, CaveDataset] = {FAFB.key: FAFB, MCNS.key: MCNS}


def get_dataset(dataset: "str | CaveDataset") -> CaveDataset:
    """Resolve a dataset key or pass through a ``CaveDataset`` instance."""
    if isinstance(dataset, CaveDataset):
        return dataset
    try:
        return DATASETS[dataset]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Known datasets: {sorted(DATASETS)}. "
            "Pass a key ('fafb'|'mcns') or a CaveDataset instance."
        ) from None
