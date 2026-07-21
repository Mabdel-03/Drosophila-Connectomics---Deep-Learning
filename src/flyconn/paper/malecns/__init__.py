"""Male CNS (MaleCNS v1.0) connectome access for the motor-mapping half (family J).

Uses the public bulk flat files (no auth) downloaded to scratch:
  raw/body-annotations.feather   (type, instance, class/subclass, superclass, somaSide)
  raw/connectome-weights.feather (body_pre, body_post, weight)
  raw/body-neurotransmitters.feather

The DN-name bridge between FlyWire and MaleCNS is the ``type`` string (both use the
Janelia DN nomenclature, e.g. DNp26, DNa04).
"""

from __future__ import annotations

from pathlib import Path

MALECNS_ROOT = Path("/orcd/scratch/orcd/012/mabdel03/connectome_data/malecns_v1.0")
RAW = MALECNS_ROOT / "raw"

ANNOTATIONS = RAW / "body-annotations.feather"
WEIGHTS = RAW / "connectome-weights.feather"
NEUROTRANSMITTERS = RAW / "body-neurotransmitters.feather"
