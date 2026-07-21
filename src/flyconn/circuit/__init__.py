"""Circuit-agnostic extract + verify machinery (generalises Stage-4's VCH pipeline).

A circuit is a declarative ``CircuitSpec`` (seeds, edge claims, robustness thresholds,
null models, an oracle of expected values). ``engine`` derives the claimed edges on two
tracks and renders verdicts through the existing ``motif.compare`` engine; ``report``
emits the JSON/Markdown deliverables (including the flat ``refuted_claims`` list the
closed-loop campaign gate reads). The VCH-specific ``motif`` package is left untouched.
"""

from __future__ import annotations

from . import engine, report, spec  # noqa: F401
from .spec import (
    CircuitSpec,
    EdgeClaim,
    NullModel,
    Seed,
    Threshold,
    spec_from_dict,
)

__all__ = [
    "CircuitSpec",
    "EdgeClaim",
    "NullModel",
    "Seed",
    "Threshold",
    "spec_from_dict",
    "engine",
    "report",
    "spec",
]
