"""Stage 5 - Muscular Projection: complete the figure-ground circuit to both wings.

Traces the paper's two named endpoints (DNbe001 + DNp26), and the seven figure-driven
DNs, through DN -> motor-neuron -> wing-muscle in the male whole-CNS connectome (MCNS),
splits each by wing (ipsi/contra relative to the DN's somaSide), and verifies the result
against the paper's S14-S19 exemplars before extending the map to BOTH wings.

``muscular_config`` is the oracle (paper tables as data); ``trace`` is the DN->MN->muscle
tracer; ``spec`` builds the CircuitSpec; ``verify`` runs self-consistency + robustness.
Run via ``python -m flyconn.muscular {extract,verify}``.
"""

from __future__ import annotations

from . import muscular_config, spec, trace, verify  # noqa: F401

__all__ = ["muscular_config", "spec", "trace", "verify"]
