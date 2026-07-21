"""Derivations for the paper verification, one module per claim family.

Every function takes a ``FlyWireSource`` (live CAVE primary, offline fallback) and a
``NeuronMeta`` lookup, and returns plain dicts/DataFrames that the matching
``oracle.<family>.build_claims`` consumes. The heavy synapse pulls are cached by the
source layer, so re-running the verification is cheap.
"""

from __future__ import annotations
