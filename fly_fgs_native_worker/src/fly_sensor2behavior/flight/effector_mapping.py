"""Explicit raw fly-FGS lane to physical-wing mapping boundary.

The captured fly-FGS application calls its two output lanes ``L`` and ``R``,
but those labels have not been tied to anatomical left and right.  In
particular, the FAFB soma-x laterality convention does not resolve the
application's raw lane labels.  This module therefore makes the two possible
lane assignments explicit hypotheses and is the only boundary at which raw
two-lane virtual-hinge commands may acquire physical FlyBody wing indices.

Neither hypothesis is an anatomical conclusion.  Signed yaw and roll claims
are prohibited until independent evidence resolves the mapping.  Results
that are invariant to swapping the two physical wings may still be compared
across both hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import numpy as np

from .types import WingKinematics


EFFECTOR_MAPPING_SCHEMA_VERSION = "1.0.0"
RAW_APP_LATERALITY_STATUS = "unknown_anatomical_laterality"
SIGNED_BEHAVIOR_CLAIM_POLICY = (
    "signed yaw and roll claims are prohibited until independent evidence "
    "resolves the raw fly-FGS app-lane to physical-wing mapping"
)
EFFECTOR_MAPPING_PROVENANCE: Tuple[str, ...] = (
    "fly-FGS L/R are raw application lane labels, not established anatomical sides",
    "FAFB soma-x laterality does not resolve the raw fly-FGS application lane labels",
    "the selected assignment is an explicit simulation hypothesis, not anatomical evidence",
)


class EffectorLateralityHypothesis(str, Enum):
    """The two exhaustive raw-lane assignments at the physics boundary."""

    RAW_L_TO_PHYSICAL_LEFT = "raw_l_to_physical_left"
    RAW_L_TO_PHYSICAL_RIGHT = "raw_l_to_physical_right"


# Public aliases make manifests and call sites read like explicit hypotheses.
RAW_L_TO_PHYSICAL_LEFT = EffectorLateralityHypothesis.RAW_L_TO_PHYSICAL_LEFT
RAW_L_TO_PHYSICAL_RIGHT = EffectorLateralityHypothesis.RAW_L_TO_PHYSICAL_RIGHT


def _hypothesis(value: Any) -> EffectorLateralityHypothesis:
    if isinstance(value, EffectorLateralityHypothesis):
        return value
    try:
        return EffectorLateralityHypothesis(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "effector laterality hypothesis must explicitly select "
            "RAW_L_TO_PHYSICAL_LEFT or RAW_L_TO_PHYSICAL_RIGHT"
        ) from exc


@dataclass(frozen=True)
class EffectorMappingReceipt:
    """Immutable scientific receipt for one physical-wing assignment."""

    hypothesis: EffectorLateralityHypothesis
    anatomical_status: str = RAW_APP_LATERALITY_STATUS
    provenance: Tuple[str, ...] = EFFECTOR_MAPPING_PROVENANCE
    signed_behavior_claim_policy: str = SIGNED_BEHAVIOR_CLAIM_POLICY
    schema_version: str = EFFECTOR_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis", _hypothesis(self.hypothesis))
        if self.anatomical_status != RAW_APP_LATERALITY_STATUS:
            raise ValueError("raw fly-FGS app-lane anatomical status must remain unknown")
        provenance = tuple(self.provenance)
        if provenance != EFFECTOR_MAPPING_PROVENANCE:
            raise ValueError("effector mapping provenance does not match the locked receipt")
        if self.signed_behavior_claim_policy != SIGNED_BEHAVIOR_CLAIM_POLICY:
            raise ValueError("effector signed-behavior claim policy is not locked")
        if self.schema_version != EFFECTOR_MAPPING_SCHEMA_VERSION:
            raise ValueError("effector mapping schema version mismatch")
        object.__setattr__(self, "provenance", provenance)

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "hypothesis": self.hypothesis.value,
                "anatomical_status": self.anatomical_status,
                "provenance": list(self.provenance),
                "signed_behavior_claim_policy": self.signed_behavior_claim_policy,
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> "EffectorMappingReceipt":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "hypothesis",
            "anatomical_status",
            "provenance",
            "signed_behavior_claim_policy",
        }:
            raise ValueError("effector mapping receipt fields do not match the schema")
        provenance = value["provenance"]
        if not isinstance(provenance, (tuple, list)):
            raise ValueError("effector mapping provenance must be an array")
        return cls(
            schema_version=value["schema_version"],
            hypothesis=value["hypothesis"],
            anatomical_status=value["anatomical_status"],
            provenance=tuple(provenance),
            signed_behavior_claim_policy=value["signed_behavior_claim_policy"],
        )


def effector_mapping_receipt(
    hypothesis: EffectorLateralityHypothesis,
) -> EffectorMappingReceipt:
    """Build the locked receipt for an explicitly selected hypothesis."""

    return EffectorMappingReceipt(hypothesis=_hypothesis(hypothesis))


def map_raw_app_wing_kinematics_to_physical(
    source: WingKinematics,
    hypothesis: EffectorLateralityHypothesis,
) -> WingKinematics:
    """Return a detached physical-left/right command under ``hypothesis``.

    Every two-lane field is permuted with the same lane order.  The optional
    six-axis torque is treated as two consecutive ``(yaw, roll, pitch)``
    triplets and receives the same whole-wing permutation.  The input object
    and all of its arrays are left untouched and are never shared by the
    returned command.
    """

    if not isinstance(source, WingKinematics):
        raise TypeError("source must be WingKinematics")
    selected = _hypothesis(hypothesis)
    lane_order = (
        np.array([0, 1], dtype=int)
        if selected is EffectorLateralityHypothesis.RAW_L_TO_PHYSICAL_LEFT
        else np.array([1, 0], dtype=int)
    )

    def lanes(name: str) -> np.ndarray:
        return np.asarray(getattr(source, name), dtype=float)[lane_order].copy()

    axis = None
    if source.wing_axis_torque_n_m is not None:
        raw_triplets = np.asarray(source.wing_axis_torque_n_m, dtype=float).reshape(2, 3)
        axis = raw_triplets[lane_order, :].reshape(6).copy()

    return WingKinematics(
        phase_rad=float(source.phase_rad),
        frequency_hz=float(source.frequency_hz),
        stroke_rad=lanes("stroke_rad"),
        stroke_velocity_rad_s=lanes("stroke_velocity_rad_s"),
        stroke_acceleration_rad_s2=lanes("stroke_acceleration_rad_s2"),
        angle_of_attack_rad=lanes("angle_of_attack_rad"),
        deviation_rad=lanes("deviation_rad"),
        generalized_torque_n_m=lanes("generalized_torque_n_m"),
        wing_axis_torque_n_m=axis,
    )


__all__ = [
    "EFFECTOR_MAPPING_PROVENANCE",
    "EFFECTOR_MAPPING_SCHEMA_VERSION",
    "EffectorLateralityHypothesis",
    "EffectorMappingReceipt",
    "RAW_APP_LATERALITY_STATUS",
    "RAW_L_TO_PHYSICAL_LEFT",
    "RAW_L_TO_PHYSICAL_RIGHT",
    "SIGNED_BEHAVIOR_CLAIM_POLICY",
    "effector_mapping_receipt",
    "map_raw_app_wing_kinematics_to_physical",
]
