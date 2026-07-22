from __future__ import annotations

import numpy as np
import pytest

from fly_sensor2behavior.flight.effector_mapping import (
    EFFECTOR_MAPPING_PROVENANCE,
    RAW_APP_LATERALITY_STATUS,
    RAW_L_TO_PHYSICAL_LEFT,
    RAW_L_TO_PHYSICAL_RIGHT,
    SIGNED_BEHAVIOR_CLAIM_POLICY,
    effector_mapping_receipt,
    map_raw_app_wing_kinematics_to_physical,
)
from fly_sensor2behavior.flight.types import WingKinematics


_TWO_LANE_FIELDS = (
    "stroke_rad",
    "stroke_velocity_rad_s",
    "stroke_acceleration_rad_s2",
    "angle_of_attack_rad",
    "deviation_rad",
    "generalized_torque_n_m",
)


def _raw_command(*, axis: bool = True) -> WingKinematics:
    return WingKinematics(
        phase_rad=1.25,
        frequency_hz=218.0,
        stroke_rad=np.array([1.0, -2.0]),
        stroke_velocity_rad_s=np.array([3.0, -5.0]),
        stroke_acceleration_rad_s2=np.array([7.0, -11.0]),
        angle_of_attack_rad=np.array([13.0, -17.0]),
        deviation_rad=np.array([19.0, -23.0]),
        generalized_torque_n_m=np.array([29.0, -31.0]),
        wing_axis_torque_n_m=(
            np.array([37.0, 41.0, 43.0, -47.0, -53.0, -59.0])
            if axis
            else None
        ),
    )


def test_both_hypotheses_map_every_lane_and_axis_triplet_consistently():
    raw = _raw_command()
    raw_before = {
        name: np.asarray(getattr(raw, name)).copy()
        for name in (*_TWO_LANE_FIELDS, "wing_axis_torque_n_m")
    }

    left_hypothesis = map_raw_app_wing_kinematics_to_physical(
        raw, RAW_L_TO_PHYSICAL_LEFT
    )
    right_hypothesis = map_raw_app_wing_kinematics_to_physical(
        raw, RAW_L_TO_PHYSICAL_RIGHT
    )

    for name in _TWO_LANE_FIELDS:
        source = raw_before[name]
        np.testing.assert_array_equal(getattr(left_hypothesis, name), source)
        np.testing.assert_array_equal(getattr(right_hypothesis, name), source[::-1])
        # The two hypotheses are exact physical-wing mirrors.
        assert right_hypothesis.__getattribute__(name)[0] == left_hypothesis.__getattribute__(name)[1]
        assert right_hypothesis.__getattribute__(name)[1] == left_hypothesis.__getattribute__(name)[0]
        assert not np.shares_memory(getattr(left_hypothesis, name), getattr(raw, name))
        assert not np.shares_memory(getattr(right_hypothesis, name), getattr(raw, name))

    np.testing.assert_array_equal(
        left_hypothesis.wing_axis_torque_n_m,
        np.array([37.0, 41.0, 43.0, -47.0, -53.0, -59.0]),
    )
    np.testing.assert_array_equal(
        right_hypothesis.wing_axis_torque_n_m,
        np.array([-47.0, -53.0, -59.0, 37.0, 41.0, 43.0]),
    )
    np.testing.assert_array_equal(
        right_hypothesis.wing_axis_torque_n_m[:3],
        left_hypothesis.wing_axis_torque_n_m[3:],
    )
    np.testing.assert_array_equal(
        right_hypothesis.wing_axis_torque_n_m[3:],
        left_hypothesis.wing_axis_torque_n_m[:3],
    )
    assert not np.shares_memory(left_hypothesis.wing_axis_torque_n_m, raw.wing_axis_torque_n_m)
    assert not np.shares_memory(right_hypothesis.wing_axis_torque_n_m, raw.wing_axis_torque_n_m)

    # The source is byte-for-byte unchanged after both mappings.
    for name, before in raw_before.items():
        np.testing.assert_array_equal(getattr(raw, name), before)


def test_absolute_and_unordered_commands_are_hypothesis_invariant():
    raw = _raw_command()
    first = map_raw_app_wing_kinematics_to_physical(raw, RAW_L_TO_PHYSICAL_LEFT)
    second = map_raw_app_wing_kinematics_to_physical(raw, RAW_L_TO_PHYSICAL_RIGHT)

    assert first.phase_rad == second.phase_rad == raw.phase_rad
    assert first.frequency_hz == second.frequency_hz == raw.frequency_hz
    for name in _TWO_LANE_FIELDS:
        np.testing.assert_array_equal(
            np.sort(np.abs(getattr(first, name))),
            np.sort(np.abs(getattr(second, name))),
        )
    np.testing.assert_array_equal(
        np.sort(np.abs(first.wing_axis_torque_n_m)),
        np.sort(np.abs(second.wing_axis_torque_n_m)),
    )


def test_optional_axis_torque_stays_absent_and_selection_is_mandatory():
    raw = _raw_command(axis=False)
    mapped = map_raw_app_wing_kinematics_to_physical(
        raw, RAW_L_TO_PHYSICAL_RIGHT
    )
    assert mapped.wing_axis_torque_n_m is None
    with pytest.raises(ValueError, match="explicitly select"):
        map_raw_app_wing_kinematics_to_physical(raw, None)


def test_receipt_locks_unknown_anatomy_provenance_and_signed_claim_policy():
    receipt = effector_mapping_receipt(RAW_L_TO_PHYSICAL_LEFT)
    assert receipt.anatomical_status == RAW_APP_LATERALITY_STATUS
    assert receipt.provenance == EFFECTOR_MAPPING_PROVENANCE
    assert any("soma-x" in item for item in receipt.provenance)
    assert receipt.signed_behavior_claim_policy == SIGNED_BEHAVIOR_CLAIM_POLICY
    assert "prohibited" in receipt.signed_behavior_claim_policy
    assert receipt.to_dict()["hypothesis"] == "raw_l_to_physical_left"
