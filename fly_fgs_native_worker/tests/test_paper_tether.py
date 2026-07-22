import numpy as np

from fly_sensor2behavior.paper_tether import (
    PAPER_TORQUE_METER_MODES,
    paper_torque_from_support_reaction,
    torque_dyne_cm_to_nm,
    torque_nm_to_dyne_cm,
)


def test_paper_torque_sign_for_known_right_turn_moment():
    # An attempted right turn is -engine-z. The support-on-fly reaction is +z.
    sample = paper_torque_from_support_reaction(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0e-7]),
        np.array([0.0, 0.0, -1.0e-7]),
    )
    np.testing.assert_array_equal(
        sample.fly_generated_moment_engine_n_m,
        [0.0, 0.0, -1.0e-7],
    )
    assert sample.reported_yaw_torque_n_m == 1.0e-7
    assert sample.reported_yaw_torque_dyne_cm == 1.0


def test_paper_torque_sign_for_known_left_turn_moment():
    sample = paper_torque_from_support_reaction(
        np.zeros(3), np.array([0.0, 0.0, -1.0e-7])
    )
    assert sample.reported_yaw_torque_n_m == -1.0e-7
    assert sample.reported_yaw_torque_dyne_cm == -1.0


def test_paper_torque_units_round_trip():
    assert torque_nm_to_dyne_cm(1.0e-7) == 1.0
    assert torque_dyne_cm_to_nm(1.0) == 1.0e-7


def test_meter_modes_and_expanded_sample_contract():
    assert PAPER_TORQUE_METER_MODES == (
        "fixed-load-cell",
        "equality-reaction",
        "comparison",
    )
    sample = paper_torque_from_support_reaction(
        np.zeros(3), np.asarray((0.0, 0.0, -2.0e-7))
    )
    assert sample.support_on_fly_yaw_reaction_engine_n_m == -2.0e-7
    assert sample.attempted_fly_yaw_moment_from_support_engine_n_m == 2.0e-7
    assert sample.fly_generated_moment_engine_n_m[2] == 2.0e-7
