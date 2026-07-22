import numpy as np

from fly_sensor2behavior.flight import (
    FlightEpisodeRunner,
    FlightSimulationConfig,
    Perturbation,
    PerturbationMode,
    PerturbationTarget,
    convergence_metrics,
)


def short_config(**overrides):
    values = {
        "duration_s": 0.050,
        "physics_dt_s": 0.0001,
        "neural_dt_s": 0.005,
        "logging_dt_s": 0.001,
        "seed": 7,
    }
    values.update(overrides)
    return FlightSimulationConfig(**values)


def test_episode_is_bitwise_deterministic_and_honestly_labelled():
    first = FlightEpisodeRunner().run(short_config())
    second = FlightEpisodeRunner().run(short_config())
    np.testing.assert_array_equal(first.position_world_m, second.position_world_m)
    np.testing.assert_array_equal(first.wing_stroke_rad, second.wing_stroke_rad)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics.status.value == "exploratory"
    assert any("FlyBody/MuJoCo was not executed" in item for item in first.diagnostics.warnings)
    assert first.diagnostics.inferred_spike_count > 0


def test_dng02_positive_control_increases_wing_amplitude_and_vertical_impulse():
    runner = FlightEpisodeRunner()
    baseline = runner.run(short_config())
    activated = runner.run(short_config(dn_drives={"DNg02": 1.0}))
    assert (
        activated.diagnostics.metrics["maximum_absolute_stroke_rad"]
        > baseline.diagnostics.metrics["maximum_absolute_stroke_rad"]
    )
    assert (
        activated.diagnostics.integrated_aerodynamic_impulse_n_s[2]
        > baseline.diagnostics.integrated_aerodynamic_impulse_n_s[2]
    )


def test_dng02_silencing_perturbation_blocks_positive_control():
    runner = FlightEpisodeRunner()
    silencing = Perturbation(
        target_type=PerturbationTarget.DN,
        target="DNg02",
        mode=PerturbationMode.SILENCE,
        start_s=0.0,
        end_s=0.050,
    )
    baseline = runner.run(short_config())
    silenced = runner.run(
        short_config(dn_drives={"DNg02": 1.0}, perturbations=(silencing,))
    )
    assert silenced.diagnostics.metrics["maximum_absolute_stroke_rad"] == (
        baseline.diagnostics.metrics["maximum_absolute_stroke_rad"]
    )


def test_dng02_positive_control_is_independent_of_vch_dch_pathway_factor():
    runner = FlightEpisodeRunner()
    intact = runner.run(short_config(dn_drives={"DNg02": 1.0}))
    pathway_ablated = runner.run(
        short_config(dn_drives={"DNg02": 1.0}, vch_dch_factor=0.0)
    )
    assert pathway_ablated.diagnostics.metrics["maximum_absolute_stroke_rad"] == (
        intact.diagnostics.metrics["maximum_absolute_stroke_rad"]
    )


def test_configured_dn_motor_map_respects_vch_dch_ablation():
    runner = FlightEpisodeRunner()
    mapped = runner.run(
        short_config(
            dn_drives={"DNa04": 1.0},
            dn_motor_rate_gains_hz={"DNa04": {"MN-b1-left": 200.0}},
        )
    )
    ablated = runner.run(
        short_config(
            dn_drives={"DNa04": 1.0},
            dn_motor_rate_gains_hz={"DNa04": {"MN-b1-left": 200.0}},
            vch_dch_factor=0.0,
        )
    )
    assert mapped.motor_event_times_s["MN-b1-left"].size > 0
    assert ablated.motor_event_times_s["MN-b1-left"].size == 0
    assert mapped.muscle_activation["left:b1"][-1] > 0.0


def test_gust_changes_free_flight_trajectory():
    runner = FlightEpisodeRunner()
    gust = Perturbation(
        target_type=PerturbationTarget.ENVIRONMENT,
        target="wind",
        mode=PerturbationMode.GUST,
        start_s=0.010,
        end_s=0.040,
        vector=(1.0, 0.0, 0.0),
    )
    baseline = runner.run(short_config())
    perturbed = runner.run(short_config(perturbations=(gust,)))
    assert not np.allclose(
        baseline.position_world_m[-1], perturbed.position_world_m[-1], atol=1e-10
    )


def test_timestep_halving_converges_below_two_percent():
    runner = FlightEpisodeRunner()
    coarse = runner.run(short_config(duration_s=0.100, physics_dt_s=0.0001))
    fine = runner.run(short_config(duration_s=0.100, physics_dt_s=0.00005))
    errors = convergence_metrics(coarse, fine)
    assert errors["aerodynamic_impulse_relative_error"] < 0.02
    assert errors["standardized_body_state_error"] < 0.02
