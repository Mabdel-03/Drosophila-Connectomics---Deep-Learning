import hashlib
from pathlib import Path

from fly_sensor2behavior.validation import (
    MetricComparator,
    ToleranceAuthority,
    load_benchmark_registry,
)


NONNEGATIVE_UPPER_BOUNDED_METRICS = {
    "uniform_motion_abs_max",
    "flicker_motion_abs_max",
    "direction_mirror_abs_error",
    "delayed_onset_luminance_jump",
    "initial_root_position_error_m",
    "paired_initial_state_standardized_delta",
    "paired_timebase_max_abs_delta_s",
    "visual_repeat_max_standardized_delta",
    "mn_silence_repeat_max_standardized_delta",
    "visual_ablation_nonsteering_activation_max_abs_delta",
    "closed_loop_final_rate_ratio",
    "closed_loop_integrated_rate_ratio",
    "mirror_yaw_rate_max_abs_error",
    "uniform_scene_max_abs_control_torque",
    "max_abs_voltage_error_v",
    "max_scaled_residual",
    "reduced_impulse_relative_change",
    "reduced_body_state_relative_change",
    "max_readout_error_v",
    "max_summary_relative_error",
    "normalized_fixture_error",
    "coarse_fine_impulse_relative_change",
    "fine_ultrafine_impulse_relative_change",
    "coarse_fine_body_state_change",
    "fine_ultrafine_body_state_change",
    "steady_flight_impulse_relative_change",
    "steady_flight_body_state_relative_change",
    "saccade_impulse_relative_change",
    "saccade_body_state_relative_change",
    "phase_angle_rmse_ratio_to_matched_baseline_ucb95",
    "max_axis_phase_angle_rmse_ratio_to_matched_baseline_ucb95",
    "coefficient_rmse_ratio_to_matched_baseline_ucb95",
    "matched_baseline_rmse_ratio_to_train_mean_ucb95",
    "indirect_mn_firing_rate_error_ratio_ucb95",
    "calcium_trace_error_ratio_ucb95",
    "wingbeat_frequency_error_ratio_ucb95",
    "splay_state_effect_error_ratio_ucb95",
    "open_loop_curve_rmse_ratio_to_line_null_ucb95",
    "closed_loop_curve_rmse_ratio_to_line_null_ucb95",
}


def test_nonnegative_error_metrics_are_bounded_on_both_sides():
    registry = load_benchmark_registry()
    metrics = {
        metric.metric_id: metric
        for benchmark in registry.cases
        for metric in benchmark.metrics
        if metric.metric_id in NONNEGATIVE_UPPER_BOUNDED_METRICS
    }

    assert set(metrics) == NONNEGATIVE_UPPER_BOUNDED_METRICS
    for metric in metrics.values():
        assert metric.comparator is MetricComparator.BETWEEN_INCLUSIVE
        assert metric.lower_bound == 0.0
        assert metric.upper_bound is not None and metric.upper_bound > 0.0
        assert metric.observe(0.0).passed
        assert not metric.observe(-1e-15).passed


def test_preregistered_protocol_tolerances_are_content_addressed_local_files():
    root = Path(__file__).resolve().parents[1]
    registry = load_benchmark_registry()
    sources = {
        (metric.tolerance_source.source_uri, metric.tolerance_source.source_sha256)
        for benchmark in registry.cases
        for metric in benchmark.metrics
        if metric.tolerance_source.authority
        is ToleranceAuthority.PREREGISTERED_PROTOCOL
    }

    assert sources
    for relative_uri, expected_sha256 in sources:
        assert expected_sha256 is not None
        protocol_path = root / relative_uri
        assert protocol_path.is_file()
        assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == expected_sha256

    protocol_evidence = tuple(
        requirement
        for benchmark in registry.cases
        for requirement in benchmark.required_evidence
        if requirement.source_uri.startswith("data/benchmarks/protocols/")
    )
    assert protocol_evidence
    for requirement in protocol_evidence:
        assert requirement.expected_sha256 is not None
        protocol_path = root / requirement.source_uri
        assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == (
            requirement.expected_sha256
        )


def test_paired_flybody_gate_registers_reset_repeat_and_isolation_oracles():
    registry = load_benchmark_registry()
    benchmark = registry.case("pipeline.retinal_to_flybody_vertical_slice")
    metrics = {metric.metric_id: metric for metric in benchmark.metrics}

    assert registry.version == "1.18.0"
    assert benchmark.version == "2.2.0"
    for metric_id in (
        "paired_initial_state_standardized_delta",
        "paired_timebase_max_abs_delta_s",
        "visual_repeat_max_standardized_delta",
        "mn_silence_repeat_max_standardized_delta",
        "visual_ablation_nonsteering_activation_max_abs_delta",
    ):
        metric = metrics[metric_id]
        assert metric.comparator is MetricComparator.BETWEEN_INCLUSIVE
        assert metric.lower_bound == 0.0
        assert metric.upper_bound == 1e-12

    exact_zero = {
        "mn_silence_upstream_mismatch_count",
        "mn_silence_nontarget_event_mismatch_count",
        "mn_silence_nontarget_muscle_state_mismatch_count",
        "mn_silence_target_event_count",
        "mn_silence_target_rate_max_abs_hz",
        "mn_silence_target_event_availability_violation_count",
    }
    for metric_id in exact_zero:
        metric = metrics[metric_id]
        assert metric.comparator is MetricComparator.EXACT
        assert metric.target_value == 0.0

    for metric_id in (
        "mn_silence_target_activation_max_abs_delta",
        "mn_silence_max_desired_wing_delta_rad",
        "mn_silence_max_measured_wing_delta_rad",
        "mn_silence_final_body_state_standardized_delta",
    ):
        metric = metrics[metric_id]
        assert metric.comparator is MetricComparator.GREATER_THAN
        assert metric.target_value == 0.0


def test_hinge_gate_v2_is_causal_date_grouped_and_sealed():
    registry = load_benchmark_registry()
    benchmark = registry.case("hinge.heldout_wing_prediction")
    metrics = {metric.metric_id: metric for metric in benchmark.metrics}

    assert registry.version == "1.18.0"
    assert benchmark.version == "2.0.0"
    assert "nine completed wingbeats k-9 through k-1" in benchmark.claim
    assert "one-wingbeat-ahead left-wing kinematics" in benchmark.claim
    assert {requirement.requirement_id for requirement in benchmark.required_evidence} == {
        "melis_hdf5_artifact",
        "melis_hdf5_topology",
        "melis_group_identity",
        "hinge_permanent_split",
        "hinge_causal_alignment",
        "hinge_preprocessing_target_protocol",
        "hinge_matched_baseline",
        "hinge_candidate_model",
        "hinge_evaluation_protocol",
        "hinge_sealed_test_execution",
    }

    exact_zero = {
        "source_artifact_digest_mismatch_count",
        "cross_split_acquisition_date_overlap_count",
        "cross_split_session_overlap_count",
        "cross_split_movie_overlap_count",
        "cross_split_source_window_overlap_count",
        "same_or_future_feature_reference_count",
        "feature_availability_violation_count",
        "non_calibration_preprocessing_fit_reference_count",
        "heldout_model_selection_reference_count",
        "matched_test_sample_mismatch_count",
        "nonfinite_prediction_count",
    }
    assert exact_zero < set(metrics)
    for metric_id in exact_zero:
        assert metrics[metric_id].comparator is MetricComparator.EXACT
        assert metrics[metric_id].target_value == 0.0

    date_count = metrics["test_acquisition_date_count"]
    assert date_count.comparator is MetricComparator.GREATER_THAN_OR_EQUAL
    assert date_count.target_value == 5.0

    ratio_metrics = {
        "phase_angle_rmse_ratio_to_matched_baseline_ucb95",
        "max_axis_phase_angle_rmse_ratio_to_matched_baseline_ucb95",
        "coefficient_rmse_ratio_to_matched_baseline_ucb95",
        "matched_baseline_rmse_ratio_to_train_mean_ucb95",
    }
    for metric_id in ratio_metrics:
        metric = metrics[metric_id]
        assert metric.comparator is MetricComparator.BETWEEN_INCLUSIVE
        assert metric.lower_bound == 0.0
        assert metric.upper_bound == 1.0


def test_power_gate_separates_endpoints_and_rejects_lineage_leakage():
    registry = load_benchmark_registry()
    benchmark = registry.case("power_muscle.heldout_calcium_flight_state")
    metrics = {metric.metric_id: metric for metric in benchmark.metrics}

    assert registry.version == "1.18.0"
    assert benchmark.version == "1.1.0"
    assert benchmark.prerequisite_case_ids == ("evidence.cross_atlas_integrity",)
    assert {requirement.requirement_id for requirement in benchmark.required_evidence} == {
        "heldout_power_motor_trials",
        "heldout_power_motor_code",
        "heldout_power_calcium_trials",
        "power_motor_permanent_split",
    }

    exact_zero = {
        "source_artifact_digest_mismatch_count",
        "cross_split_biological_individual_overlap_count",
        "cross_split_recording_overlap_count",
        "duplicate_derived_representation_count",
        "non_calibration_fit_reference_count",
        "heldout_model_selection_reference_count",
        "undeclared_cross_sex_transfer_count",
        "nonfinite_prediction_count",
    }
    endpoint_ratios = {
        "indirect_mn_firing_rate_error_ratio_ucb95",
        "calcium_trace_error_ratio_ucb95",
        "wingbeat_frequency_error_ratio_ucb95",
        "splay_state_effect_error_ratio_ucb95",
    }
    assert set(metrics) == exact_zero | endpoint_ratios
    for metric_id in exact_zero:
        assert metrics[metric_id].comparator is MetricComparator.EXACT
        assert metrics[metric_id].target_value == 0.0
    for metric_id in endpoint_ratios:
        metric = metrics[metric_id]
        assert metric.comparator is MetricComparator.BETWEEN_INCLUSIVE
        assert metric.lower_bound == 0.0
        assert metric.upper_bound == 1.0
        assert metric.tolerance_source.authority.value == "preregistered_protocol"


def test_dng02_gate_separates_open_and_closed_loop_driver_line_tests():
    registry = load_benchmark_registry()
    benchmark = registry.case("dng02.heldout_wingbeat_amplitude")
    metrics = {metric.metric_id: metric for metric in benchmark.metrics}
    evidence = {
        requirement.requirement_id: requirement
        for requirement in benchmark.required_evidence
    }

    assert registry.version == "1.18.0"
    assert benchmark.version == "3.0.0"
    assert "permanent five-driver-line sealed-test partition" in benchmark.claim
    assert "scored separately" in benchmark.claim
    assert "no held-out access before predictions are frozen" in benchmark.claim
    assert "driver-line-grouped" in benchmark.tags
    assert "leakage-safe" in benchmark.tags
    assert set(evidence) == {
        "dng02_public_artifact_receipt",
        "dng02_leakage_safe_protocol",
        "dng02_publication",
        "dng02_activation_onset_mapping",
        "dng02_sealed_test_execution",
    }
    assert evidence["dng02_public_artifact_receipt"].source_uri == (
        "data/benchmarks/receipts/dng02-mendeley-public-artifacts.v1.json"
    )
    assert evidence["dng02_public_artifact_receipt"].expected_sha256 == (
        "edb81f26d6c0b1c75ec04d020366aa45b401f70defc2e1e8514b5fd7bb74c019"
    )
    assert evidence["dng02_leakage_safe_protocol"].source_uri == (
        "data/benchmarks/protocols/dng02-driver-line-heldout.v3.json"
    )
    assert evidence["dng02_leakage_safe_protocol"].expected_sha256 == (
        "bfe167a8768848d3d0ed1d22be8f9ba2d8f9c2fc8c7d21adf72f7ad87561ac45"
    )
    assert evidence["dng02_publication"].source_uri == (
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/"
    )
    assert evidence["dng02_activation_onset_mapping"].source_uri.startswith("urn:")
    assert evidence["dng02_sealed_test_execution"].source_uri == (
        "urn:sha256:7946dfd6b5185c9d2eee39328c2c1eb96ac19ad86d1b00955c0433455b9247ff"
    )

    exact_zero = {
        "source_artifact_digest_mismatch_count",
        "early_heldout_access_count",
        "permanent_split_mismatch_count",
        "cross_split_driver_line_overlap_count",
        "protocol_pooling_count",
        "stable_identity_collision_count",
        "duplicate_derived_representation_count",
        "heldout_model_selection_reference_count",
        "prohibited_exclusion_count",
        "missing_frozen_prediction_count",
        "nonfinite_metric_count",
        "activation_onset_mapping_mismatch_count",
        "activation_trial_count_mismatch_count",
    }
    coverage = {
        "open_loop_evaluable_test_line_count": 5.0,
        "closed_loop_evaluable_test_line_count": 5.0,
        "open_loop_targeted_pair_level_count": 2.0,
        "closed_loop_targeted_pair_level_count": 2.0,
    }
    directional = {
        "open_loop_activation_effect_lcb_rad",
        "closed_loop_activation_effect_lcb_rad",
        "open_loop_targeted_pair_slope_lcb_rad_per_pair",
        "closed_loop_targeted_pair_slope_lcb_rad_per_pair",
    }
    curve_ratios = {
        "open_loop_curve_rmse_ratio_to_line_null_ucb95",
        "closed_loop_curve_rmse_ratio_to_line_null_ucb95",
    }
    assert set(metrics) == exact_zero | set(coverage) | directional | curve_ratios
    for metric_id in exact_zero:
        assert metrics[metric_id].comparator is MetricComparator.EXACT
        assert metrics[metric_id].target_value == 0.0
    for metric_id, target in coverage.items():
        assert (
            metrics[metric_id].comparator
            is MetricComparator.GREATER_THAN_OR_EQUAL
        )
        assert metrics[metric_id].target_value == target
    for metric_id in directional:
        assert metrics[metric_id].comparator is MetricComparator.GREATER_THAN
        assert metrics[metric_id].target_value == 0.0
    for metric_id in curve_ratios:
        metric = metrics[metric_id]
        assert metric.comparator is MetricComparator.BETWEEN_INCLUSIVE
        assert metric.lower_bound == 0.0
        assert metric.upper_bound == 1.0
    assert all(
        metric.tolerance_source.authority.value == "preregistered_protocol"
        for metric in metrics.values()
    )
    assert {
        (
            metric.tolerance_source.source_uri,
            metric.tolerance_source.source_sha256,
        )
        for metric in metrics.values()
    } == {
        (
            "data/benchmarks/protocols/dng02-driver-line-heldout.v3.json",
            "bfe167a8768848d3d0ed1d22be8f9ba2d8f9c2fc8c7d21adf72f7ad87561ac45",
        )
    }
