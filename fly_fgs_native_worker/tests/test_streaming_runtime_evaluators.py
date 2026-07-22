import shutil

import pytest

from fly_sensor2behavior.cli import main
from fly_sensor2behavior.evaluators import (
    default_evaluators,
    evaluate_canonical_fly_fgs_closed_loop,
    evaluate_effector_raw_lane_physical_hypotheses,
    evaluate_muscle_force_stage_intervention_contract,
    evaluate_streaming_causal_neuromuscular_runtime,
    evaluate_streaming_fresh_process_checkpoint_reentry,
)
from fly_sensor2behavior.validation import (
    GateStatus,
    PromotionGate,
    ValidationReport,
    load_benchmark_registry,
)


def _assert_registered_metrics_pass(case, result):
    assert result.blocked_reason is None
    observed = {value.metric_id: value.value for value in result.values}
    assert set(observed) == {metric.metric_id for metric in case.metrics}
    failed = {
        metric.metric_id: observed[metric.metric_id]
        for metric in case.metrics
        if not metric.observe(observed[metric.metric_id]).passed
    }
    assert failed == {}


def test_registry_adds_only_software_gates_and_preserves_scientific_limits():
    registry = load_benchmark_registry()
    streaming = registry.case("streaming.causal_neuromuscular_runtime")
    fresh_process = registry.case("streaming.fresh_process_checkpoint_reentry")
    muscle = registry.case("muscle.force_stage_intervention_contract")
    effector = registry.case("effector.raw_lane_physical_hypotheses")
    canonical = registry.case("feedback.canonical_fly_fgs_closed_loop")

    assert registry.version == "1.18.0"
    assert len(registry.cases) == 29
    assert streaming.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert fresh_process.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert muscle.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert effector.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert canonical.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert "exploratory" in streaming.tags
    assert "exploratory" in fresh_process.tags
    assert "exploratory" in muscle.tags
    assert "exploratory" in effector.tags
    assert "exploratory" in canonical.tags
    assert "software correctness only" in streaming.claim
    assert "distinct Python processes" in fresh_process.claim
    assert "manufactured virtual-hinge mechanics only" in fresh_process.claim
    assert "not FlyBody" in fresh_process.claim
    assert "software-correctness claim only" in muscle.claim
    assert "mapping software correctness only" in effector.claim
    assert "neither hypothesis resolves anatomical laterality" in effector.claim
    assert "signed yaw/roll conclusions" in effector.claim
    assert "not FlyBody validation" in canonical.claim
    assert "resolved anatomical laterality" in canonical.claim
    assert "signed yaw/roll claim" in canonical.claim
    assert {
        "canonical_scientific_limitations_preserved",
        "canonical_source_receipt_match",
        "canonical_component_checkpoint_digest_match",
    } <= {metric.metric_id for metric in canonical.metrics}
    assert set(default_evaluators(registry)) >= {
        streaming.case_id,
        fresh_process.case_id,
        muscle.case_id,
        effector.case_id,
        canonical.case_id,
    }


def test_streaming_evaluator_exercises_causal_phase_and_checkpoint_contracts():
    case = load_benchmark_registry().case(
        "streaming.causal_neuromuscular_runtime"
    )
    result = evaluate_streaming_causal_neuromuscular_runtime(case)

    _assert_registered_metrics_pass(case, result)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["piecewise_nontrivial_crossing_count"] >= 1.0
    assert observed["streaming_corrupt_checkpoint_rejection_count"] == 2.0
    assert observed["same_interval_future_effect_count"] == 0.0


def test_fresh_process_evaluator_exercises_true_process_boundary_reentry():
    case = load_benchmark_registry().case(
        "streaming.fresh_process_checkpoint_reentry"
    )
    result = evaluate_streaming_fresh_process_checkpoint_reentry(case)

    _assert_registered_metrics_pass(case, result)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["fresh_process_unique_pid_count"] == 5.0
    assert observed["fresh_process_success_count"] == 3.0
    assert observed["fresh_process_runtime_source_receipt_match"] == 1.0
    assert observed["fresh_process_tail_exact_match"] == 1.0
    assert observed["fresh_process_final_bridge_checkpoint_match"] == 1.0
    assert observed["fresh_process_final_mechanics_checkpoint_match"] == 1.0
    assert observed["fresh_process_corrupt_checkpoint_rejection_count"] == 1.0
    assert observed["fresh_process_receipt_mismatch_rejection_count"] == 1.0
    assert observed["fresh_process_scientific_status_match"] == 1.0


def test_muscle_evaluator_exercises_true_force_stage_interventions():
    case = load_benchmark_registry().case(
        "muscle.force_stage_intervention_contract"
    )
    result = evaluate_muscle_force_stage_intervention_contract(case)

    _assert_registered_metrics_pass(case, result)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["silence_onset_effective_force_n"] == 0.0
    assert observed["silence_nmj_suppression_mismatch_count"] == 0.0
    assert observed["scale_hidden_state_mismatch_count"] == 0.0
    assert observed["intervention_schedule_corruption_rejection_count"] == 2.0
    assert observed["intervention_ledger_corruption_rejection_count"] == 1.0


def test_effector_evaluator_exercises_both_unresolved_lane_hypotheses():
    case = load_benchmark_registry().case(
        "effector.raw_lane_physical_hypotheses"
    )
    result = evaluate_effector_raw_lane_physical_hypotheses(case)

    _assert_registered_metrics_pass(case, result)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["effector_mapping_involution_mismatch_count"] == 0.0
    assert observed["effector_source_mutation_count"] == 0.0
    assert observed["effector_receipt_mismatch_count"] == 0.0
    assert observed["effector_signed_claim_prohibition_preserved"] == 1.0
    assert observed["effector_cross_hypothesis_factory_call_count"] == 0.0


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_canonical_evaluator_runs_live_source_feedback_and_fresh_stack_reentry():
    case = load_benchmark_registry().case(
        "feedback.canonical_fly_fgs_closed_loop"
    )
    result = evaluate_canonical_fly_fgs_closed_loop(case)

    _assert_registered_metrics_pass(case, result)
    observed = {value.metric_id: value.value for value in result.values}
    assert observed["canonical_circuit_observation_count"] == 3.0
    assert observed["canonical_feedback_mismatch_count"] == 0.0
    assert observed["canonical_scientific_limitations_preserved"] == 1.0


def test_streaming_gate_is_selectable_through_validation_cli(tmp_path, capsys):
    output = tmp_path / "streaming-validation.json"

    return_code = main(
        (
            "validate",
            "--output",
            str(output),
            "--case",
            "streaming.causal_neuromuscular_runtime",
            "--no-dependents",
            "--no-prerequisites",
        )
    )
    capsys.readouterr()
    report = ValidationReport.from_json(output.read_text(encoding="utf-8"))

    assert return_code == 0
    assert report.selected_case_ids == (
        "streaming.causal_neuromuscular_runtime",
    )
    assert len(report.results) == 1
    assert report.results[0].status is GateStatus.PASS


def test_fresh_process_gate_is_selectable_through_validation_cli(
    tmp_path, capsys
):
    output = tmp_path / "fresh-process-validation.json"

    return_code = main(
        (
            "validate",
            "--output",
            str(output),
            "--case",
            "streaming.fresh_process_checkpoint_reentry",
            "--no-dependents",
            "--no-prerequisites",
        )
    )
    capsys.readouterr()
    report = ValidationReport.from_json(output.read_text(encoding="utf-8"))

    assert return_code == 0
    assert report.selected_case_ids == (
        "streaming.fresh_process_checkpoint_reentry",
    )
    assert len(report.results) == 1
    assert report.results[0].status is GateStatus.PASS
