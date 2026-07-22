from fly_sensor2behavior.validation import (
    MetricComparator,
    OracleClass,
    PromotionGate,
    RuntimeTier,
    ToleranceAuthority,
    load_benchmark_registry,
)


def test_flybody_checkpoint_reentry_is_a_hard_scheduled_exact_gate():
    registry = load_benchmark_registry()
    case = registry.case("flybody.checkpoint_reentry")

    assert case.version == "1.0.0"
    assert case.oracle_class is OracleClass.INVARIANT_METAMORPHIC
    assert case.promotion_gate is PromotionGate.SOFTWARE_CORRECT
    assert case.runtime_tier is RuntimeTier.SCHEDULED
    assert case.hard_gate is True
    assert case.prerequisite_case_ids == (
        "contracts.json_roundtrip",
        "flybody.adapter_analytic_smoke",
    )
    assert {metric.metric_id for metric in case.metrics} == {
        "restored_checkpoint_max_abs_delta",
        "restored_tail_max_abs_delta",
        "corrupted_checkpoint_rejection_count",
        "cross_model_checkpoint_rejection_count",
        "rejected_restore_state_max_abs_delta",
    }
    for metric in case.metrics:
        assert metric.comparator is MetricComparator.EXACT
        assert metric.tolerance_source.authority is (
            ToleranceAuthority.ENGINEERING_REQUIREMENT
        )
        if metric.metric_id.endswith("rejection_count"):
            assert metric.target_value == 1.0
        else:
            assert metric.target_value == 0.0
