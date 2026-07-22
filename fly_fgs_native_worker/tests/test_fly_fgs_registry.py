from fly_sensor2behavior.validation import (
    OracleClass,
    RuntimeTier,
    ToleranceAuthority,
    load_benchmark_registry,
)


def test_fly_fgs_integrity_and_worker_cases_are_registered_without_replacing_history():
    registry = load_benchmark_registry()
    integrity = registry.case("fly_fgs.fixed_step_integrity")
    worker = registry.case(
        "pipeline.registered_fly_fgs_to_flybody_vertical_slice"
    )
    runtime = registry.case("fly_fgs.incremental_runtime_parity")
    checkpoint = registry.case("fly_fgs.checkpoint_reentry")

    assert registry.version == "1.18.0"
    assert integrity.oracle_class is OracleClass.FROZEN_REGRESSION
    assert integrity.runtime_tier is RuntimeTier.FAST
    assert integrity.hard_gate is True
    assert integrity.fixture_uri == worker.fixture_uri
    assert integrity.input_sha256 == worker.input_sha256
    assert {metric.metric_id for metric in integrity.metrics} == {
        "registered_source_inventory_match",
        "fixed_step_contract_match",
        "registered_trace_digest_match",
        "full_state_readout_match",
        "t4_activity_peak_to_peak",
        "nod1_voltage_peak_to_peak_v",
        "forbidden_downstream_field_count",
    }
    for metric in integrity.metrics:
        assert (
            metric.tolerance_source.authority
            is ToleranceAuthority.REGISTERED_FIXTURE
        )
        assert metric.tolerance_source.source_uri == integrity.fixture_uri
        assert metric.tolerance_source.source_sha256 == integrity.input_sha256

    assert worker.runtime_tier is RuntimeTier.SCHEDULED
    assert worker.hard_gate is True
    assert "fly_fgs.fixed_step_integrity" in worker.prerequisite_case_ids
    source_metric = next(
        metric
        for metric in worker.metrics
        if metric.metric_id == "registered_fly_fgs_source_scope_match"
    )
    assert (
        source_metric.tolerance_source.authority
        is ToleranceAuthority.REGISTERED_FIXTURE
    )
    assert source_metric.tolerance_source.source_sha256 == worker.input_sha256
    assert runtime.runtime_tier is RuntimeTier.PULL_REQUEST
    assert runtime.prerequisite_case_ids == ("fly_fgs.fixed_step_integrity",)
    assert checkpoint.runtime_tier is RuntimeTier.PULL_REQUEST
    assert checkpoint.prerequisite_case_ids == (
        "fly_fgs.incremental_runtime_parity",
    )
    assert runtime.fixture_uri == integrity.fixture_uri == checkpoint.fixture_uri
    assert runtime.input_sha256 == integrity.input_sha256 == checkpoint.input_sha256

    # The earlier Chromium/Python and historical circuit-to-FlyBody cases remain
    # available as explicitly historical regressions, not silent aliases.
    assert registry.case("nod1.browser_python_parity")
    assert registry.case("pipeline.registered_nod1_to_flybody_vertical_slice")
