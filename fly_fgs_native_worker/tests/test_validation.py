import hashlib
import json
from pathlib import Path

import pytest

from fly_sensor2behavior.validation import (
    BenchmarkCase,
    BenchmarkRegistry,
    BenchmarkResult,
    DataSplit,
    EvidenceReceipt,
    EvidenceRequirement,
    EvaluatorResult,
    GateStatus,
    MetricComparator,
    MetricSpec,
    MetricValue,
    OracleClass,
    PromotionGate,
    RuntimeTier,
    SourceDigest,
    ToleranceAuthority,
    ToleranceSource,
    ValidationContractError,
    ValidationReport,
    ValidationRunner,
    default_benchmark_registry_path,
    load_benchmark_registry,
    verified_preregistered_protocol_files,
)


ZERO_SHA = "0" * 64
ONE_SHA = "1" * 64


def source(
    authority=ToleranceAuthority.ENGINEERING_REQUIREMENT,
    uri="urn:test:tolerance",
    source_sha256=None,
):
    return ToleranceSource(
        authority=authority,
        source_uri=uri,
        rationale="preregistered test fixture acceptance rule",
        source_sha256=source_sha256,
    )


def metric(metric_id="error", target=0.0, comparator=MetricComparator.LESS_THAN_OR_EQUAL, authority=None):
    selected_authority = authority or ToleranceAuthority.ENGINEERING_REQUIREMENT
    return MetricSpec(
        metric_id=metric_id,
        unit="1",
        comparator=comparator,
        target_value=target,
        tolerance_source=source(
            selected_authority,
            source_sha256=(
                ZERO_SHA
                if selected_authority is ToleranceAuthority.PREREGISTERED_PROTOCOL
                else None
            ),
        ),
    )


def case(
    case_id="software.case",
    *,
    gate=PromotionGate.SOFTWARE_CORRECT,
    oracle=OracleClass.EXACT_MANUFACTURED,
    metrics=None,
    prerequisites=(),
    evidence=(),
    data_split=DataSplit.NOT_APPLICABLE,
    hard=True,
    fixture_uri=None,
    input_sha256=None,
):
    return BenchmarkCase(
        case_id=case_id,
        version="1.0.0",
        title="Test benchmark",
        claim="The registered scalar satisfies its source-backed tolerance.",
        oracle_class=oracle,
        promotion_gate=gate,
        metrics=tuple(metrics or (metric(),)),
        dependency_keys=("component.%s" % case_id.replace(".", "_"),),
        prerequisite_case_ids=tuple(prerequisites),
        required_evidence=tuple(evidence),
        data_split=data_split,
        runtime_tier=RuntimeTier.FAST,
        hard_gate=hard,
        fixture_uri=fixture_uri,
        input_sha256=input_sha256,
    )


def registry(*cases):
    return BenchmarkRegistry(
        registry_id="test-registry",
        version="1.0.0",
        description="Deterministic validation test registry",
        cases=tuple(cases),
    )


def values(**items):
    return EvaluatorResult(
        values=tuple(MetricValue(metric_id=key, value=value) for key, value in items.items())
    )


def test_default_declarative_registry_is_valid_complete_and_deterministic():
    first = load_benchmark_registry()
    second = BenchmarkRegistry.from_json(first.to_json(indent=2))

    assert first == second
    assert first.content_sha256 == second.content_sha256
    assert first.version == "1.18.0"
    assert len(first.cases) == 29
    assert {case.oracle_class for case in first.cases} == set(OracleClass)
    assert {case.promotion_gate for case in first.cases} == set(PromotionGate)
    assert all(case.metrics for case in first.cases)
    assert all(metric.tolerance_source.source_uri for case in first.cases for metric in case.metrics)


def test_registry_load_rejects_missing_or_tampered_preregistered_protocol(tmp_path):
    source_registry = default_benchmark_registry_path()
    registry = load_benchmark_registry(source_registry)
    target_registry = tmp_path / "data" / "benchmarks" / "registry.v1.json"
    target_registry.parent.mkdir(parents=True)
    target_registry.write_bytes(source_registry.read_bytes())
    protocol_dir = target_registry.parent / "protocols"
    protocol_dir.mkdir()
    protocol_files = verified_preregistered_protocol_files(registry, source_registry)
    for _source_uri, source_path, _sha256, source_bytes in protocol_files:
        (protocol_dir / source_path.name).write_bytes(source_bytes)

    assert load_benchmark_registry(target_registry) == registry
    first_protocol = protocol_dir / protocol_files[0][1].name
    first_protocol.write_bytes(first_protocol.read_bytes() + b"\n")
    with pytest.raises(ValidationContractError, match="digest mismatch"):
        load_benchmark_registry(target_registry)

    first_protocol.unlink()
    with pytest.raises(ValidationContractError, match="unavailable"):
        load_benchmark_registry(target_registry)


def test_metric_comparators_are_evaluated_by_the_registry_not_the_evaluator():
    approximate = MetricSpec(
        metric_id="voltage",
        unit="V",
        comparator="approximate",
        target_value=-0.05,
        absolute_tolerance=0.001,
        relative_tolerance=0.01,
        tolerance_source=source(),
    )
    assert approximate.observe(-0.049).passed
    assert not approximate.observe(-0.048).passed

    interval = MetricSpec(
        metric_id="coverage",
        unit="1",
        comparator="between_inclusive",
        lower_bound=0.85,
        upper_bound=0.95,
        tolerance_source=source(),
    )
    assert interval.observe(0.9).passed
    assert not interval.observe(0.8).passed

    with pytest.raises(ValidationContractError, match="require a tolerance"):
        MetricSpec(
            metric_id="bad",
            unit="1",
            comparator="approximate",
            target_value=1.0,
            tolerance_source=source(),
        )


def test_held_out_empirical_case_requires_public_evidence_and_public_tolerance():
    public_metric = metric(
        "effect",
        target=0.0,
        comparator=MetricComparator.GREATER_THAN,
        authority=ToleranceAuthority.PEER_REVIEWED_PUBLICATION,
    )
    with pytest.raises(ValidationContractError, match="require public evidence"):
        case(
            "empirical.missing_evidence",
            oracle=OracleClass.HELD_OUT_EMPIRICAL,
            gate=PromotionGate.EMPIRICALLY_VALIDATED,
            metrics=(public_metric,),
            data_split=DataSplit.HELD_OUT,
        )

    requirement = EvidenceRequirement(
        requirement_id="heldout_trials",
        description="Immutable held-out public trials",
        source_uri="https://example.org/public-data",
    )
    with pytest.raises(ValidationContractError, match="public dataset"):
        case(
            "empirical.private_tolerance",
            oracle=OracleClass.HELD_OUT_EMPIRICAL,
            gate=PromotionGate.EMPIRICALLY_VALIDATED,
            metrics=(metric("effect", comparator=MetricComparator.GREATER_THAN),),
            evidence=(requirement,),
            data_split=DataSplit.HELD_OUT,
        )

    preregistered = case(
        "empirical.preregistered",
        oracle=OracleClass.HELD_OUT_EMPIRICAL,
        gate=PromotionGate.EMPIRICALLY_VALIDATED,
        metrics=(
            metric(
                "effect",
                comparator=MetricComparator.GREATER_THAN,
                authority=ToleranceAuthority.PREREGISTERED_PROTOCOL,
            ),
        ),
        evidence=(requirement,),
        data_split=DataSplit.HELD_OUT,
    )
    assert (
        preregistered.metrics[0].tolerance_source.authority
        is ToleranceAuthority.PREREGISTERED_PROTOCOL
    )


def test_preregistered_tolerance_requires_a_content_digest():
    with pytest.raises(ValidationContractError, match="immutable source_sha256"):
        source(authority=ToleranceAuthority.PREREGISTERED_PROTOCOL)

    pinned = source(
        authority=ToleranceAuthority.PREREGISTERED_PROTOCOL,
        source_sha256=ZERO_SHA,
    )
    assert pinned.source_sha256 == ZERO_SHA


def test_frozen_regression_requires_fixture_or_released_baseline_authority():
    with pytest.raises(ValidationContractError, match="registered fixture"):
        case(
            "legacy.trace",
            oracle=OracleClass.FROZEN_REGRESSION,
            metrics=(metric(),),
        )

    with pytest.raises(ValidationContractError, match="fixture_uri and input_sha256"):
        case(
            "legacy.unpinned",
            oracle=OracleClass.FROZEN_REGRESSION,
            metrics=(
                MetricSpec(
                    metric_id="error",
                    unit="1",
                    comparator=MetricComparator.LESS_THAN_OR_EQUAL,
                    target_value=0.0,
                    tolerance_source=source(
                        ToleranceAuthority.REGISTERED_FIXTURE,
                        source_sha256=ZERO_SHA,
                    ),
                ),
            ),
        )

    frozen = case(
        "legacy.trace",
        oracle=OracleClass.FROZEN_REGRESSION,
        metrics=(
            MetricSpec(
                metric_id="error",
                unit="1",
                comparator=MetricComparator.LESS_THAN_OR_EQUAL,
                target_value=0.0,
                tolerance_source=source(
                    ToleranceAuthority.REGISTERED_FIXTURE,
                    uri="urn:test:frozen-fixture",
                    source_sha256=ZERO_SHA,
                ),
            ),
        ),
        fixture_uri="urn:test:frozen-fixture",
        input_sha256=ZERO_SHA,
    )
    assert frozen.oracle_class is OracleClass.FROZEN_REGRESSION


def test_dependency_selection_adds_downstream_cases_and_prerequisites_in_order():
    base = case("solver.base")
    parity = case(
        "solver.parity",
        gate=PromotionGate.NUMERICALLY_CONVERGED,
        oracle=OracleClass.DIFFERENTIAL_CONVERGENCE,
        prerequisites=(base.case_id,),
    )
    downstream = case(
        "flight.closed_loop",
        gate=PromotionGate.STRUCTURALLY_SUPPORTED,
        oracle=OracleClass.INVARIANT_METAMORPHIC,
        prerequisites=(parity.case_id,),
    )
    suite = registry(downstream, parity, base)

    selected = suite.select_cases(changed_dependencies=(base.dependency_keys[0],))
    assert [item.case_id for item in selected] == [
        "solver.base",
        "solver.parity",
        "flight.closed_loop",
    ]

    direct = suite.select_cases(
        case_ids=(parity.case_id,), include_dependents=False, include_prerequisites=True
    )
    assert [item.case_id for item in direct] == ["solver.base", "solver.parity"]


@pytest.mark.parametrize(
    ("prerequisite_result", "expected_status"),
    (
        (values(error=1.0), GateStatus.FAIL),
        (EvaluatorResult.blocked("required worker is unavailable"), GateStatus.BLOCKED),
        (
            EvaluatorResult.not_applicable("case does not apply to this backend"),
            GateStatus.NOT_APPLICABLE,
        ),
    ),
)
def test_nonpassing_selected_prerequisite_prevents_dependent_pass(
    prerequisite_result, expected_status
):
    prerequisite = case("solver.prerequisite")
    dependent = case(
        "solver.dependent", prerequisites=(prerequisite.case_id,)
    )
    suite = registry(prerequisite, dependent)
    report = ValidationRunner(
        suite,
        {
            prerequisite.case_id: lambda _: prerequisite_result,
            dependent.case_id: lambda _: values(error=-1.0),
        },
    ).run(evaluation_id="prerequisite-propagation")
    results = {result.case_id: result for result in report.results}

    propagated = results[dependent.case_id]
    assert propagated.status is expected_status
    assert propagated.observations
    assert all(observation.passed for observation in propagated.observations)
    assert prerequisite.case_id in propagated.reason
    assert expected_status.value in propagated.reason


def test_hard_not_applicable_sibling_prevents_gate_pass():
    passing = case("software.pass")
    unavailable = case("software.not_applicable")
    suite = registry(passing, unavailable)
    report = ValidationRunner(
        suite,
        {
            passing.case_id: lambda _: values(error=-1.0),
            unavailable.case_id: lambda _: EvaluatorResult.not_applicable(
                "optional implementation is unavailable"
            ),
        },
    ).run(evaluation_id="mixed-not-applicable")

    software_gate = report.gate_vector[0]
    assert software_gate.passed_count == 1
    assert software_gate.not_applicable_count == 1
    assert software_gate.status is GateStatus.NOT_APPLICABLE
    assert report.promotion_ceiling is None


def test_partial_report_declares_omissions_and_cannot_claim_promotion():
    selected = case("software.selected")
    omitted_hard = case("software.omitted_hard")
    omitted_soft = case("software.omitted_soft", hard=False)
    suite = registry(selected, omitted_hard, omitted_soft)
    report = ValidationRunner(
        suite, {selected.case_id: lambda _: values(error=-1.0)}
    ).run(
        evaluation_id="partial-suite",
        case_ids=(selected.case_id,),
        include_dependents=False,
        include_prerequisites=False,
    )

    assert not report.suite_complete
    assert set(report.omitted_case_ids) == {
        omitted_hard.case_id,
        omitted_soft.case_id,
    }
    assert report.gate_vector[0].status is GateStatus.NOT_APPLICABLE
    assert report.gate_vector[0].omitted_case_ids == (omitted_hard.case_id,)
    assert report.overall_status is GateStatus.NOT_APPLICABLE
    assert report.promotion_ceiling is None
    assert not report.passes_through(PromotionGate.SOFTWARE_CORRECT)
    report.validate_against(suite)


def test_registry_rejects_unknown_prerequisites_and_cycles():
    with pytest.raises(ValidationContractError, match="unknown prerequisites"):
        registry(case("orphan", prerequisites=("missing.case",)))

    first = case("cycle.first", prerequisites=("cycle.second",))
    second = case("cycle.second", prerequisites=("cycle.first",))
    with pytest.raises(ValidationContractError, match="contains a cycle"):
        registry(first, second)


def test_runner_applies_tolerance_and_hard_gate_vector_without_averaging():
    passing = case("software.pass")
    failing = case("software.fail")
    suite = registry(passing, failing)
    runner = ValidationRunner(
        suite,
        {
            passing.case_id: lambda _: values(error=-1.0),
            failing.case_id: lambda _: values(error=0.1),
        },
    )
    report = runner.run(evaluation_id="tolerance-run")

    result_status = {result.case_id: result.status for result in report.results}
    assert result_status == {
        "software.fail": GateStatus.FAIL,
        "software.pass": GateStatus.PASS,
    }
    software_gate = report.gate_vector[0]
    assert software_gate.status is GateStatus.FAIL
    assert software_gate.passed_count == 1
    assert software_gate.failed_count == 1
    assert report.overall_status is GateStatus.FAIL
    assert report.promotion_ceiling is None


def test_missing_evaluator_is_blocked_and_exception_is_fail():
    missing = case("software.missing")
    crashed = case("software.crashed")

    def raise_error(_):
        raise RuntimeError("deterministic crash")

    report = ValidationRunner(registry(missing, crashed), {crashed.case_id: raise_error}).run(
        evaluation_id="failure-modes"
    )
    results = {result.case_id: result for result in report.results}
    assert results[missing.case_id].status is GateStatus.BLOCKED
    assert results[crashed.case_id].status is GateStatus.FAIL
    assert "RuntimeError" in results[crashed.case_id].reason


def test_passing_empirical_values_without_evidence_are_forced_to_blocked():
    requirement = EvidenceRequirement(
        requirement_id="heldout_trials",
        description="Held-out public trials",
        source_uri="https://example.org/public-data",
    )
    empirical = case(
        "empirical.effect",
        oracle=OracleClass.HELD_OUT_EMPIRICAL,
        gate=PromotionGate.EMPIRICALLY_VALIDATED,
        metrics=(
            metric(
                "effect",
                target=0.0,
                comparator=MetricComparator.GREATER_THAN,
                authority=ToleranceAuthority.PUBLIC_DATASET,
            ),
        ),
        evidence=(requirement,),
        data_split=DataSplit.HELD_OUT,
    )
    runner = ValidationRunner(
        registry(empirical),
        {empirical.case_id: lambda _: values(effect=100.0)},
    )
    report = runner.run(evaluation_id="no-evidence")

    assert report.results[0].status is GateStatus.BLOCKED
    assert "required evidence is unavailable" in report.results[0].reason
    assert report.gate_vector[-1].status is GateStatus.BLOCKED
    assert report.overall_status is GateStatus.BLOCKED


def test_unpinned_empirical_receipt_cannot_satisfy_heldout_gate():
    requirement = EvidenceRequirement(
        requirement_id="heldout_trials",
        description="Held-out public trials awaiting a registered digest",
        source_uri="https://example.org/public-data",
    )
    empirical = case(
        "empirical.unpinned",
        oracle=OracleClass.HELD_OUT_EMPIRICAL,
        gate=PromotionGate.EMPIRICALLY_VALIDATED,
        metrics=(
            metric(
                "effect",
                target=0.0,
                comparator=MetricComparator.GREATER_THAN,
                authority=ToleranceAuthority.PUBLIC_DATASET,
            ),
        ),
        evidence=(requirement,),
        data_split=DataSplit.HELD_OUT,
    )
    receipt = EvidenceReceipt(
        requirement_id="heldout_trials",
        source_uri=requirement.source_uri,
        artifact_sha256=ONE_SHA,
        verified_by="sha256 over an otherwise valid artifact",
    )
    report = ValidationRunner(
        registry(empirical),
        {
            empirical.case_id: lambda _: EvaluatorResult(
                values=(MetricValue("effect", 1.0),), evidence=(receipt,)
            )
        },
    ).run(evaluation_id="unpinned-evidence")

    assert report.results[0].status is GateStatus.BLOCKED
    assert "digest is not registered" in report.results[0].reason


def test_empirical_case_passes_only_with_matching_hashed_receipt():
    requirement = EvidenceRequirement(
        requirement_id="heldout_trials",
        description="Held-out public trials",
        source_uri="https://example.org/public-data",
        expected_sha256=ONE_SHA,
    )
    empirical = case(
        "empirical.effect",
        oracle=OracleClass.HELD_OUT_EMPIRICAL,
        gate=PromotionGate.EMPIRICALLY_VALIDATED,
        metrics=(
            metric(
                "effect",
                target=0.0,
                comparator=MetricComparator.GREATER_THAN,
                authority=ToleranceAuthority.PUBLIC_DATASET,
            ),
        ),
        evidence=(requirement,),
        data_split=DataSplit.HELD_OUT,
    )

    def evaluator(_):
        return EvaluatorResult(
            values=(MetricValue("effect", 0.2),),
            evidence=(
                EvidenceReceipt(
                    requirement_id="heldout_trials",
                    source_uri="https://example.org/public-data",
                    artifact_sha256=ONE_SHA,
                    verified_by="sha256 over immutable held-out artifact",
                ),
            ),
        )

    report = ValidationRunner(registry(empirical), {empirical.case_id: evaluator}).run(
        evaluation_id="with-evidence"
    )
    assert report.results[0].status is GateStatus.PASS
    assert report.gate_vector[-1].status is GateStatus.PASS
    # Earlier unselected promotion gates remain not applicable, so this is not
    # a claim that the complete model passed every gate.
    assert report.overall_status is GateStatus.NOT_APPLICABLE
    assert not report.passes_through(PromotionGate.EMPIRICALLY_VALIDATED)


def test_report_is_hashable_and_has_deterministic_lossless_json_roundtrip():
    benchmark = case("software.roundtrip")
    suite = registry(benchmark)
    runner = ValidationRunner(
        suite, {benchmark.case_id: lambda _: values(error=-1.0)}
    )
    digests = (
        SourceDigest("model", "test-model", ONE_SHA),
        SourceDigest("code", "repository", ZERO_SHA),
    )
    first = runner.run(evaluation_id="immutable-report", source_digests=digests)
    second = ValidationReport.from_json(first.to_json(indent=2), registry=suite)

    assert first == second
    assert hash(first) == hash(second)
    assert first.content_sha256 == second.content_sha256
    assert first.to_json() == second.to_json()
    assert [item.kind for item in first.source_digests] == ["code", "model"]
    assert first.suite_complete
    assert first.omitted_case_ids == ()
    assert first.promotion_ceiling is PromotionGate.SOFTWARE_CORRECT
    assert first.passes_through(PromotionGate.SOFTWARE_CORRECT)
    assert not first.passes_through(PromotionGate.NUMERICALLY_CONVERGED)


def test_report_json_rejects_duplicate_keys_and_nonfinite_values():
    with pytest.raises(ValidationContractError, match="duplicate JSON key"):
        BenchmarkRegistry.from_json('{"registry_id":"a","registry_id":"b"}')
    with pytest.raises(ValidationContractError, match="finite"):
        MetricValue("metric", float("nan"))


def test_passing_result_requires_at_least_one_metric_observation():
    with pytest.raises(
        ValidationContractError,
        match="passing result must contain its registered metric observations",
    ):
        BenchmarkResult(
            case_id="software.empty_pass",
            case_version="1.0.0",
            oracle_class=OracleClass.EXACT_MANUFACTURED,
            promotion_gate=PromotionGate.SOFTWARE_CORRECT,
            hard_gate=True,
            status=GateStatus.PASS,
        )


def test_report_registry_binding_rejects_metadata_digest_and_metric_forgery():
    benchmark = case(
        "software.bound",
        metrics=(metric("first"), metric("second")),
    )
    suite = registry(benchmark)
    report = ValidationRunner(
        suite,
        {
            benchmark.case_id: lambda _: values(
                first=-1.0,
                second=-2.0,
            )
        },
    ).run(evaluation_id="bound-report")
    ValidationReport.from_json(report.to_json(), registry=suite)

    wrong_digest = report.to_dict()
    wrong_digest["registry_sha256"] = ZERO_SHA
    with pytest.raises(
        ValidationContractError, match="registry_sha256 does not match"
    ):
        ValidationReport.from_json(json.dumps(wrong_digest), registry=suite)

    wrong_metadata = report.to_dict()
    wrong_metadata["results"][0]["case_version"] = "9.9.9"
    with pytest.raises(
        ValidationContractError, match="metadata does not match registry case"
    ):
        ValidationReport.from_json(json.dumps(wrong_metadata), registry=suite)

    missing_metric = report.to_dict()
    missing_metric["results"][0]["observations"] = missing_metric["results"][0][
        "observations"
    ][:1]
    structurally_valid = ValidationReport.from_json(json.dumps(missing_metric))
    assert structurally_valid.results[0].status is GateStatus.PASS
    with pytest.raises(
        ValidationContractError, match="does not contain every registered metric"
    ):
        ValidationReport.from_json(json.dumps(missing_metric), registry=suite)

    forged_observation = report.to_dict()
    forged_observation["results"][0]["observations"][0][
        "acceptance"
    ] = "value accepted without registry comparison"
    with pytest.raises(
        ValidationContractError, match="observation does not match registry metric"
    ):
        ValidationReport.from_json(json.dumps(forged_observation), registry=suite)


def test_default_registry_fixture_hashes_match_the_checked_in_files():
    suite = load_benchmark_registry()
    repository_root = Path(__file__).resolve().parents[1]
    for benchmark in suite.cases:
        if benchmark.input_sha256 is None:
            continue
        fixture = repository_root / benchmark.fixture_uri
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        assert digest == benchmark.input_sha256


def test_not_applicable_never_counts_as_pass():
    benchmark = case("optional.case")
    report = ValidationRunner(
        registry(benchmark),
        {
            benchmark.case_id: lambda _: EvaluatorResult.not_applicable(
                "optional backend is not installed"
            )
        },
    ).run(evaluation_id="optional-run")

    assert report.results[0].status is GateStatus.NOT_APPLICABLE
    assert report.gate_vector[0].status is GateStatus.NOT_APPLICABLE
    assert report.overall_status is GateStatus.NOT_APPLICABLE
    assert not report.passes_through(PromotionGate.SOFTWARE_CORRECT)
