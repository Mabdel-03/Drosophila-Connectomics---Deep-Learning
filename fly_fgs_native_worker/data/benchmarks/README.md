# Scientific benchmark registry

`registry.v1.json` is the declarative source of truth for simulator acceptance
claims. Each metric has an explicit oracle class, promotion gate, dependency
set, comparison rule, and tolerance authority. Evaluators provide measurements
and evidence receipts; they do not choose their own pass status.

The registry deliberately separates manufactured, invariant, differential,
frozen-regression, and held-out empirical evidence. A missing evaluator or a
missing held-out artifact is reported as `blocked`, never `pass`.

Changes to a case, threshold, fixture, evidence requirement, or dependency
require a new case or registry semantic version. Existing validation reports
remain content-addressable against the older registry hash.

Registry v1.18.0 contains 29 cases: 6 fast, 12 pull-request, 8
scheduled-worker, and 3 held-out promotion cases. In addition to the retained
frozen/open-loop gates, it registers:

- exact incremental fly-FGS parity and fresh-process circuit checkpoint re-entry;
- the 5 ms circuit → 0.5 ms streaming bridge → 0.1 ms muscle/hinge causal contract;
- independent-process streaming checkpoint re-entry with exact source receipts;
- true force-stage `SILENCE`/`SCALE` semantics and complete event dispositions;
- both exhaustive raw-application-lane→physical-wing hypotheses;
- canonical body-feedback orchestration and full-stack checkpoint continuation; and
- the paired 100 ms live fly-FGS→native-FlyBody baseline/raw-L-iv2-silence gate.

Seven of eight scheduled cases have executable evaluators; official
released-policy convergence remains blocked. The three held-out hinge,
indirect-power-muscle, and DNg02 cases also remain blocked. Final
suite-complete evaluation `aws-worker-20260718-v12-canonical-online` observed
25 pass, 4 blocked, and 0 fail. Its report SHA-256 is
`212ea6ac1c7c8f48796df535a96f524bc3e479e1a5149a2b38a17a7459ffb4e6`;
the exact registry-file and canonical registry SHA-256 values are
`880dcaf8b79f1b9869e8c7df8266ed7260cdb1cdf3fe24cfe5e0b53c96c04b1d`
and `a6d8ba583a6b39388960ded150da01988bd00580341540aae2b0071349aef795`;
and the immutable worker-image ID is
`sha256:e925ecd09b736faeee2d5f49d30a5f527d5952d29a5be49cd6574eef3233e66b`.
The four blockers cap the result at `software_correct`; these counts do not
establish calibration, biological validation, or stable flight. A future
release must bind fresh receipts from its own frozen source tree and worker
run rather than reusing these v12 values.

The fly-FGS frozen gate is intentionally narrower than a reproducibility gate.
It checks content hashes, the 48-step pre-roll, 100-sample fixed clock,
1,684-cell inventory, nonzero T4a/NOD1 response, four-NOD1 trace digest,
full-state projections, and the forbidden-downstream-field boundary. Actual
Node reproducibility requires two independent executions of the pinned engine
and bundle. The scheduled FlyBody case accepts only the four exact NOD1
voltages as motor input; the 1,441-point retinal grid and full circuit state are
visualization/audit-only.

The incremental runtime and checkpoint cases separately execute the exact
registered Node engine/bundle. The streaming cases bind exact availability,
piecewise phase crossing, no-retroactivity, event ledgers, source receipts, and
fresh-object/process continuation. The native online case is frozen by
`scenarios/canonical-online-native-pair.v1.json`, which in turn binds
`scenarios/canonical-online-raw-l-iv2-silence.v1.json`. Baseline and perturbation
share seed 73, a 100 ms moving figure, explicit raw-L→physical-left hypothesis,
and all non-intervention inputs. The perturbation gates effective raw-L iv2
force and suppresses target NMJ excitation only on `[60, 100) ms`; it does not
rewrite natural muscle state. This is software causality, not calibrated or
stable flight.

The `protocols/` directory contains four byte-pinned promotion protocols. A
registry tolerance with authority `preregistered_protocol` must carry the
exact protocol SHA-256; validation fails if the local regular file is missing,
symlinked, changed while read, or digest-mismatched. Protocol admission does
not admit external data or make an empirical case pass.

## Calibration evidence contract

`fly_sensor2behavior.calibration.CalibrationEvidenceContract` is the P0
ingress contract for future calibration and held-out datasets. It records
license disposition, source and file receipts (exact bytes plus SHA-256), a
trial inventory, permanent individual/session-grouped calibration, validation,
and held-out partitions, preprocessing and metric protocols, and fit/posterior
provenance. Files must verify on disk before use; missing, tampered, or
size-mismatched artifacts and any unknown, duplicate, unassigned, or leaking
trial fail closed.

The contract contains no biological dataset or expected digest. Adding a valid
contract does not modify registry v1.18.0, satisfy a required evidence receipt,
or unblock a benchmark. A gate remains blocked until its separately reviewed,
licensed artifacts, permanent split, evaluator, and registry change are
approved through the promotion workflow.
