# Evaluation and iteration loop

The evaluation framework is the authority for simulator acceptance claims. It
separates software correctness from biological accuracy and uses four outcomes:
`pass`, `fail`, `blocked`, and `not_applicable`. Missing data, missing evaluators,
unpinned held-out evidence, and unavailable worker dependencies are `blocked`;
they are never converted to a pass.

The declarative registry is
[`data/benchmarks/registry.v1.json`](../data/benchmarks/registry.v1.json). The
runtime contracts and decision logic are in
[`validation.py`](../src/fly_sensor2behavior/validation.py). Evaluators measure
registered metrics; only the registry owns comparators, tolerances, evidence
requirements, prerequisites, and promotion gates.

The source tree declares registry v1.18.0 with 29 cases: 6 fast, 12
pull-request, 8 scheduled-worker, and 3 promotion cases. In addition to the
earlier contracts, vision, structural, compatibility, convergence, and
open-loop vertical slices, it registers exact incremental fly-FGS parity,
circuit and streaming checkpoint re-entry, a 5 ms→0.5 ms→0.1 ms online causal
handoff, true force-stage interventions, explicit unresolved effector
hypotheses, canonical body-feedback orchestration, and a paired native online
fly-FGS→FlyBody gate. A report or release manifest must bind the exact registry
file and canonical content digests from the same frozen source tree; a digest
from another revision is not interchangeable.

The final suite-complete worker evaluation
`aws-worker-20260718-v12-canonical-online` observed 25
software/numerical/structural passes, 4 explicit blockers, and 0 failures. Its
report SHA-256 is
`212ea6ac1c7c8f48796df535a96f524bc3e479e1a5149a2b38a17a7459ffb4e6`;
the exact registry-file and canonical registry SHA-256 values are
`880dcaf8b79f1b9869e8c7df8266ed7260cdb1cdf3fe24cfe5e0b53c96c04b1d`
and `a6d8ba583a6b39388960ded150da01988bd00580341540aae2b0071349aef795`;
and the immutable worker-image ID is
`sha256:e925ecd09b736faeee2d5f49d30a5f527d5952d29a5be49cd6574eef3233e66b`.
The audited 13-episode web manifest has SHA-256
`f9290ac9bd81d57044ec3d96ae40fdc12e020affa351a8c7087a836e9ede2902`.
The four blockers are the absent immutable released FlyBody
policy/WPG/normalization/inference bundle, held-out steering/hinge trials,
held-out indirect-power-muscle trials, and held-out DNg02 trials. The report's
promotion ceiling is `software_correct`; it is not evidence of calibrated,
validated, or stable flight. Historical v10/v11 reports and catalog paths
remain regression evidence only.

Every mathematically nonnegative error or convergence metric uses an inclusive
lower bound of zero as well as its registered upper tolerance. A malformed
negative error therefore cannot pass. Report validation recomputes every
comparator outcome from the registry, binds evidence and prerequisites, and
requires selected, omitted, and per-gate omission sets to form the exact
registry partition.

## Promotion gates

Gates are ordered and non-fungible:

1. `software_correct`
2. `numerically_converged`
3. `structurally_supported`
4. `calibrated`
5. `empirically_validated`

A passing manufactured solution can support software or numerical correctness,
but it cannot satisfy a held-out empirical gate. A model's promotion ceiling is
the highest contiguous passing gate. A targeted report that omits cases is useful
for development, but `suite_complete` is false and its promotion ceiling is
necessarily null; it cannot make a full release-promotion decision.

Use the registry's runtime tiers as the execution budget: `fast` and
`pull_request` cases run on the dependency-light host, `scheduled` cases run on
the pinned FlyBody worker, and `promotion` cases run only with preregistered
held-out evidence. Dependency selection reduces latency without changing a
case's metric or tolerance.

## Registered suite

| Case | Oracle | Gate | What it checks | Expected availability |
|---|---|---|---|---|
| `contracts.json_roundtrip` | Exact/manufactured | Software | Canonical JSON survives a lossless round trip with the same digest | Local/fast |
| `timing.no_future_reads` | Invariant/metamorphic | Software | No sample is consumed before its availability timestamp | Local/PR |
| `vision.causal_golden_suite` | Invariant/metamorphic | Software | Uniform/flicker nulls, direction mirror, phase periodicity, continuous onset, causal availability | Local/fast |
| `pipeline.retinal_to_body_vertical_slice` | Invariant/metamorphic | Software | Analytic retinal frames traverse reduced NOD1, DN/VNC, named muscles, hinge, and body mechanics | Local/PR |
| `pipeline.retinal_to_flybody_vertical_slice` v2.2.0 | Invariant/metamorphic | Software | Neutral, preferred-direction/repeat, and typed MN-b3-left-silence/repeat arms share one resettable FlyBody adapter; exact reset/timebase/repeat invariants, exact upstream and non-target causal-cut locality, target-event availability/removal, visual steering removal, and downstream target-muscle/wing/body propagation are enforced | Pinned worker/scheduled; 20 ms actuation test, not stable flight or a biological silencing-effect claim |
| `feedback.reduced_closed_loop_yaw` | Invariant/metamorphic | Software | Manufactured yaw feedback is causal, stabilizing, mirror symmetric, and silent for a uniform scene | Local/PR; reduced model only |
| `nod1.hines_dense_manufactured` | Exact/manufactured | Numerical | Tree Hines solve matches the equivalent dense solve | Local/fast |
| `physics.reduced_timestep_convergence` | Differential | Numerical | Reduced-order 0.1 ms versus 0.05 ms impulse/body-state drift are each at most 2% | Local/PR; software backend only |
| `nod1.browser_python_parity` | Frozen regression | Numerical | A real Chromium Web Worker and local Python replay agree under exact source, app-asset, circuit, and event-inventory pins | Registered compressed fixture; local/PR |
| `pipeline.registered_nod1_to_flybody_vertical_slice` v1.2.0 | Invariant/metamorphic | Software | The historical 100-sample/four-readout output of the frozen 1,208-cell Chromium circuit runs for 0.5 s through DN/VNC, named wing MNs and muscles, the virtual hinge, and native FlyBody; an exact repeat and typed `MN-iv1-right` cut enforce reset/timebase identity, upstream and non-target locality, causal event availability, and physics-rate torque/fluid/wing/body propagation | Pinned worker/scheduled; retained regression, not the canonical upstream source; no contact-bearing transition may end at or before 50 ms |
| `fly_fgs.fixed_step_integrity` v1.0.0 | Frozen regression | Software | Exact source receipts, 1,684-cell inventory, 48-step pre-roll, 100-sample fixed timebase, four-NOD1 trace digest, full-state projections, nonzero T4a/NOD1 response, and exclusion of source downstream mechanics remain locked | Local/fast; validates one frozen fixture, not an independent Node/browser rerun or biological model |
| `fly_fgs.incremental_runtime_parity` v1.0.0 | Frozen regression | Software | A fresh Node process incrementally reproduces every registered value across 100 samples: 1,684-cell voltage/activity, 1,441 retinal inputs, pooled traces, clocks, receipts, and four NOD1 channels | Local/PR; deterministic execution, not physiology |
| `fly_fgs.checkpoint_reentry` v1.0.0 | Invariant/metamorphic | Software | A hash-protected 250 ms circuit checkpoint restores in a fresh process, rejects tampering before mutation, and reproduces checkpoint/tail/final hidden state exactly | Local/PR; circuit component only |
| `pipeline.registered_fly_fgs_to_flybody_vertical_slice` v1.0.0 | Invariant/metamorphic | Software | The canonical fly-FGS capture exposes only four exact NOD1 voltages to the evidence-qualified bridge and runs the complete 0.5 s episode twice through named wing MNs/muscles, the virtual hinge, and native FlyBody | Pinned worker/scheduled; requires finite deterministic nonzero motor, muscle, torque, fluid-force, and measured-wing propagation, not stable or validated flight |
| `flybody.adapter_analytic_smoke` | Frozen regression | Numerical | The pinned adapter matches the checked-in analytic-fallback compatibility fixture | Pinned worker/scheduled; not released policy |
| `flybody.checkpoint_reentry` | Invariant/metamorphic | Software | Native MuJoCo state and adapter-owned continuity state restore into a new compiled object and reproduce the tail exactly while corrupt/cross-config checkpoints fail closed | Pinned worker/scheduled; adapter state only |
| `physics.flybody_open_loop_convergence` | Differential | Numerical | A fixed 0.2 ms held command program excites aerodynamics while 0.1/0.05/0.025 ms refinements satisfy drift and monotonicity limits | Pinned worker/scheduled; prescribed open loop only |
| `physics.timestep_convergence` | Differential | Numerical | Released-policy steady-flight and saccade runs satisfy the 0.1/0.05 ms impulse/body-state limits | Blocked: released policy/WPG/normalization/inference bundle absent |
| `evidence.cross_atlas_integrity` | Frozen regression | Structural | No direct cross-atlas neuron-ID joins; explicit type crosswalks exist | Local/PR |
| `evidence.banc_fanc_wing_pathway` | Frozen regression | Structural | Content-addressed female BANC v888/FANC v840 snapshots preserve structural pathway counts, source receipts, and atlas separation without promoting premotor candidates | Local/PR; structural evidence only |
| `hinge.heldout_wing_prediction` v2.0.0 | Held-out empirical | Calibration | At target wingbeat onset, predict left-wing kinematics from only completed beats `k-9..k-1`; compare on acquisition-date-held-out Melis data against a matched causal CNN retraining | Source and protocol located; blocked pending exact local intake/topology/split receipts, availability audit, frozen models/predictions, evaluator, and sealed execution |
| `power_muscle.heldout_calcium_flight_state` v1.1.0 | Held-out empirical | Calibration | Power-muscle firing, calcium, wingbeat-frequency, and narrow splay-state relationships match animal-held-out observations | Data/code archives and protocol located; blocked because complete per-animal calcium/ROI lineage and reconciled cohort evidence are unavailable, and no approved intake/evaluator has run |
| `dng02.heldout_wingbeat_amplitude` v3.0.0 | Held-out empirical | Empirical | Separately test open-loop and closed-loop DNg02 activation direction and targeted-pair population dependence on driver-line-held-out trials | Source and protocol located; blocked pending exact intake, permanent line split, sealed evaluator, and passing hinge/power prerequisites |
| `streaming.causal_neuromuscular_runtime` v1.0.0 | Invariant/metamorphic | Software | Manufactured NOD1 drive traverses the 0.5 ms bridge and 0.1 ms mechanics with strict receipts, exact piecewise phase crossing, no same-interval future effect, exact NMJ availability, and checkpoint continuation | Local/PR; virtual mechanics and unfit gains only |
| `streaming.fresh_process_checkpoint_reentry` v1.0.0 | Invariant/metamorphic | Software | Independent Python processes execute baseline, prefix, and resume; exact tail/final checkpoints, transition/source receipts, poison guards, and corruption/source-mismatch rejection are required | Local/PR; manufactured mechanics only |
| `muscle.force_stage_intervention_contract` v1.0.0 | Invariant/metamorphic | Software | Exact half-open 0.1 ms `SILENCE`/`SCALE` contracts gate effective force while preserving natural state, suppressing only in-window target excitation, retaining non-target invariance, and binding schedule/ledger checkpoints | Local/fast; no physiological force claim |
| `effector.raw_lane_physical_hypotheses` v1.0.0 | Invariant/metamorphic | Software | Both exhaustive raw-L/R→physical-wing hypotheses map every bilateral field and six-axis triplet with mirror/involution, receipt, immutability, and checkpoint guards | Local/fast; anatomy remains unknown and signed yaw/roll claims prohibited |
| `feedback.canonical_fly_fgs_closed_loop` v1.0.0 | Invariant/metamorphic | Software | Live Node circuit, streaming bridge/mechanics, explicit effector hypothesis, and checkpointable manufactured physics execute on 5/0.5/0.1 ms clocks with current body state fed back at circuit boundaries and exact fresh-stack restore | Local/PR; manufactured physics, not FlyBody validation |
| `pipeline.canonical_online_fly_fgs_to_flybody` v1.0.0 | Invariant/metamorphic | Software | A 100 ms moving-figure baseline and identical-seed raw-L iv2 force-stage `SILENCE` arm run through live fly-FGS, causal streaming, six-axis mapping, and native FlyBody. Exact clocks/receipts, causal events, pre-onset identity, unique event dispositions, target suppression, command→actuator identity, nonzero native telemetry/divergence, no contact, and fresh-stack checkpoint continuation are required | Pinned worker/scheduled; exploratory short airborne integration, not stable flight, calibrated behavior, or resolved laterality |

Twenty-six cases are self-supervised in the engineering sense: their oracles
come from a manufactured solution, invariant/metamorphic relation,
differential comparison, immutable fixture, or exact content/checkpoint
contract rather than a fitted biological target. The three empirical cases are
deliberately not self-supervised; biological accuracy requires independent
held-out observations.

Eighteen cases run in the dependency-light fast/pull-request tiers. Eight are
scheduled for the pinned worker; seven have executable compatibility,
causality, convergence, checkpoint, or native-online evaluators, while the
official released-policy timestep case remains blocked even on that worker.
The three promotion cases also remain blocked because exact admitted held-out
datasets, frozen predictions, and sealed evaluator receipts are incomplete.
The final suite-complete report asserts the observed 25-pass, 4-blocked,
0-fail partition. These are software-evaluation facts, not evidence that the
flight model is biologically accurate.

Retinal-to-FlyBody v2.2.0 runs neutral, preferred-direction/repeat, and a
full-episode typed `MN-b3-left` silence/repeat under identical preferred-
direction retinal records, seed, clocks, and reset state. The evaluator requires
retinal, reduced-circuit, and DN records to remain exactly unchanged; requires
all non-target motor-event inventories and intrinsic muscle activation, force,
and phase memory to remain exact; verifies that the baseline target event enters
the runtime no earlier than its declared availability and that silence removes
the target rate/event; and then requires nonzero target activation, desired-wing,
measured-wing, and body-state differences. Non-target cumulative work is not a
locality invariant because unchanged force can do different work against the
changed shared-wing velocity. These are manufactured software-causality oracles
only. A dependency-light fake-adapter test verifies the evaluator contract, but
the registered case still requires the pinned worker and is not satisfied by
that fake-adapter test.

Historical Registered-Chromium-NOD1-to-FlyBody v1.2.0 consumes fixture
`a76794e630533822468971cbdbe2164a3d1ddce8a8a38811c9b9c1ad5b97a8a4`:
100 samples at 5 ms, four exact voltage readouts, 1,208 cells, 5,188 edges, and
53,715 prepared circuit events. It preserves the fixture's complete half-open
0.5 s duration through 5,000 native 0.1 ms physics steps, including the latency
tail at the episode boundary. Baseline, exact repeat, and a typed
`MN-iv1-right` silence share the same reset and clocks. The cut must leave source,
DN, non-target events, and intrinsic non-target muscle state exact while changing
the target muscle, desired wing, physics-rate actuator torque, root-total fluid
force, physics-rate measured wing state, and pre-contact body state. No
contact-bearing transition may end at or before 50 ms. Later ground contact is
permitted but disclosed, so this remains a deterministic causal software result,
not continuous airborne flight, effect-size validation, or biological accuracy.

Canonical fly-FGS intake consumes compressed fixture
`5612cc9c2218a917bf2ad939d6aaa2a8402db5f11fd2de36cd3168d9198cbd83`
and source manifest
`fef59e4b8abe5216081b6d64decd0d1775454d06e7631bb9af8ae409c1cd05c3`.
After 48 static 5 ms pre-roll steps, the fixture records 100 half-open samples
from 1,684 cells: 1,457 T4a, 219 LLPC1, two vCH, two DCH, and four NOD1.
Exactly 1,441 T4a cells receive registered retinotopic normalized luminance.
There is no T5, photoreceptor, or lamina model. Sample 0 is already the
post-pre-roll state, so the exported causal source intervals yield 99
`RetinalFrame`s. The full voltage/activity state, retinotopic grid, pooled
traces, and topology are visualization/audit records only. Only four exact
NOD1 voltages form `CircuitOutputTrace` motor input, and raw application `L/R`
is not anatomical laterality.

`fly_fgs.fixed_step_integrity` recomputes the registered inventory, clock,
trace digest, projections, nonzero T4a/NOD1 excursions, and forbidden-field
count from those exact bytes. It deliberately does **not** claim a second
execution. Actual runtime reproducibility requires executing
`capture_fly_fgs_fixed_step.mjs` twice in Node against the pinned assets and
comparing outputs. `pipeline.registered_fly_fgs_to_flybody_vertical_slice`
then requires a paired reset/repeat on the native worker and nonzero motor
events, muscle activation, actuator torque, root-fluid force, and measured
wing motion. Neither gate validates the upstream physiology or downstream
flight behavior.

The newer online stack separates five contracts so a regression is localized
before native physics is interpreted:

1. `fly_fgs.incremental_runtime_parity` proves exact live Node equivalence to
   the frozen capture, and `fly_fgs.checkpoint_reentry` proves fresh-process
   circuit continuation and pre-mutation tamper rejection.
2. `streaming.causal_neuromuscular_runtime` proves the exact 5 ms→0.5 ms→0.1 ms
   causal handoff, six-point phase-crossing interpolation, event availability,
   and no retroactive mechanics. `streaming.fresh_process_checkpoint_reentry`
   repeats the manufactured stack across independent operating-system
   processes and binds runtime/source receipts.
3. `muscle.force_stage_intervention_contract` proves half-open force-stage
   semantics and the applied/suppressed event ledger. The natural hidden state
   remains continuous while effective force is gated.
4. `effector.raw_lane_physical_hypotheses` proves both exhaustive raw-lane
   mappings and their six-axis mirror/involution behavior without selecting an
   anatomical truth. `feedback.canonical_fly_fgs_closed_loop` then proves body
   feedback and full composite re-entry with checkpointable manufactured
   physics.
5. `pipeline.canonical_online_fly_fgs_to_flybody` replaces manufactured physics
   with a newly compiled native FlyBody adapter. The registered contract in
   `data/benchmarks/scenarios/canonical-online-native-pair.v1.json` runs a
   moving figure for 100 ms at seed 73. Its second arm applies the separately
   hashed raw-L iv2 `SILENCE` fixture on `[60, 100) ms`. The arms must match
   numerically and by generated event ID before onset; every delivered event
   must then receive exactly one terminal applied/suppressed disposition with
   target suppression if and only if its availability lies in that interval.
   Target effective force must be zero, while measured wing and final root
   state diverge. Post-onset circuit rates/events may diverge through body
   feedback. The native gate also requires command→actuator identity,
   articulated telemetry, root-fluid force, a no-contact window, and exact
   50 ms fresh-stack continuation.

The native pair is a short, deliberately non-dead integration test. It does not
show a stable trim condition, reproduce the official ordinary-flight policy,
calibrate any neural/muscle/hinge parameter, resolve raw application laterality,
or license a signed yaw/roll conclusion.

The unit/integration suite provides finer localization beneath those registered
claims:

| Boundary | Self-checks |
|---|---|
| Identity/contracts | String-safe FlyWire IDs, materialization qualification, soma-x laterality, SI/radiometric units, direct cross-atlas join rejection, mechanics-owner exclusivity |
| Analytic vision | Uniform/flicker null response, preferred/null equal-and-opposite response, continuous signed speed sweep, `0`/`2π` phase equivalence, continuous figure onset, seeded random-texture reproducibility |
| NOD1/bridge timing | Arbitrary-order Hines parity, invalid-tree rejection, zero-order hold, exact update counts, combined delay, no future reads |
| Signal semantics | Exact events remain exact, seeded streams repeat for one seed and differ for another, synthetic events remain inferred/synthetic, unsupported edits to exact events are rejected |
| Causal anatomy | Contralateral NOD1→DNp26, ipsilateral MN pattern, side-local perturbations, no atlas-ID transport, structural-count magnitude independent of functional gain |
| Female downstream evidence | Exact BANC/FANC source receipts, 487 NOD1→DNp26 synapses, corrected 129-synapse BANC DNp26→wing-MN total, 62 proofread BANC wing MNs, FANC matrix totals, zero direct joins, zero promoted premotor candidates |
| Muscles/mechanics | Named individual force/phase/work, six-axis hinge torque, one external step per physics tick, no double aerodynamics, nonzero unsupported wind rejected |
| FlyBody telemetry | Desired hinge versus authoritative physics-rate measured six-axis joints, actuator torque and root-total fluid wrench, exact compiled root initialization, exact pre-integration transition-contact counts, logging projections, whole-fly COM separate from root/thorax, per-wing decomposition never inferred |
| Canonical fly-FGS boundary | Source/asset hashes, 48-step pre-roll, exact 5 ms/100-sample clock, 1,684-cell inventory, 1,441-point retinal grid, 99 causal frames, four-only NOD1 motor projection, full-state read-only scope, forbidden downstream fields, raw-side non-anatomical disclosure |
| Online causal runtime | Exact 5/0.5/0.1 ms clocks, body feedback at circuit boundaries, six-endpoint phase crossings, generated/held rate provenance, unique event dispositions, no future/retroactive effects, force-stage intervention locality, explicit laterality hypothesis |
| Checkpoint/re-entry | Circuit-only, streaming, manufactured full-stack, and native full-stack fresh-object/process continuation; hash/source/config/schedule/ledger corruption rejected before continuation |
| Canonical artifacts/UI | Schema v1.1 content-bound run identity, registered axes, chunk/table/checkpoint hashes, reset-boundary telemetry, digest-bound online circuit replay, strict replay projection, publisher idempotence/tamper rejection, static release audit |

These tests are necessary regression guards. They demonstrate internal
consistency under their assumptions, not that those assumptions match a fly.
The key visual invariants and retinal-to-body slice are also registered hard
software gates. Their promotion scope remains software correctness; they cannot
promote the surrogate to calibrated or empirically validated.

## Run the loop

Install the dependency-light development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run all unit and integration tests before interpreting a benchmark report:

```bash
python -m pytest -q
```

Exercise the retained open-loop software modes when their dependencies change:

```bash
# Complete analytic image-sample-to-body path. All visual/neural gains are surrogates.
fly-s2b simulate-retinal \
  --duration-s 0.1 \
  --output artifacts/retinal-change-001

# Audited legacy circuit-output path. The input JSON must contain time and
# individual NOD1 voltage channels; its deprecated steering mean is ignored.
fly-s2b simulate-nod1 \
  --input path/to/nod1-result.json \
  --output artifacts/nod1-change-001

# Canonical registered fly-FGS circuit output. The default fixture contains
# the exact scene/circuit capture; its source toy mechanics are excluded.
fly-s2b simulate-fly-fgs-fixture \
  --output artifacts/fly-fgs-change-001
```

On the pinned worker, repeat an affected pipeline with the external backend
selected explicitly. The default operational physics step is `0.1 ms`. A 0.1 m
spawn is contact-free at reset; every claim about a later contact-free interval
must be established from the physics-rate transition trace rather than inferred
from spawn height or decimated frames:

```bash
fly-s2b simulate-retinal \
  --physics-backend flybody \
  --flybody-spawn-height-m 0.1 \
  --physics-dt-s 0.0001 \
  --duration-s 0.02 \
  --output artifacts/retinal-flybody-change-001
```

Do not substitute the reduced backend for a FlyBody case. Conversely, a FlyBody
execution pass establishes only the declared software or numerical claim; the
retina, NOD1 surrogate, motor timing, muscles, and virtual hinge remain
uncalibrated.

For changes to the preferred integration path, run the registered canonical
online pair in the pinned worker. Keep the scenario ID, seed, moving figure,
duration, effector hypothesis, and every non-intervention input identical:

```bash
fly-s2b simulate-canonical-online \
  --output /tmp/canonical-online-baseline \
  --web-replay-output /tmp/canonical-online-baseline.web.json \
  --scenario-id canonical-online-baseline \
  --label "Canonical online baseline" \
  --duration-s 0.1 --seed 73 \
  --figure-velocity-rad-s 0.5235987755982988 \
  --effector-hypothesis raw_l_to_physical_left

fly-s2b simulate-canonical-online \
  --output /tmp/canonical-online-raw-l-iv2-silence \
  --web-replay-output /tmp/canonical-online-raw-l-iv2-silence.web.json \
  --scenario-id canonical-online-raw-l-iv2-silence \
  --label "Raw-app-L iv2 silence" \
  --duration-s 0.1 --seed 73 \
  --figure-velocity-rad-s 0.5235987755982988 \
  --effector-hypothesis raw_l_to_physical_left \
  --muscle-interventions \
    data/benchmarks/scenarios/canonical-online-raw-l-iv2-silence.v1.json
```

The exact pair and intervention contracts are
`data/benchmarks/scenarios/canonical-online-native-pair.v1.json` and
`data/benchmarks/scenarios/canonical-online-raw-l-iv2-silence.v1.json`.
The first file is the evaluator contract; the second is accepted directly by
the CLI. Full circuit state is included by default for the browser workspace;
`--omit-full-cell-state` is appropriate for evaluator-only runs and deliberately
removes that rich display attachment without changing the four motor channels.

Run the registered evaluators and write a content-addressable JSON report:

```bash
fly-s2b validate --output artifacts/validation/report.json
```

Select an explicit case, or select the prerequisite/dependent closure affected
by a component change:

```bash
fly-s2b validate \
  --case nod1.hines_dense_manufactured \
  --output artifacts/validation/nod1-hines.json

fly-s2b validate \
  --dependency visual.nod1.hines \
  --output artifacts/validation/nod1-change.json
```

Use a new output path for every approved run. Read the command's JSON summary
and the report's per-case `reason`; an overall `blocked` result is an honest
statement about unavailable evidence, not a successful full-suite validation.
The CLI returns a non-zero status for registered metric failures or evaluator
errors. See `fly-s2b validate --help` for the exact selection and evaluation-ID
options supported by the installed revision.

Attach that unchanged native report to a newly generated static replay:

```bash
fly-s2b export-web \
  --output web/public/data \
  --validation-report artifacts/validation/report.json \
  --overwrite
```

The exporter binds the report to the exact registry contract before writing any
output, copies the unchanged report and registry bytes, and records their exact
file SHA-256 values plus the canonical registry digest in the web manifest. The
client hashes those bytes again and validates report coverage, comparator
outcomes, evidence, prerequisites, and per-gate omissions before displaying a
claim. Web Crypto is preferred; a self-tested pure-TypeScript SHA-256 fallback
keeps the same integrity checks on the plain-HTTP AWS IP route. Per-muscle replay
values are decimated from actual reduced-model arrays. The standard scenario
bundle still labels neural origin unavailable and its circuit animation
illustrative because it does not ingest a neural trace.

Canonical online runs use the stricter publication path after both artifacts
and their bound replays have completed:

```bash
python scripts/publish_canonical_web_run.py \
  --artifact-dir /tmp/canonical-online-baseline \
  --replay /tmp/canonical-online-baseline.web.json \
  --episode-id canonical-online-baseline \
  --condition "moving figure; baseline" \
  --description "Live fly-FGS circuit through the causal actuator and native FlyBody." \
  --color '#2dd4bf' \
  --position 0

cd web
npm ci
npm run typecheck
npm run build
cd ..
python scripts/audit_web_release.py
```

The publisher validates schema-v1.1 artifact/replay identity and installs the
immutable run before making the public index visible. It is idempotent for the
same bytes and rejects a changed episode unless replacement is explicit. It
does not attach a validation decision or promote `exploratory` output. The
release auditor independently rechecks the static artifact, replay, registry,
and report bindings.

For the worker-only cases, build the pinned image and mount a persistent FlyGym
asset cache:

```bash
docker build -t fly-s2b-worker .
docker run --rm \
  -v flygym-assets:/opt/flygym-assets \
  fly-s2b-worker flybody-smoke --mode analytic-wingbeat --duration-s 0.005
```

The smoke proves compatibility, not straight-flight, saccade, neuromuscular, or
biological accuracy. Run heavyweight benchmarks on a dedicated worker; never on
the public display server or in an HTTP request.

The worker separates four claims that are easy to conflate:

- `pipeline.retinal_to_flybody_vertical_slice` v2.2.0 is the reset-controlled
  neutral → moving → repeated-moving → typed MN-b3-left silence → repeated
  silence causal actuation check. It proves exact initial-state/timebase/repeat,
  upstream-cut, non-target event, and intrinsic non-target muscle-state
  invariants while running for 20 ms through the reduced four-readout NOD1
  surrogate, not the frozen 1,208-cell browser circuit. Its nonzero downstream
  differences establish software propagation only, not biological silencing
  magnitude or stable flight;
- `pipeline.registered_nod1_to_flybody_vertical_slice` v1.2.0 consumes the exact
  content-addressed 1,208-cell Chromium fixture for its complete 0.5 s capture.
  Baseline/repeat and a typed `MN-iv1-right` cut must preserve reset, clocks,
  source/DN records, non-target events, and intrinsic non-target muscle state,
  then propagate through physics-rate actuator torque, root-total fluid force,
  measured wing state, and body state. Its registered airborne comparison ends
  at 50 ms; later contact-bearing transitions do not make it stable flight;
- `fly_fgs.fixed_step_integrity` v1.0.0 verifies the exact fixed fixture and
  its read-only projections. It is intentionally distinct from rerunning the
  pinned JavaScript engine and cannot be cited as Node/browser reproducibility;
- `pipeline.registered_fly_fgs_to_flybody_vertical_slice` v1.0.0 consumes only
  the four exact NOD1 voltages from the canonical 1,684-cell capture. A paired
  reset/repeat must propagate nonzero activity into native FlyBody telemetry;
  full circuit/retinal arrays remain visualization/audit-only;
- `flybody.adapter_analytic_smoke` is a v3 frozen compatibility fixture that
  requires exact runtime receipts, compiled-model fingerprint/topology,
  advance-then-hold control semantics, collision-disabled fluid proxies, and
  legs-only contact in addition to its tolerance-bounded numeric summaries;
- `physics.flybody_open_loop_convergence` uses a prescribed `0.2 ms` command
  clock with `0.1/0.05/0.025 ms` physics refinement and makes no stable-flight
  claim;
- `physics.timestep_convergence` requires the released straight-flight and
  saccade policy stack and remains blocked until that immutable stack is
  installed and reviewed.

### Released FlyBody bundle intake boundary

The official data record is Figshare DOI
[`10.25378/janelia.25309105`](https://doi.org/10.25378/janelia.25309105),
version 4. The ordinary flight path requires both official archives:

- file `51196859`, containing
  `datasets_flight-imitation/wing_pattern_fmech.npy` and
  `datasets_flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5`;
- file `44815195`, containing the TensorFlow SavedModel rooted at
  `trained-fly-policies/flight`.

The controller-reuse checkpoint in file `51196886` is not interchangeable with
that policy: its released example uses a nonzero joint filter. The paper uses
one shared controller for straight flight and saccades, a `0.05 ms` physics
clock, a `0.2 ms` control clock, and a 218 Hz WPG with a ±5% frequency range.
No separate observation-normalization file is documented; LayerNorm state is
embedded in the SavedModel, while the canonical wrapper scales actions.

These files are necessary but not sufficient for the registered gate. The
record does not publish archive SHA-256 values or a complete extracted-member
inventory; the paper does not bind an exact source commit; the original
runtime is not lockfile- or container-pinned; the 216/56 trajectory split does
not include exact held-out indices; and no example freezes trajectory, start
step, seed, initial WPG phase, or 100 ms evaluation window. FlyGym 2.1.0 also
ships no policy or WPG and its FlyBody integration is experimental. Therefore
the project must first reproduce a deterministic rollout in a locked original
dm-control runtime and then establish state/action/force equivalence to the
FlyGym adapter.

After staging the archives on a dedicated worker, create a non-authoritative
candidate inventory without downloading or rewriting anything:

```bash
python -m fly_sensor2behavior.flybody_release_intake intake \
  --root /path/to/extracted-official-release \
  --output /path/outside-the-release/candidate.json
```

The intake hashes every member and validates required roles, but it labels the
result `candidate_unreviewed`. A separately reviewed expected inventory,
SavedModel signature/normalization attestation, exact runtime image, permanent
test split, deterministic case manifest, original-runtime golden trace, and
FlyGym-equivalence report are all required before an evaluator may replace the
registered blocker. Run this staging on object-backed worker storage, not the
public web host.

### Held-out neuromuscular evidence boundaries

Registry v1.18.0 retains the content-addressed public sources and evaluation
protocols frozen before
held-out execution. Locating a source is not the same as admitting its bytes to
an evaluation. Each case remains blocked until the exact artifact, source
topology and identity lineage, permanent split, frozen candidate and baseline
predictions, evaluator/runtime, and sealed execution receipts required by that
case are present.

The protocol documents are themselves content-addressed registry evidence:

| Protocol | SHA-256 | Role |
|---|---|---|
| [`melis-wing-hinge-heldout.v2.json`](../data/benchmarks/protocols/melis-wing-hinge-heldout.v2.json) | `8c538c3866c81c349f2b3d7917df0cc0e4c47f50a436fff5b63fe726881dbf45` | Causal date-grouped wing prediction, matched baseline, hierarchical metrics |
| [`asynchronous-power-muscle-heldout.v2.json`](../data/benchmarks/protocols/asynchronous-power-muscle-heldout.v2.json) | `927eacb716bc27573269a5b156bf258be376e6d73ddbd919e2ca6d0082d4dddc` | Animal/recording lineage, cross-sex declaration, physiology endpoints and blockers |
| [`dng02-driver-line-heldout.v2.json`](../data/benchmarks/protocols/dng02-driver-line-heldout.v2.json) | `ef2756ac18d43d9ed2683fe85cff9bde19bf3556e55cfc9a3ddbe1e0e965c410` | Driver-line split and separate open-/closed-loop positive-control endpoints |
| [`sealed-heldout-execution.v1.json`](../data/benchmarks/protocols/sealed-heldout-execution.v1.json) | `7946dfd6b5185c9d2eee39328c2c1eb96ac19ad86d1b00955c0433455b9247ff` | Freeze order, no-test-access log, one-time execution, immutable outputs |

The associated empirical tolerances use the registry's
`preregistered_protocol` authority. This means the comparator and uncertainty
rule were frozen in a hashed protocol; it does not mean the biological result
has passed. Editing a protocol requires a new content hash and registry
revision, and cannot retroactively change an old report.

#### Melis wing-hinge intake

The primary source is the CC0 CaltechDATA record
[`10.22002/aypcy-ck464`](https://doi.org/10.22002/aypcy-ck464), file
`main_muscle_and_wing_data.h5`: 2,642,932,080 bytes with source MD5
`8aa8629237b5cc7f845c8fd23872c815`. The reviewed upstream code is pinned to
commit `cd5081aac460754b8ff3d6426835920e43542f95`. The expected HDF5 topology is
74 sessions from 33 acquisition dates and 377 movies. An eligible sample has a
`9 x 13` input—twelve left steering-muscle fluorescence summaries plus measured
wingbeat frequency—and an 80-coefficient left-wing target.

The registered claim is deliberately causal: a prediction issued at the onset
of wingbeat `k` may use only completed wingbeats `k-9` through `k-1` to predict
wingbeat `k`. It therefore yields `N_wbs - 9` candidate windows per movie before
other exclusions. Same-beat or future features, centered filters, acausal
deconvolution, and the published within-movie first-30 split/leading-window
construction are inadmissible for this claim. Feature availability timestamps,
not ordinal indices alone, must prove that every input was available at issue
time.

Whole acquisition dates are assigned before window construction to permanent
calibration/validation/held-out partitions of 23/5/5 dates. Dates, sessions,
movies, source wingbeats, and derived windows must have zero cross-split
overlap. Because the public artifact does not provide a reviewed
session-to-animal crosswalk, the result is **date-held-out**, never
animal-held-out. The matched reference is a causal retraining of the pinned
Melis CNN on the same samples, not its published acausal score. Candidate and
baseline are compared with one-sided 95% acquisition-date-cluster bootstrap
bounds in reconstructed phase-angle and coefficient space.

[`wing_hinge_intake.py`](../src/fly_sensor2behavior/wing_hinge_intake.py) is a
non-downloading, fail-closed intake boundary for a caller-supplied local HDF5.
It verifies the exact byte count and MD5, records an observed SHA-256, guards
the file identity across lazy HDF5 inspection, validates the complete topology,
constructs the date-grouped split, audits causality and overlap, and emits
canonical JSON receipts. Successful intake is still labelled
`candidate_unreviewed` and cannot set `hinge_gate_unblocked`; independent review,
matched model training, frozen predictions, and sealed evaluation remain
required. The data validate tethered left-wing fluorescence-to-kinematics only,
not bilateral mechanics, force, sclerites, aerodynamics, or free flight.

#### Asynchronous power-muscle evidence

The experimental archives are located at Zenodo
[`10.5281/zenodo.7737730`](https://doi.org/10.5281/zenodo.7737730) under
CC-BY-4.0; the analysis-code archive is
[`10.5281/zenodo.7740678`](https://doi.org/10.5281/zenodo.7740678) under
CC-BY-NC-4.0. Raw recordings, workbooks, event tables, NPY derivatives, images,
and movies from one recording must collapse onto one source-recording and
biological-individual identity before splitting or counting. Treating derived
representations as independent trials is a hard integrity failure. Male
physiology used by the canonical female simulator must be declared and scored
as cross-sex transfer evidence.

The reviewed public inventory does not provide a complete reconciled
per-animal lineage for the calcium cohort: raw movies, ROIs, traces, fits,
exclusions, processing code, and reported animal counts are not all jointly
recoverable. Representative or aggregate calcium traces cannot substitute.
The splay comparison is a narrow challenge, not general muscle-force
validation, and direct force, stretch activation, thoracic compliance, and
power-transfer evidence also remain incomplete. Consequently the protocol is
frozen, but the power-muscle calibration gate is explicitly blocked.

#### DNg02 positive-control evidence

The female tethered-flight source is the CC-BY-4.0 Mendeley record
[`10.17632/7g984jm2zc.1`](https://doi.org/10.17632/7g984jm2zc.1). Its HDF5 trial
data, README, analysis code, units, and axis semantics require exact receipts in
one canonical intake manifest. Source-code reproduction of published summary
slopes is an integrity fixture, not held-out model validation.

Every fly, activation trial, time bin, and derivative for one driver line must
remain in one permanent split. Open-loop striped-drum and closed-loop stripe
trials are fitted, predicted, bootstrapped, and reported separately; pooling
them is a hard failure. Source wingbeat-amplitude values are converted from
degrees to radians before scoring. Targeted-pair count is a driver-line proxy,
not an exact measurement of cells recruited in each fly, and the public layout
does not expose a reviewed globally unique subject identity across all
protocols. The gate requires positive one-sided driver-line-cluster bounds for
activation effect and population slope in each protocol, plus registered curve
non-inferiority. It remains blocked until exact intake, permanent split, frozen
model outputs, and sealed execution exist—and until its hinge and power-muscle
prerequisites pass.

## Change a model safely

1. Identify the replaceable boundary and its dependency keys. Preserve public
   schemas, SI units, clock/availability semantics, identity spaces, and signal
   provenance.
2. Add or update focused unit tests. For a causal change, include a negative or
   metamorphic test (for example, side-local silencing, no-future-read, seed
   repeatability, or zero-input symmetry), not only a golden output.
3. If parameters, evidence, or implementation hashes change, create a new model
   record or version in
   [`data/models/flight_model_registry.json`](../data/models/flight_model_registry.json).
   Do not overwrite a calibrated or approved record in place.
4. Run tests, then `fly-s2b validate --dependency <key> ...`. Inspect every
   selected prerequisite and dependent case, including blocked cases.
5. Compare immutable scientific artifacts and validation reports. Accept a
   change only when unexpected metric drift is explained; never loosen a
   threshold merely to make a run green.
6. Iterate on the model using calibration partitions only. Run held-out cases
   only through the preregistered promotion workflow, and retain every failed
   report.

## Add a benchmark or evidence set

1. Add a case to a new semantic version of the benchmark registry. Declare one
   oracle class, promotion gate, runtime tier, dependency keys, prerequisites,
   metrics, units, comparators, and source-backed tolerance authority.
2. Pin every frozen fixture with SHA-256. For empirical cases, register the
   public/licensed source and split on the coarsest reviewed independent group
   before fitting: biological individual where identity is available,
   acquisition date for the present Melis artifact, or driver line for DNg02.
   Pin both the held-out artifact/split receipt and the preregistered protocol;
   adjacent observations or derivative files are never independent groups.
3. Implement an evaluator that returns only metric values and evidence receipts.
   It must not decide its own pass status or substitute synthetic output for a
   missing observation.
4. Add tests for pass, fail, blocked, malformed input, deterministic replay, and
   dependency selection. Run the complete local suite and the appropriate worker
   tier.
5. Preserve the old registry and reports. Registry changes create a new claim;
   they do not retroactively alter prior results.

## Debugging order

When an end-to-end case changes, localize it in causal order:

```text
contract/identity
  -> availability and clocks
  -> scene / retinal exposure / image-derived motion
  -> 5 ms fly-FGS full-state/control audit or reduced/legacy NOD1 solver/import
  -> four motor-bound NOD1 voltages
  -> 0.5 ms NOD1-to-DNp26 encoder and held causal rates
  -> phase-crossing DNp26-to-MN generation / availability queue
  -> 0.1 ms natural/effective muscle state and event disposition
  -> raw-lane six-axis hinge torque
  -> explicit raw-lane-to-physical-wing hypothesis
  -> selected physics/aerodynamics owner
  -> component/composite checkpoint receipts
  -> schema-v1.1 artifact tables/chunks and replay projection
```

Compare intermediate arrays at the first divergent boundary. Preserve the
original signal kind: exact spikes remain exact, rate-derived events remain
labelled inferred, and seeded synthetic events retain their generator seed.
Structural synapse counts may enable a route, but they must never become gains.

## Current limits

- The analytic-retinal path is a one-dimensional normalized-luminance software
  model with a Reichardt detector and reduced preferred-direction NOD1
  surrogate. It omits calibrated compound-eye optics/radiometry, photon noise,
  photoreceptor physiology, T5/null pathways, and the full T4/T5→vCH/DCH→NOD1
  circuit. It cannot satisfy neural or empirical promotion gates.
- The saved-legacy-result path has the audited circuit voltages but no
  source-of-record retinal frames, so its visual provenance cannot be recovered
  from the result JSON alone.
- Browser/Python NOD1 parity has a registered real-Chromium fixture with strict
  app/source/circuit/event receipts. It establishes numerical cross-runtime
  agreement only; it does not validate anatomy, physiology, laterality, or the
  circuit-to-muscle mapping.
- The canonical fly-FGS fixture improves the upstream scene/circuit boundary,
  but it is still T4a-only. It has no T5, photoreceptor, lamina, calibrated
  compound-eye optics, or physiological latency claim. Its fixed-fixture gate
  is not a runtime-reproducibility result; the separate incremental and
  checkpoint gates now cover exact software execution. Source downstream toy
  mechanics remain excluded.
- The canonical online path closes figure/world azimuth against current body
  yaw at 5 ms boundaries. It does not render calibrated 3-D compound-eye optic
  flow, translation/parallax, occlusion, T5/null pathways, or photoreceptor and
  lamina physiology. Its DN/MN rates and event timing are inferred, its phase
  is model-owned rather than measured from FlyBody, and its power/tension/
  steering/hinge parameters remain uncalibrated.
- Raw fly-FGS `L/R` is not anatomical laterality. Both actuator hypotheses are
  implemented and receipt-bound, but neither is preferred by evidence. Signed
  yaw/roll interpretation is prohibited until this is resolved.
- Both causal pipeline CLIs can select FlyBody. Native episodes record
  authoritative physics-rate actuator torque, measured six-axis wing qpos/qvel,
  root-total fluid wrench, transition-contact counts, and articulated whole-fly
  COM; logging-rate fields are exact indexed projections. Desired virtual-hinge
  kinematics and root/thorax pose remain distinct. A reviewed per-wing
  fluid-force decomposition is unavailable.
- The historical v10 release catalog includes a 100 ms analytic-retinal/reduced-NOD1
  FlyBody replay and a complete 500 ms registered-Chromium-NOD1 FlyBody replay.
  Each catalog entry links its display JSON to an immutable scientific run whose
  manifest records runtime/dependency receipts, compiled-model fingerprint,
  exact clocks, model hashes, and atlas/simulation identity boundaries. Their
  release-specific hashes are generated with the catalog. They are actuation
  demonstrations, not sustained or biologically validated flight.
- The FlyBody path rejects a nonzero wind because this adapter has no reviewed
  ambient-air setter. It also retains articulated legs, the floor, and leg-ground
  contact, unlike the released flight-evaluation topology.
- Released FlyBody straight-flight and saccade policies are not reproduced. The
  policy convergence gate remains blocked independently of the passing/failing
  status of the adapter smoke or prescribed open-loop convergence test.
- A content-addressed female BANC v888/FANC v840 structural fixture is present.
  It locks 487 BANC NOD1→DNp26 synapses, 62 proofread BANC wing MNs, the
  corrected 129-synapse BANC DNp26→wing-MN total, and the published one-sided
  FANC premotor matrix. These counts are topology, not physiological strength.
- A complete BANC/FANC DN→identified-premotor→MN→muscle graph is not present:
  premotor matching was not performed, the crosswalk is explicitly incomplete,
  and no topological candidate was promoted to an identity match.
- Public evidence sources and content-addressed protocols referenced by v1.18 are registered
  for Melis wing-hinge prediction, asynchronous power-muscle physiology, and
  DNg02 positive control. They do not contain admitted held-out receipts or
  frozen simulator predictions, so all three empirical gates remain blocked.
- The Melis intake can establish exact source/topology/split receipts only. It
  is date-held-out rather than animal-held-out, observes left tethered-flight
  kinematics rather than force or free flight, and still needs an availability
  audit, matched causal baseline, candidate fit, and sealed evaluation.
- The power-muscle sources lack complete reconciled per-animal calcium/ROI
  provenance and direct force/stretch/thorax evidence. The DNg02 source lacks a
  durable cross-protocol subject identity and uses targeted driver-line pairs as
  a proxy for recruited cells; its open- and closed-loop trials cannot be
  pooled.
- Hinge, power, steering, tension, signs, moment arms, gains, and uncertainty
  distributions therefore remain unfitted to an approved versioned calibration
  set.
- A manufactured reduced-order yaw-feedback suite and canonical body-coupled
  fly-FGS loop exist. Measured wing phase, haltere, strain/proprioception,
  translation-aware vision, and flight-state feedback are not implemented as a
  validated multisensory loop.
- The replay UI is a static exploratory display. Canonical online episodes can
  show exact sampled retinal/full-circuit state, causal rates/events,
  intervention receipts, mapping, checkpoints, and native telemetry, but they
  cannot run or alter the simulator. Any unattached channel remains explicitly
  unavailable; visual coherence is not validation.
