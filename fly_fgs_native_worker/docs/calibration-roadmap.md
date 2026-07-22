# Calibration and validation roadmap

## Promotion rule

All current outputs are `exploratory; calibrated-by-none`. A model becomes `calibrated` only after a versioned fitting dataset, immutable split definition, parameter posterior, units, likelihood/objective, code hash, and fit diagnostics are recorded. It becomes `validated` only after passing a preregistered held-out gate. Fitting and evaluation trials remain permanently separate. The machine-enforced gate order and current benchmark cases are documented in [evaluation.md](evaluation.md); a `blocked` case is never promotion evidence.

Visual plausibility is never a validation criterion.

## Stage 1: complete the anatomical evidence graph

1. Freeze exact female BANC/FANC releases and licenses, with immutable source hashes.
2. Match FAFB descending-neuron types to the female VNC through evidence-scored cell-type crosswalks. Never join FlyWire roots to BANC/FANC/MANC body IDs.
3. Add explicit `DN → VNC premotor → wing MN → named muscle` edges, side, specimen, release, matching method, structural coverage, confidence, and citations.
4. Reproduce the complete LLPC1 and NOD1 coverage audit from a checked-in raw table. Retain every unresolved output as an uncertainty category.
5. Add DNg02 as an independent positive-control pathway. Preserve MANC v1.0 only as cross-sex/specimen replication, not interchangeable identity.
6. Audit laterality against each atlas convention and validate MN→muscle innervation against the experimental wing-MN atlas.

All FlyWire FAFB inputs remain qualified to materialization 783, with root IDs
serialized as decimal strings and local skeleton/synapse coordinates retained
in nanometres. BANC, FANC, MANC, FlyWire, and simulator identifiers remain
separate even when a type-level crosswalk is proposed.

Current partial milestone: a content-addressed female BANC v888/FANC v840
fixture now locks exact source receipts and structural observations, including
the corrected BANC DNp26→wing-MN total of 129 synapses and the independent FANC
premotor matrix. It does not complete this stage: BANC/FANC premotor matching
was not performed, the crosswalk remains incomplete, and no candidate pair was
promoted to an identity match.

Exit gate: no implicit cross-atlas joins; all simulated muscle paths terminate through an explicit MN and innervation edge; coverage arithmetic is reproducible from locked raw data.

## Stage 2: reproduce upstream FlyBody flight baselines

Acquire and hash the exact released straight-flight and saccade controllers, initial states, model assets, and reference metrics. Run them with the pinned worker at their released timestep and record wing angles, fluid and total wrenches, trajectory, energy, and numerical diagnostics.

The adapter's analytic fallback smoke does not satisfy this stage.

Exit gates:

- released straight-flight and saccade metrics fall within upstream published/test variability;
- halving physics timestep from 0.1 ms to 0.05 ms changes integrated aerodynamic impulse and standardized 100 ms body state by less than 2% for the FlyBody worker, not just the reduced-order scaffold;
- container, assets, controller, model, and metric hashes reproduce on a clean worker.

## Stage 3: assemble calibration datasets

Use trial-level data with synchronized clocks and documented preprocessing:

| Model block | Required observations | Main fitted quantities |
|---|---|---|
| MN signal adapter | Exact MN spikes where available; voltage/rate otherwise; wingbeat phase | Latency, phase distributions, rate-to-event uncertainty |
| DLM/DVM power | MN ensemble spikes, calcium, stretch state, wingbeat frequency, power, splay state | Calcium/stretch time constants, force law, oscillator coupling |
| Steering | Individual b1–b3, i1/i2, iii, hg activity with 3D wing motion | Preferred phase, phase dispersion, sign, gain, delay |
| Tension | ps/tp/tpn activity, thoracic stiffness/resonance, WBF, transmitted power | Stiffness/frequency/power coupling |
| Wing hinge | Muscle state, 3D wing kinematics, hinge/sclerite measurements | Moment arms, nonlinear geometry, compliance, damping |
| Aerodynamics | Wing kinematics, body state, measured forces where available | Quasi-steady coefficients and residual model |

Record individual, sex, preparation, temperature, stimulus, measurement
uncertainty, calibration, units, missingness, and license. Split on the coarsest
reviewed independent group before fitting. Use biological individual where a
durable cross-recording identity exists, whole acquisition date for the present
Melis artifact, and driver line for the DNg02 population-response claim. Never
put adjacent wingbeats, trials, time bins, or derived representations from one
source group into different partitions.

### Current evidence-intake state

Registry v1.18.0 retains the evaluation design frozen before held-out access. These
source discoveries narrow the work; they do not promote a model:

| Block | Located source and frozen protocol | Admissible scope | Remaining blocker |
|---|---|---|---|
| Wing hinge | CC0 CaltechDATA [`10.22002/aypcy-ck464`](https://doi.org/10.22002/aypcy-ck464); content-addressed [`melis-wing-hinge-heldout.v2.json`](../data/benchmarks/protocols/melis-wing-hinge-heldout.v2.json) | Date-held-out, one-beat-ahead left-wing kinematics from completed beats `k-9..k-1` | Exact local HDF5 receipt/topology/split, feature-availability audit, matched causal CNN, candidate fit, frozen predictions, evaluator, sealed execution |
| Asynchronous power | CC-BY experimental archive [`10.5281/zenodo.7737730`](https://doi.org/10.5281/zenodo.7737730), CC-BY-NC code archive [`10.5281/zenodo.7740678`](https://doi.org/10.5281/zenodo.7740678); [`asynchronous-power-muscle-heldout.v2.json`](../data/benchmarks/protocols/asynchronous-power-muscle-heldout.v2.json) | Separate firing-rate, calcium, wingbeat-frequency, and narrow splay-state endpoints | Complete reconciled per-animal calcium movie/ROI/trace/fit/code lineage is absent; raw/derived recording identities and cross-sex transfer remain to audit; force/stretch/thorax evidence is incomplete |
| DNg02 | CC-BY Mendeley [`10.17632/7g984jm2zc.1`](https://doi.org/10.17632/7g984jm2zc.1); [`dng02-driver-line-heldout.v2.json`](../data/benchmarks/protocols/dng02-driver-line-heldout.v2.json) | Driver-line-held-out activation and targeted-pair dependence, separately for open-loop striped-drum and closed-loop stripe trials | Exact source intake, permanent line split, frozen model outputs, sealed evaluator, and passing hinge/power prerequisites |

All three protocols reference the content-addressed
[`sealed-heldout-execution.v1.json`](../data/benchmarks/protocols/sealed-heldout-execution.v1.json).
It freezes the model, preprocessing, split, metric implementation, and evaluator
before one-time test access. An evaluator must report source/split leakage and
fit-reference guards alongside biological errors; a source being public does
not make repeated test-set iteration acceptable.

The Melis source expects 74 sessions across 33 acquisition dates and 377
movies. Its permanent 23/5/5 date split is constructed before windows. Each
prediction for target beat `k` uses the twelve left steering-muscle fluorescence
summaries plus wingbeat frequency from only `k-9..k-1`, never same-beat/future
values or centered/acausal processing. Because no reviewed session-to-animal
crosswalk is published, the claim cannot be called animal-held-out. It also
does not observe bilateral mechanics, muscle force, aerodynamics, or free
flight. A successful non-downloading intake remains `candidate_unreviewed` and
cannot unblock the hinge gate by itself.

## Stage 4: fit a probabilistic neuromuscular actuator

Fit the blocks in causal order while propagating upstream uncertainty:

1. Infer spike/phase posteriors only for signals that are not exact; retain the original signal kind.
2. Fit asynchronous power-muscle calcium and stretch activation over many wingbeats. Do not model one MN spike as one wing contraction.
3. Fit steering activation as a circular, wingbeat-phase-dependent process.
4. Fit tension-muscle effects on thoracic resonance and power transmission.
5. Fit a hierarchical virtual hinge with distributions over signs, moment arms, delays, and gains; share only biologically justified parameters across sides/individuals.
6. Fit aerodynamic residuals after hinge fitting, so aerodynamic coefficients do not absorb neuromuscular errors.

Use structural synapse counts only as anatomical topology or a declared weak prior over edge existence. They must not directly set physiological signs or magnitudes. Store posterior draws and covariance, not only a best fit. Report identifiability, posterior predictive checks, residual structure, and sensitivity to priors.

Exit gates:

- on the permanent Melis date-held-out split, one-beat-ahead steering activity
  predicts left-wing angles no worse than a matched **causal retraining** of the
  pinned Melis CNN under identical samples and preprocessing; the published
  within-movie first-30/acausal score is not an admissible comparator;
- every candidate-to-baseline error ratio and the matched-baseline-to-
  calibration-mean guard passes its one-sided 95% acquisition-date-cluster
  bound, including each reconstructed wing-angle axis and coefficient space;
- held-out indirect-muscle trials reproduce firing rate, calcium,
  wingbeat-frequency, and the narrow splay-state effect with declared
  individual-cluster intervals, complete per-animal provenance, and explicit
  male-to-female transfer status; neither this endpoint nor the present archive
  establishes direct force or stretch-activation accuracy;
- left/right and individual-level residuals show no unmodeled systematic bias
  large enough to dominate the target perturbations once suitable bilateral
  evidence exists.

## Stage 5: open-loop pathway experiments

Feed recorded or exact circuit outputs into the actuator before closing the sensory loop. Standardize airborne initial conditions and compare baseline with vCH/DCH, DN, MN, muscle, and gust interventions. Include DNa04, DNp26, DNg32, and the independent DNg02 positive control.

Current source milestone: registry v1.18.0 makes the registered fly-FGS capture
the canonical upstream visual/circuit source. A 240 ms static pre-roll precedes
100 fixed 5 ms samples from 1,684 cells: 1,457 T4a (1,441 with registered
retinotopic input), 219 LLPC1, two vCH, two DCH, and four NOD1. The fixture has
no T5, photoreceptor, or lamina model. It exports 99 causal `RetinalFrame`s;
full cell state is retained only for visualization/audit, and only the four
exact NOD1 voltages can drive the downstream bridge. Raw source `L/R` labels
are not anatomical laterality. The source page's hand-tuned DN/muscle/wing/yaw/
body mechanics are excluded.

The retained scheduled frozen vertical slice runs the complete 0.5 s capture twice
through the evidence-qualified DN/VNC bridge, named wing MNs and muscles, the
virtual six-axis hinge, and native FlyBody/MuJoCo. It checks reset/timebase
identity, deterministic output, and nonzero propagation into motor events,
muscle activation, actuator torque, fluid force, and measured wing state. The
older 1,208-cell Chromium slice remains a historical numerical/causal
regression with its typed `MN-iv1:right` intervention and 50 ms contact-free
comparison. The analytic-retinal/reduced-NOD1 case remains a manufactured
visual-boundary test. All establish repeatable software actuation only; none
establishes stable flight, calibrates a biological effect size, or satisfies
this stage's empirical exit gate.

Registry v1.18.0 now also freezes the streaming actuator and intervention
semantics used to move beyond that open-loop regression. Exact 5 ms circuit,
0.5 ms bridge, and 0.1 ms force-stage/hinge/physics clocks are independently
tested; complete circuit/bridge/mechanics checkpoints resume in fresh objects
or processes; and both raw-app-lane→physical-wing hypotheses are explicit. The
registered native comparison runs a 100 ms moving-figure baseline and an
identical-seed raw-L iv2 `SILENCE` arm over `[60, 100) ms`. Pre-onset state and
generated event IDs must match, every delivered event receives one applied or
suppressed disposition, and the target cut must propagate to measured native
wing/root differences. This is a causal software gate, not an empirical
perturbation result.

Final complete-worker evaluation `aws-worker-20260718-v12-canonical-online`
observed the v1.18 partition of 25 pass, 4 explicitly blocked, and 0 fail. Its
report SHA-256 is
`212ea6ac1c7c8f48796df535a96f524bc3e479e1a5149a2b38a17a7459ffb4e6`;
the registry-file/canonical SHA-256 values are
`880dcaf8b79f1b9869e8c7df8266ed7260cdb1cdf3fe24cfe5e0b53c96c04b1d`
and `a6d8ba583a6b39388960ded150da01988bd00580341540aae2b0071349aef795`;
and the worker-image ID is
`sha256:e925ecd09b736faeee2d5f49d30a5f527d5952d29a5be49cd6574eef3233e66b`.
This closes the registered software receipt only: four scientific gates remain
blocked and the result's ceiling is `software_correct`, not calibrated or
validated. The DNg02 protocol requires a permanent driver-line split and
treats open-loop striped-drum and
closed-loop stripe trials as separate claims. Targeted pair count remains a
driver-line proxy rather than an exact recruited-cell measurement. Published
summary reproduction is useful as an intake-integrity fixture, but it is not a
held-out simulator score.

Every result reports:

- wingbeat frequency, stroke amplitude, rotation, deviation, and bilateral asymmetry;
- aerodynamic force/moment and integrated impulse;
- body yaw/roll/pitch rate, translation, trajectory error, and energy;
- numerical diagnostics, evidence coverage, parameter uncertainty, and effect-size intervals.

Exit gate: for both open- and closed-loop DNg02 protocols separately, held-out
driver lines show a positive one-sided 95% cluster bound for activation-induced
wingbeat-amplitude change and targeted-pair slope, and the predicted response
curve meets its preregistered null comparison. The hinge and power-muscle
prerequisites must already pass. All other qualitative/quantitative perturbation
claims meet their literature-derived preregistered criteria.

## Stage 6: close the sensorimotor loop

Current software milestone: canonical online orchestration feeds moving-world
azimuth and the current FlyBody yaw/rate back into the live T4a-only fly-FGS
circuit at each 5 ms boundary. A complete checkpoint restores circuit, bridge,
muscles/hinge, effector hypothesis, and compiled native physics at 50 ms. This
is a first closed software loop, not stable flight or a biologically validated
feedback controller. It has no T5, photoreceptor, lamina, translation/parallax,
measured wing-phase, haltere, or strain pathway; raw `L/R` anatomy and signed
yaw/roll interpretation remain unresolved.

Add one feedback channel at a time with measured latency:

1. moving-world optic flow through the fly visual sampling model;
2. wingbeat phase and wing strain/proprioception;
3. haltere-derived angular velocity;
4. body pose/velocity and flight-state gating.

Validate steady flight before fixation/tracking, optomotor stabilization, figure-ground competition, saccades, and gust recovery. Check stability under latency and sensor-noise perturbations. Keep take-off, landing, and walking out of the flight acceptance suite until airborne control is stable.

Exit gate: held-out closed-loop trials meet trajectory and body-rate criteria with uncertainty coverage; no controller is tuned on its evaluation scenario.

## Stage 7: higher-fidelity verification

Select wingbeats across operating regimes and compare the runtime quasi-steady forces with an independently configured higher-fidelity CFD workflow. Lock geometry, mesh, boundary conditions, convergence study, and solver version. Use CFD to quantify bias and define the runtime model's domain of validity; do not make CFD an interactive web dependency.

## Regression and release policy

- Pull requests run schema/evidence/unit tests, deterministic reduced-order smoke tests, and web typecheck/build.
- FlyBody compilation is manually enabled because the model assets are large.
- A scheduled dedicated worker should run released-policy baselines, timestep convergence, neuromuscular held-out tests, perturbation suites, and posterior predictive checks against immutable approved metrics.
- Any metric change outside its tolerance blocks promotion and requires a new reviewed baseline rather than overwriting the old one.
- Public replays inherit the source run's `exploratory`, `calibrated`, or `validated` label and cannot promote it.
- Registered evaluators return measurements and evidence receipts only. Comparators and source-backed tolerances remain in the versioned benchmark registry, so changing model code cannot silently redefine success.
- Empirical comparators, permanent split rules, hierarchical uncertainty, and
  sealed-execution order live in content-addressed preregistration files. A
  protocol edit requires a new registry revision and cannot alter a historical
  report.
- Every mathematically nonnegative error/convergence metric has an explicit zero lower bound; malformed negative values cannot satisfy a one-sided upper-limit comparator.
- A selected or dependency-scoped report must expose omitted cases and cannot make a release-promotion claim. Only a registry-complete, contract-bound report may compute a promotion ceiling.
- Canonical online scientific runs use artifact schema v1.1. Content-derived
  identity, registered circuit/retinal axes, event dispositions, intervention
  activity, component checkpoints, source/runtime/compiled-model receipts, and
  the replay projection digest must all verify before publication.
- `scripts/publish_canonical_web_run.py` is the only supported online publisher.
  It installs the immutable run before updating the public index and cannot
  promote an exploratory result. The TypeScript build and static release audit
  are independent required gates.
- The replay UI must preserve exact causal distinctions: generated versus held
  rates, pending/applied/suppressed events, natural versus effective muscle
  state, raw versus physical lanes, model-owned versus measured wing phase,
  root versus whole-fly COM, and display-only circuit state versus four
  motor-eligible NOD1 signals.

## Release status and next deliverables

1. Preserve and reproduce the completed v12 software release receipt: the
   final report above and the audited 13-episode web manifest with SHA-256
   `f9290ac9bd81d57044ec3d96ae40fdc12e020affa351a8c7087a836e9ede2902`.
   This receipt cannot change the exploratory scientific status.
2. Version the complete female BANC/FANC DN→premotor→MN→muscle edge table and the raw LLPC1/NOD1 coverage audit.
3. Import the released FlyBody straight-flight/saccade policies and approve clean-worker regression fixtures.
4. Stage the exact Melis HDF5 on object-backed worker storage, run the
   non-downloading fail-closed intake, review its observed SHA-256/topology and
   23/5/5 date split, then resolve feature availability before any fitting.
5. Build canonical intake manifests for the asynchronous-power and DNg02
   archives. Collapse every derivative onto its source recording, resolve or
   explicitly block the calcium animal/ROI lineage, and freeze the DNg02
   driver-line partition without pooling open- and closed-loop trials.
6. Implement the matched causal Melis baseline and three empirical evaluators,
   with frozen predictions and leakage/fit-reference audits. Keep held-out
   targets sealed until all calibration-only model choices are immutable.
7. Calibrate the implemented exploratory `CircuitOutputTrace → DNp26 → WingMotorTrace` bridge, replace its demonstration voltage/rate gains and phases, and extend it through evidence-backed premotor neurons.
8. Fit and register the first parameter posterior, then export paired baseline/perturbation FlyBody episodes without changing their scientific labels prematurely.
