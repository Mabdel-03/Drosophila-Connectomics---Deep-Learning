# fly-sensor2behavior

Flight-first infrastructure for translating Drosophila visual-circuit output into wing motor events, individual muscle state, wing motion, aerodynamic load, and body motion. The repository contains tested exploratory retinal/NOD1-to-flight software slices, a registry-driven evaluation loop, a real FlyGym/FlyBody worker adapter, immutable run artifacts, and a static Three.js replay client.

Live scientific replay: [http://54.160.228.98/fly-sensor2behavior/](http://54.160.228.98/fly-sensor2behavior/). The public host is display-only; authoritative native simulations run in the pinned offline worker and publish immutable artifacts. The replay manifest, rather than this README, is the source of truth for the currently deployed report, registry, protocol, image, and run digests.

This is **not yet a biologically calibrated visual-input-to-flight simulator**. The canonical upstream visual/circuit source is now the deployed [`/fly-fgs`](http://54.160.228.98/fly-fgs/) application. Its content-addressed, wall-clock-independent capture runs a 240 ms static-scene pre-roll followed by 100 samples at 5 ms over a half-open 0.5 s sweep. It contains 1,684 cells: 1,457 T4a cells (1,441 with registered retinotopic luminance input), 219 LLPC1, two vCH, two DCH, and four NOD1 cells. It contains no T5, photoreceptor, or lamina model. Ninety-nine causal `RetinalFrame`s describe the post-pre-roll source intervals, while all cell states remain visualization/audit data. Only the four exact NOD1 voltages enter our evidence-qualified downstream bridge.

The repository also retains an analytic one-dimensional retina/reduced-NOD1 path for manufactured tests, a saved-result legacy importer, and the frozen 1,208-cell Chromium fixture as a historical numerical regression. The fly-FGS page's hand-tuned `SCALE_M`, DN table/gains, muscle state, wing amplitudes/angles, yaw, and toy body mechanics are explicitly excluded; our causal DN/MN bridge, force-stage muscle system, virtual hinge, explicit effector hypothesis, and FlyBody adapter are the downstream implementation. All generated flight output remains `exploratory; calibration: none`. The current behavior is neither stable nor validated flight, and the released-policy convergence plus biological calibration gates remain unsatisfied. The operational topology graph is still a FAFB v783 + MANC v1.0 seed. A separate content-addressed female BANC v888/FANC v840 fixture locks downstream structural observations, but its BANC/FANC premotor identity crosswalk is incomplete and it is not yet the simulator's calibrated causal graph.

## What works now

- Versioned, SI-unit public contracts for `CircuitOutputTrace`, `WingMotorTrace`, `MuscleState`, `FeedbackFrame`, `FlightEpisodeConfig`, `FlightEpisodeResult`, and `ModelRegistry`.
- A frozen manifest for the deployed legacy NOD1 simulator, a strict mV-to-V importer that discards the legacy population-mean `steering` field, and a manufactured-solution-tested Hines tree solver.
- Deterministic analytic grating/figure-ground/random-column scenes, finite-exposure panoramic sampling, causal image-derived Reichardt motion, and a low-confidence reduced NOD1 surrogate for self-supervised end-to-end software tests.
- A canonical fly-FGS intake that pins the deployed page, engine, circuit bundle, and excluded downstream asset; verifies the 1,684-cell/60,200-effective-event inventory; and stores a deterministic fixed-step circuit capture independent of `requestAnimationFrame` or wall time.
- A live, checkpointable Node sidecar that executes that same registered fly-FGS circuit at `5 ms`, feeds body-coupled figure/world controls back at circuit boundaries, and exposes only the four exact NOD1 voltages as motor-eligible output. The complete T4a/LLPC1/vCH/DCH/NOD1 state and 1,441-point retinal input can be attached as digest-bound, display-only online circuit replay data.
- A separate paper-assay sidecar that executes the Figure 3a/3b/3c analytic stimulus at `2.5 ms`, keeps the body/head observation fixed, and preserves the same four-NOD1 motor boundary. Its deterministic 3° random-dot compositor is shared with the browser source, and every runtime asset is content-hashed into the run receipt.
- A causal bridge from individual NOD1 signals to contralateral DNp26, then ipsilateral iv2/i1/iv1/b3 motor channels. It enforces availability timestamps, preserves exact/inferred/seeded-synthetic provenance, and supports typed side-local interventions.
- A streaming runtime with exact `5 ms` circuit, `0.5 ms` bridge, and `0.1 ms` muscle/hinge/physics clocks. Phase crossings use six mechanics endpoints, discrete events are never interpolated, and complete component checkpoints support exact fresh-process/fresh-stack continuation.
- True force-stage `SILENCE` and `SCALE` interventions. Natural muscle state is retained separately from effective force; every generated event is ledgered as pending, applied, or suppressed; and intervention definitions, activity, and schedule digests are checkpoint-bound.
- An explicit two-hypothesis mapping between raw fly-FGS `L/R` lanes and physical wings. Source labels remain anatomically unknown and signed yaw/roll conclusions are prohibited until independent evidence resolves the mapping.
- A paper-only v783 laterality receipt resolving the four NOD1 soma coordinates: both raw `L` cells are anatomically right and both raw `R` cells anatomically left. Paper runs therefore use raw-L→physical-right while retaining the legacy two-hypothesis boundary elsewhere; FAFB roots and MANC motor identifiers remain separate.
- Evidence validation that keeps FlyWire FAFB materialization 783, female BANC/FANC, MANC, and simulation identifiers in separate identity spaces; requires explicit type-level crosswalks; preserves FlyWire root IDs as decimal strings and source coordinates as nanometres; and rejects direct cross-atlas neuron-ID joins.
- A frozen female BANC v888/FANC v840 structural-evidence gate: four NOD1→DNp26 BANC edges total 487 synapses, 18 DNp26→wing-MN edges total the corrected 129 synapses, and the published one-sided FANC premotor matrix remains an independent observation rather than an inferred cross-atlas match.
- A deterministic NumPy flight scaffold with independent physics, neural, and logging clocks; exact, inferred-rate, and seeded-synthetic event paths; asynchronous DLM/DVM, named phase-coded steering muscles, tension, thorax, six-axis virtual-hinge, quasi-steady aerodynamic, and free 6-DoF body models.
- Open-loop baseline, vCH/DCH ablation, DNg02/DNa04/DNp26/DNg32 activation, and gust scenarios.
- Hashed, chunked scientific artifacts with units, provenance, backend/aerodynamic ownership, and explicitly derived browser replays. Legacy/open-loop episodes retain artifact schema v2; the canonical online path uses schema v1.1 with content-bound run identity, complete event/intervention ledgers, component checkpoint receipts, registered circuit axes, and `manifest.sha256`.
- A pinned adapter that compiles and steps the actual FlyBody model through FlyGym/MuJoCo, accepts six-axis wing generalized torques in N m, and records physics-rate actuator torque, measured wing joints, root-total fluid wrench, and exact ground-contact transition-point counts separately from requested hinge motion, root pose, and logging-rate projections.
- Native-worker paths for the 100 ms analytic-retinal/reduced-NOD1 slice, the historical 500 ms registered-Chromium fixture, and the canonical 500 ms fly-FGS fixture. The canonical path forwards only four NOD1 voltages to the motor bridge while retaining the full 1,684-cell state and 1,441-point retinal grid for visualization/audit. Each exported run is linked to an immutable scientific manifest carrying Python/FlyGym/MuJoCo versions, dependency-record hashes, the compiled-model fingerprint, model hashes, exact clocks, units, and identity-boundary receipts.
- A relocatable TypeScript/Three.js replay UI for static hosting at `/fly-sensor2behavior/`, with synchronized body-coupled scene, retinal/T4a/LLPC1/vCH/DCH/NOD1 circuit workspace, causal DN/MN rate views, a discrete event raster, intervention receipts, raw-to-physical mapping, muscle timing, measured wing/body/aerodynamic telemetry, clocks, checkpoints, provenance, and baseline/perturbation comparison.
- A fail-closed, non-downloading Melis wing-hinge intake boundary. It verifies a local candidate's exact source size/MD5 and observed SHA-256, rejects file-identity changes, validates the 74-session/33-date/377-movie HDF5 metadata topology, and constructs a permanent 23/5/5 whole-date split with nine completed input beats. Success remains metadata-only and `candidate_unreviewed`; it cannot pass the hinge gate.
- A 29-case scientific benchmark registry with exact, invariant, differential, frozen-regression, and held-out empirical oracles. Registry v1.18.0 contains 6 fast, 12 pull-request, 8 scheduled-worker, and 3 promotion cases. It adds incremental and checkpointed fly-FGS execution, causal streaming, fresh-process checkpoint re-entry, true force-stage interventions, explicit effector hypotheses, canonical closed-loop orchestration, and the paired native online gate. The final pinned-worker evaluation observed 25 software/numerical/structural passes, 4 explicit blockers, and 0 failures. The blockers are released-policy timestep convergence and the held-out hinge, power-muscle, and DNg02 workflows; the result therefore has a `software_correct` promotion ceiling and does not establish calibrated or validated flight.

The lightweight scaffold is useful for software integration, perturbation semantics, schema development, and regression testing. It is not a substitute for FlyBody or an empirical flight prediction.

## Architecture

```text
evidence graph ───────────────┐
                             v
analytic scene → test retina → reduced NOD1 ─┐
body + moving world → fly-FGS T4a/LLPC1/vCH/DCH → four NOD1 voltages ─┤ 5 ms
saved legacy NOD1 result → strict importer ───────────┴→ DNp26 → wing MNs → individual muscles
                                                       │ 0.5 ms                  │ 0.1 ms
                                                       └→ force-stage muscles → virtual hinge
                                                                                │
                                                                    explicit L/R hypothesis
                                                                                │ 0.1 ms
                                  ┌─────────────────────────────────────────────┴───────┐
                                  v                                                     v
                         exploratory NumPy                                    pinned FlyBody/MuJoCo
                                                                                │
                                                        body pose/yaw ──────────┘ next 5 ms sample
                                  └────────────→ immutable artifacts → static replay
```

Connectome counts constrain topology only. They never become physiological synaptic weights, motor rates, muscle activation, or actuator gains. The FlyBody boundary depends on a `WingTorqueMapper`, so neural and muscle code never addresses MuJoCo actuator indices.

`FlightEpisodeRunner` can inject its muscle-derived six-axis torques through the FlyBody adapter, with FlyBody as the exclusive aerodynamic and contact owner. The canonical fly-FGS, saved-NOD1, and analytic-retinal CLIs expose this path through `--physics-backend flybody`, and their registered FlyBody vertical-slice evaluators exercise it on the pinned worker. The adapter exports physics-rate actuator torque, root-total MuJoCo fluid force/torque, measured six-axis wing qpos/qvel, exact contact-point counts, and the articulated whole-fly center of mass. Logging-rate torque, wing, aerodynamic, and contact channels are exact indexed projections of those authoritative physics-rate traces. Desired virtual-hinge kinematics and the free-joint root/thorax state remain separately labelled; a reviewed per-wing fluid-force decomposition is not available.

See [docs/architecture.md](docs/architecture.md) for component boundaries, clocks, units, identity rules, and the FlyGym 2.1 correction. The canonical live `/fly-fgs` → streaming muscle → external-physics contract is specified in [docs/canonical-online-runtime.md](docs/canonical-online-runtime.md). See [docs/evaluation.md](docs/evaluation.md) for the self-supervised suite and change/validate/iterate workflow, and [docs/calibration-roadmap.md](docs/calibration-roadmap.md) for the evidence, fitting, validation, and promotion sequence.

## Quick start: exploratory core

The evidence and NumPy core support Python 3.9 or newer. FlyBody does not.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

fly-s2b audit-evidence
fly-s2b simulate baseline --duration-s 0.2 --output artifacts/baseline-001
fly-s2b simulate-retinal --duration-s 0.1 --output artifacts/retinal-001
fly-s2b simulate-fly-fgs-fixture --output artifacts/fly-fgs-001
# Or pass an exported legacy result:
fly-s2b simulate-nod1 --input path/to/nod1-result.json --output artifacts/nod1-001
python -m pytest -q
fly-s2b validate --output artifacts/validation/local.json
```

Scientific artifact directories are immutable: choose a new output directory for every changed configuration or seed.

## Paper figure-ground tether assay

The dedicated runner prescribes Reichardt et al. Figure 3a, 3b, or 3c at
400 Hz through the captured 1,684-cell circuit, advances the motor/mechanics
bridge at 2 kHz and FlyBody/MuJoCo at 10 kHz, and measures the root weld's yaw
reaction. It writes 10 kHz raw reaction data, a synchronized 1 kHz channel
table, compressed arrays, phase-locked analysis, trial CSVs, and a 200 Hz
browser replay. The 0.4 s synchronous pre-roll is unrecorded; every trial is
retained. Each published physics sample uses four recorded, deterministic
MuJoCo solver substeps (40 kHz internally for the canonical run) so the stiff
weld remains numerically stable; both cadences are written to provenance.

```bash
fly-s2b-paper-fgs \
  --protocol R83_Fig3a_0_to_90 \
  --repetitions 100 \
  --run-seed 73 \
  --texture-mode registered_copy \
  --output artifacts/paper-fig3a-seed73 \
  --web-replay-output ../fly_fgs_source/data/paper_replay.json
```

The output directory and browser registration paths must not already exist.
Run this command in the pinned Python 3.12/FlyGym/MuJoCo worker. When the
worker checkout does not contain the sibling browser snapshot, mount it
read-only and pass `--source-root /mounted/fly_fgs_source`. The runner refuses
to publish if sign injection, tether drift, fixed-head, contact, measured body
yaw-rate, or generalized yaw-balance checks fail. Instantaneous generalized
root velocity is retained as a separate solver diagnostic and is not presented
as measured body motion. A passing apparatus status does not imply that this
exploratory Drosophila FAFB/MANC/FlyBody model reproduces the housefly response.

The declared release matrix runs all three 100-repeat canonical protocols,
independent-texture and alternate-seed sensitivity cases, plus 800 Hz circuit
and 20 kHz physics convergence cases against the Figure 3a baseline:

```bash
fly-s2b-paper-fgs-release \
  --output-root artifacts/paper-release-v1 \
  --source-root ../fly_fgs_source \
  --expected-node-version v22.22.1
```

For torque mean and harmonic amplitude, the dimensionally inconsistent `2°`
wording in the implementation plan is operationalized as a 2% normalized
change; harmonic phase remains a 2° threshold and phase-binned waveform NRMSE
must remain at or below 5%. The release manifest records this interpretation.

The full validation command is expected to finish with `blocked` gates on a dependency-light host. The registered fly-FGS integrity case validates exact frozen bytes and projections; incremental runtime parity and fresh-process checkpoint re-entry separately test actual Node execution. Native FlyBody cases require the pinned worker. Final evaluation `aws-worker-20260718-v12-canonical-online` produced 25 passes, 4 explicit blockers, and 0 failures. Its report SHA-256 is `212ea6ac1c7c8f48796df535a96f524bc3e479e1a5149a2b38a17a7459ffb4e6`; the registry-file and canonical registry SHA-256 values are `880dcaf8b79f1b9869e8c7df8266ed7260cdb1cdf3fe24cfe5e0b53c96c04b1d` and `a6d8ba583a6b39388960ded150da01988bd00580341540aae2b0071349aef795`; the immutable worker-image ID is `sha256:e925ecd09b736faeee2d5f49d30a5f527d5952d29a5be49cd6574eef3233e66b`; and the 13-episode web manifest SHA-256 is `f9290ac9bd81d57044ec3d96ae40fdc12e020affa351a8c7087a836e9ede2902`. Public Melis, asynchronous-motor, and DNg02 sources have been located, but they have not completed every exact artifact receipt, permanent group split, reconciled identity/provenance requirement, fitted evaluator, and sealed held-out execution. A selected partial suite is useful for development but cannot produce a promotion decision.

Generate the browser replay and run the UI locally:

```bash
fly-s2b export-web \
  --output web/public/data \
  --validation-report artifacts/validation/local.json \
  --overwrite
cd web
npm ci
npm run dev
```

The web build uses relative URLs and can be copied directly beneath the static route:

```bash
npm run build
# Deploy web/dist/* as /fly-sensor2behavior/ on the display host.
```

The public host is display-only. It serves replay artifacts and runs no MuJoCo simulation per request.

The replay is operated from one page: choose an episode or perturbation, toggle a compatible baseline overlay, scrub or play the shared timeline, select slow motion, and inspect the 3D trajectory, circuit path, retinotopic T4a/LLPC1 activity, vCH/DCH/NOD1 traces, individual muscles, body state, evidence, and provenance together. The canonical fly-FGS replay synchronizes those circuit views to the same stored timeline as the FlyBody output; browser rendering selects a time and never advances the authoritative circuit. Its full state is display/audit-only, and only four NOD1 voltages are motor inputs. Standard reduced-order scenarios without an upstream trace remain explicitly illustrative. FlyBody episodes may include an exact 10 kHz inspection window showing measured wing joints and six-axis actuator torque without reconstructing them from decimated display frames. Every current native-worker trace remains an uncalibrated actuation demonstration, not stable-flight or a biological prediction.

## Actual FlyBody worker

The reviewed worker stack is pinned to:

- Python `3.12.11` in the container image;
- FlyGym `2.1.0` (`ca65a510c2afe6ac61c51df4f274c8d190c2f95f`);
- MuJoCo `3.9.x` (the lock currently resolves `3.9.0`);
- reviewed FlyBody aerodynamics source `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.

FlyGym 2.1's experimental importer attaches a centimetre-authored FlyBody model to a millimetre world but does not perform all flight-specific reconstruction. The adapter corrects the air density/viscosity units, restores the two wing-fluid ellipsoids, applies the released flight wing stiffness and damping, retains the passive abdomen/tarsus tendons, and exposes six torque-controlled wing DoFs. Outputs are converted to SI units.

Operational flight episodes use a `0.1 ms` MuJoCo physics step. The reviewed analytic fallback and fixed-command convergence stimulus advance their prescribed controller at `0.2 ms` and zero-order-hold each command across physics substeps; this controller clock is distinct from the visual, neural, muscle, and logging clocks. The default worker topology retains the articulated legs and leg-ground contact against the FlyGym arena floor. Contact telemetry counts detected MuJoCo contact points, not active contact forces: sample 0 is the reset-state `mj_forward` list, and sample `i >= 1` is the pre-integration `mjContact` list used for the transition ending at that sample time. It is not a collision query at the displayed post-step pose. Released flight evaluations disabled legs/floor contact, so the current adapter is not policy-equivalent even when its integration and fluid-force checks pass.

Build and exercise the worker:

```bash
docker build -t fly-s2b-worker .
docker run --rm \
  -v fly-s2b-assets:/opt/flygym-assets \
  fly-s2b-worker flybody-smoke --mode analytic-wingbeat --duration-s 0.005
```

The first worker run downloads FlyBody meshes into the mounted asset cache. `analytic-wingbeat` is only a compatibility smoke using FlyBody's analytic fallback pattern; it is not the released learned straight-flight/saccade controller and not a calibrated neuromuscular result. The registered v3 fixture pins the exact FlyGym/MuJoCo runtime receipts, compiled-model fingerprint and topology counts, `0.2 ms` advance-then-hold control semantics, `0.05 ms` fixture step, collision-disabled wing-fluid proxies, legs-only contact, and explicit `released_policy_topology_equivalent: false`. Use `--mode zero-torque --steps 2` for the smallest structural smoke. Routine CI deliberately does not install FlyGym or download these assets; a manually enabled workflow job runs the compiled-model test.

Inside the pinned worker, either causal pipeline can select FlyBody explicitly:

```bash
fly-s2b simulate-retinal \
  --physics-backend flybody \
  --flybody-spawn-height-m 0.1 \
  --duration-s 0.02 \
  --output artifacts/retinal-flybody-001

fly-s2b simulate-nod1 \
  --input path/to/nod1-result.json \
  --physics-backend flybody \
  --flybody-spawn-height-m 0.1 \
  --output artifacts/nod1-flybody-001
```

The canonical online path is the preferred integration workflow. The registered
pair uses a 100 ms moving figure, seed 73, the explicit raw-L→physical-left
hypothesis, and a raw-L iv2 force-stage silence over `[60, 100) ms`:

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

The complete pair contract is frozen in
`data/benchmarks/scenarios/canonical-online-native-pair.v1.json`. Generate every
immutable artifact and its bound replay before publishing either arm. Then use
the fail-closed publisher; it verifies the manifest, projection, run identity,
and byte receipts and updates the public index last:

```bash
python scripts/publish_canonical_web_run.py \
  --artifact-dir /tmp/canonical-online-baseline \
  --replay /tmp/canonical-online-baseline.web.json \
  --episode-id canonical-online-baseline \
  --condition "moving figure; baseline" \
  --description "Live fly-FGS circuit through the causal actuator and native FlyBody." \
  --color '#2dd4bf' \
  --position 0
```

Publication does not promote the scientific status. Build the web client and
run `python scripts/audit_web_release.py` after publication.

FlyBody currently has no reviewed ambient-wind setter at this boundary. Any nonzero wind perturbation therefore fails closed instead of being ignored; the checked-in reduced-order gust scenario remains a reduced-model test.

## CLI

| Command | Purpose | Scientific status |
|---|---|---|
| `fly-s2b audit-evidence` | Validate the seed graph and report provisional LLPC1/NOD1 coverage | Evidence audit only |
| `fly-s2b simulate SCENARIO --output DIR` | Run the reduced-order open-loop engine and write hashed arrays | Exploratory, uncalibrated |
| `fly-s2b simulate-nod1 --input FILE --output DIR [--physics-backend ...]` | Run saved individual NOD1 voltages through DNp26, wing MNs, muscles, and reduced-order or FlyBody mechanics | Executable legacy-circuit slice; source retinal frames unavailable and motor/hinge model uncalibrated |
| `fly-s2b simulate-retinal --output DIR [--physics-backend ...]` | Run an analytic scene through causal retinal samples, reduced NOD1, and reduced-order or FlyBody mechanics | Complete software slice; visual/neural/motor models are uncalibrated surrogates |
| `fly-s2b simulate-fly-fgs-fixture --output DIR [--physics-backend ...]` | Run the registered fixed-step fly-FGS circuit output through the evidence-qualified bridge and selected mechanics | Canonical upstream software slice; only four NOD1 voltages drive motor output; mechanics uncalibrated |
| `fly-s2b simulate-canonical-online --output DIR --effector-hypothesis ...` | Run live fly-FGS at 5 ms through the 0.5 ms streaming bridge, 0.1 ms force-stage mechanics, explicit physical-wing mapping, and native FlyBody; optionally write a bound browser replay | Preferred causal online integration; exploratory and uncalibrated, with raw laterality unresolved |
| `fly-s2b export-web --output DIR` | Generate projected replay JSON plus bounded exact physics-rate inspection data for selected FlyBody episodes | Display derivative, not source of record |
| `fly-s2b flybody-smoke` | Compile and step the pinned real FlyBody worker | Compatibility test only |
| `fly-s2b validate` | Evaluate registered cases and write a deterministic gate report | Mixed: local passes plus explicit blocked worker/empirical gates |

Run `fly-s2b COMMAND --help` for clock, scenario, seed, decimation, and output options.

## Artifact layout

```text
artifacts/<run>/
├── manifest.json          # config, hashes, clocks, event/intervention ledgers, checkpoints
├── manifest.sha256        # hash of the complete manifest
├── events.json            # canonical online generated-event disposition table
├── rates.json             # generated and causally held DN/MN rates
├── intervention_activity.json
├── checkpoint.json        # complete resumable component state
└── arrays/
    └── <logical-name>-<hash>/
        ├── 000000.npy     # independently hashed, pickle-disabled chunk
        └── ...

web/public/data/
├── manifest.json          # replay catalog and evidence notices
├── episodes/*.json        # decimated display products
├── validation/report.json # optional native gate report attached at export
└── runs/<content-id>/     # immutable chunked scientific source runs
```

The artifact writers validate the complete episode before publication and stage every file atomically, so an invalid clock, shape, unit, non-finite value, event ledger, checkpoint, or backend/ownership receipt cannot leave a partial run. Legacy/open-loop artifact schema v2 stores model/evidence hashes, individual-muscle force/phase/work, and run-specific physics telemetry. Canonical online artifact schema v1.1 additionally binds run identity to scientific content, stores generated and held causal rates in declared axes, assigns every generated event exactly one pending/applied/suppressed disposition, records force-stage intervention activity, and preserves a complete hash-protected checkpoint plus source/runtime/compiled-model receipts. Its registered 1,684-cell and 1,441-retinal axes are explicit. A digest-bound `online_circuit_replay` exposes the exact sampled retinal/full-state circuit data to the browser while remaining display-only and motor-ineligible. FlyBody telemetry preserves desired command, measured wing qpos/qvel, root-total aerodynamic wrench, free-joint root/thorax state, and articulated whole-fly center of mass as distinct quantities. Browser JSON is a verified projection, not the source of record. Historical fixture/run paths remain useful for regression but their release counts and hashes are not current receipts.

## Evidence limits

The operational seed graph contains only:

- selected FAFB v783 LLPC1/NOD1-to-DN structural counts copied from the hashed deployed `fly-fgs` artifact;
- explicit type-label crosswalks into MANC v1.0; and
- MANC type-to-muscle summaries aggregated **through** unnamed motor neurons.

The separately registered female evidence fixture contains BANC v888 NOD1→DNp26 and DNp26→individual wing-MN edges, 62 proofread wing MNs (58 accepted named-atlas mappings and four unresolved rows), and the FANC v840 one-sided 1,784-row premotor-to-wing-MN matrix. The BANC DNp26→wing-MN total is **129** raw structural synapses across 18 edges, correcting the earlier provisional value of 131. FANC contains 144,668 thresholded structural synapses across 7,289 nonzero pairs and two annotated DNp26 rows totaling 153 synapses onto named wing-MN columns.

These are independent structural snapshots, not physiological weights and not a completed BANC↔FANC individual-neuron map. No premotor matching was performed, no candidate premotor pair was promoted, and the premotor identity crosswalk remains incomplete. The fixture therefore supports a structural-evidence gate but does not yet supply an explicit end-to-end DN→identified premotor→MN→muscle causal graph for the simulator. No FlyWire root ID is joined to a BANC, FANC, or MANC body/root ID. FlyWire FAFB records are locked to materialization 783; root IDs remain decimal strings so JavaScript cannot round them, and local skeleton/synapse coordinates remain in nanometres until an explicit SI conversion. The reported 28.4% LLPC1 and 76.5% NOD1 whole-pathway coverage values remain provisional plan-supplied totals whose raw audit artifact is not in this clone. They must not be described as reproduced coverage. FAFB laterality follows the locked rule that higher soma x is fly-left.

## Acceptance status

| Gate | Current status |
|---|---|
| Versioned interfaces, SI units, cross-atlas ID rejection, evidence provenance | Implemented and unit-tested |
| Frozen legacy NOD1 manifest, strict individual-voltage import, and mV→V conversion | Implemented and unit-tested; legacy side labels and source Git provenance remain unresolved |
| Saved NOD1→contralateral DNp26→ipsilateral steering-MN vertical slice | Implemented and unit-tested; gains, phases, and route remain exploratory |
| Analytic scene→test retina→Reichardt→reduced NOD1→body software slice | Implemented with causal/self-supervised tests; not the audited full circuit or a calibrated eye |
| Registered fly-FGS scene→retinotopic input→T4a→LLPC1/vCH/DCH→NOD1 integration | Implemented as a deterministic 5 ms fixture with 240 ms pre-roll, 100 full-state samples, and 99 causal `RetinalFrame`s; no T5, photoreceptor, lamina, or calibrated eye model |
| Deterministic reduced-order episodes and immutable hashed arrays | Implemented and unit-tested |
| Reduced-order 0.1 ms versus 0.05 ms convergence | Registered local gate passes its 2% impulse/body-state limits; cannot satisfy the separate FlyBody gate |
| Registered Chromium-worker/Python NOD1 numerical parity | Content-addressed fixture and strict evaluator implemented; this validates cross-runtime numerics only, not circuit biology |
| Real FlyGym/FlyBody model compilation, air correction, non-colliding fluid geoms, damping, tendons, and six wing actuators | Implemented; v3 compatibility fixture pins compiled/runtime/control/contact fingerprints and the scheduled worker test downloads large assets |
| Route visual/circuit-derived muscle torque through real FlyBody | FlyBody backend is CLI-selectable; retinal slice v2.2.0 adds a typed `MN-b3-left` cut, while registered-Chromium slice v1.2.0 runs all 100 fixture samples for 0.5 s and adds an exact repeat plus a typed `MN-iv1-right` cut. Both prove software causality only |
| Route canonical fly-FGS output through real FlyBody | Frozen 0.5 s vertical-slice regression remains registered; the preferred scheduled online gate ran the live circuit/streaming stack for a paired 100 ms baseline and raw-L iv2 `SILENCE` arm and passed its causal software contract in final evaluation `aws-worker-20260718-v12-canonical-online`. Neither gate establishes stable or validated flight |
| FlyBody measured-vs-desired/body-reference telemetry | Authoritative physics-rate actuator torque, measured six-axis wing state, root-total aerodynamic wrench, and transition-contact counts are stored separately from desired virtual-hinge kinematics, root/thorax pose, and logging-rate projections; per-wing fluid decomposition is unavailable |
| Fixed-command FlyBody 0.1/0.05/0.025 ms open-loop convergence | Registered scheduled numerical stress gate; it is not a stable-flight or released-policy claim |
| Reproduce released FlyBody straight-flight and saccade regressions | Explicitly blocked until the immutable policy, WPG, normalization, inference adapter, and baselines are installed |
| Female BANC/FANC wing-pathway structural evidence | Content-addressed gate implemented: BANC NOD1→DNp26, 62 wing MNs, corrected 129-synapse DNp26→wing-MN total, and independent FANC premotor matrix |
| Explicit cross-atlas DN→identified-premotor→MN→muscle causal graph | Pending; BANC/FANC premotor matching was not performed and candidate pairs remain unresolved |
| Fit indirect, steering, tension, hinge, sign, moment-arm, and gain distributions | Pending |
| Held-out muscle-to-wing-kinematics and indirect-muscle physiological validation | Public sources and content-addressed evaluation protocols are located; the Melis metadata/date-split intake is implemented, while held-out fitting/evaluation and complete per-animal calcium provenance remain blocked |
| DNg02 population/dose validation | Public female tethered-flight HDF5/code and a separate-protocol driver-line-held-out contract are identified; exact intake, stable identity, fitted evaluator, and sealed execution remain blocked |
| Closed-loop retinal/body feedback | Canonical online orchestration feeds current body yaw/rate and moving-world state into the next 5 ms fly-FGS sample and checkpoints the full stack; measured wing phase, haltere, strain/proprioception, and flight-state gating remain pending |
| End-to-end validated flight, uncertainty intervals, and offline CFD comparison | Pending |
| Static replay and baseline/perturbation comparison | Implemented with synchronized online scene/circuit/rate/event/intervention/muscle/wing/body/provenance views; outputs remain exploratory |
| Registry-driven self-supervised evaluation and hard promotion gates | Registry v1.18.0 contains 29 cases: 6 fast, 12 pull-request, 8 scheduled, and 3 promotion. The final suite-complete worker report contains 25 pass + 4 explicit blocked + 0 fail and has a `software_correct` ceiling; partial reports cannot promote |
| Held-out hinge, power-muscle, and DNg02 empirical gates | Content-addressed protocols are registered, but gates remain blocked until every licensed artifact/intake receipt, group identity, permanent split, fit, evaluator, and sealed execution is complete |
| Authenticated queued worker and S3-backed authoritative runs | Pending |

Promotion is deliberately conservative: no model can move from `exploratory` to `calibrated` or `validated` until its parameters and held-out evaluation data are versioned and the declared acceptance gate passes.
