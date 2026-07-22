# Native MuJoCo Figure–Ground Scientific Release

This file is the authoritative execution runbook for the paper-faithful
figure–ground assay in this repository. Follow it in order. Do not substitute a
browser signal, schematic wing command, dimensionless yaw proxy, or fitted
paper overlay for a native physical measurement.

## 1. Scientific authority and non-negotiable boundaries

- The browser model is an exploratory neural preview. Its output is
  `steering_signal_au`; it is never torque and never receives a physical unit.
- Only the structurally fixed FlyBody/MuJoCo worker in
  `../fly_fgs_native_worker` may produce `authoritative_native_torque`.
- The whole-fly thorax/root support load cell is the authoritative paper
  measurement. The equality-weld reaction is an independent validator.
- Left and right wing-root measurements are calibrated simulation-only
  mechanical decompositions. Reichardt et al. did not publish a per-wing trace,
  so never score either wing separately against Figure 3.
- Never replace whole-fly torque with the sum of the two wing channels.
- Do not fit gain, offset, baseline, latency, time shift, time warp, a transfer
  function, or protocol-specific scaling to the paper.
- Apparatus validation and biological agreement are separate. A mismatch with
  the housefly traces must remain visible and must not be tuned away.
- The prescribed figure and ground angles are open loop. Neural, muscle, wing,
  torque, body, or head values may not alter the stimulus or its clock.
- Do not regenerate a locked input, dependency lock, calibration receipt, or
  source manifest merely to make a failed check pass.

Channel authority labels are fixed:

| Channel | Required label |
|---|---|
| Whole-fly support torque | `authoritative_native_torque` |
| Left/right complete wing-root torque | `calibrated_simulation_decomposition` |
| Left/right aerodynamic-only probes | `validated_fluid_diagnostic` |
| Digitized Figure 3 trace | `digitized_reference_no_reported_sem` |

## 2. Locked source and dataset provenance

The native worker is a content-addressed sibling of this directory. It includes
the Python/MuJoCo source, Node runtime bridge, Dockerfile, hash-locked Python
dependencies, reference data, release matrix, tests, and the paper inputs. It
deliberately excludes the unrelated native repository web build and all
generated scientific arrays.

Before any build, run:

```bash
export REPO_ROOT=/absolute/path/to/Drosophila-Connectomics---Deep-Learning
export PAPER_FGS_SOURCE_ROOT="$REPO_ROOT/fly_fgs_source"
export PAPER_FGS_WORKER_ROOT="$REPO_ROOT/fly_fgs_native_worker"

python3 "$PAPER_FGS_WORKER_ROOT/scripts/verify_vendored_bundle.py"
sha256sum "$PAPER_FGS_WORKER_ROOT/reference/BF00595226.pdf"
sha256sum "$PAPER_FGS_WORKER_ROOT/reference/figure_ground_relative_motion_simulation_spec.md"
```

The required reference hashes are:

```text
BF00595226.pdf
9159ec24548e1ee0eaf4edca0d4790d7656890d2fb611dd0b7d27cc85a6f05d7

figure_ground_relative_motion_simulation_spec.md
c43b9ec417a52d5fac5333c1df01ca0765961f5c88779a3769fc9b3a2968647c
```

Any inventory, size, or checksum mismatch is a hard stop. Create a new reviewed
bundle and new run directory; never resume an old release after source drift.

The circuit is the frozen FlyWire FAFB materialization 783 browser bundle:

- Preserve all FlyWire root IDs as decimal strings.
- Higher FAFB soma x is anatomical fly left and lower soma x is anatomical fly
  right. Raw application `L/R` labels are not anatomical truth.
- The NOD1 receipt is the only resolved laterality transform used by this assay.
- FAFB-v783 brain IDs and MANC `male-cns:v1.0` body IDs are separate identifier
  spaces and must never be joined as though they were one connectome.
- Structural synapse counts are exploratory relative weights, not calibrated
  physiological strengths.
- The frozen circuit does not gain authority from a newer live catalog.

At release time, record the live public manifest and status for provenance only:

```bash
mkdir -p "$PAPER_FGS_ARTIFACT_PARENT/${PAPER_FGS_RELEASE_NAME}-preflight"
curl -fsS http://54.160.228.98/drosophila/api/manifest \
  -o "$PAPER_FGS_ARTIFACT_PARENT/${PAPER_FGS_RELEASE_NAME}-preflight/flywire_manifest.json"
curl -fsS http://54.160.228.98/drosophila/api/status \
  -o "$PAPER_FGS_ARTIFACT_PARENT/${PAPER_FGS_RELEASE_NAME}-preflight/flywire_status.json"
```

Record an endpoint failure rather than changing the frozen v783 circuit. As of
2026-07-22, the live manifest reported 139,255 proofread roots, 112,790
community-named records, 144,248 catalog records, and 139,259 full skeleton
files; live status reported 139,264 neurons with skeletons. This scope/refresh
discrepancy is provenance metadata, not a reason to rewrite the model.

## 3. Host, storage, and container setup

The release host must be Linux x86-64 with Docker, at least 16 logical CPUs,
128 GiB RAM, and 250 GiB free on a dedicated artifact volume. GPU access is not
required. MuJoCo uses EGL.

Choose an absolute external location. It must not be inside the Git checkout:

```bash
export PAPER_FGS_ARTIFACT_PARENT=/absolute/path/on/large/artifact-volume
export PAPER_FGS_RELEASE_NAME=paper-fgs-v3-release-001
export PAPER_FGS_ARTIFACT_ROOT="$PAPER_FGS_ARTIFACT_PARENT/$PAPER_FGS_RELEASE_NAME"
```

Do not create `PAPER_FGS_ARTIFACT_ROOT` manually. The release runner creates and
journals it atomically. A separate `-preflight` sibling is safe for logs.

Run the fail-closed preflight:

```bash
python3 "$PAPER_FGS_WORKER_ROOT/scripts/paper_fgs_preflight.py" \
  --source-root "$PAPER_FGS_SOURCE_ROOT" \
  --artifact-root "$PAPER_FGS_ARTIFACT_ROOT"
```

Build the exact digest-pinned image and make the FlyGym asset cache persistent:

```bash
docker build --pull -t fly-fgs-native:locked "$PAPER_FGS_WORKER_ROOT"
docker volume create fly-fgs-assets-2-1-0
docker image inspect fly-fgs-native:locked \
  > "$PAPER_FGS_ARTIFACT_PARENT/${PAPER_FGS_RELEASE_NAME}-preflight/docker_image_inspect.json"

docker run --rm --entrypoint python fly-fgs-native:locked \
  /app/scripts/verify_vendored_bundle.py
```

The locked image uses Python 3.12.11, Node 22.22.1, FlyGym 2.1.0, and MuJoCo
3.9.0. Its base images are pinned by digest in the Dockerfile. Do not install or
upgrade packages interactively inside the release container.

## 4. Mandatory tests before scientific trials

Run the static browser tests with the image's locked Node executable:

```bash
docker run --rm \
  --entrypoint /bin/sh \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source:ro" \
  fly-fgs-native:locked \
  -lc 'node --test /fly_fgs_source/tests/*.test.mjs'
```

Run the complete native test suite with EGL and the persistent FlyGym cache:

```bash
docker run --rm --init \
  --entrypoint python \
  -e MUJOCO_GL=egl \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source:ro" \
  -v fly-fgs-assets-2-1-0:/opt/flygym-assets \
  fly-fgs-native:locked \
  -m pytest -q
```

No scientific run may start with a failed, skipped-for-convenience, or
unexpectedly collected test. In particular, require the stimulus, runtime
integrity, open-loop isolation, native topology, calibration, per-wing sensor,
aerodynamic probe, FIR, replay-v2 fallback, replay-v3, and interruption/resume
tests to pass.

## 5. Mechanical measurement contract

The authoritative adapter must meet all of these conditions:

1. The thorax/root is structurally fixed to world; it has no free root joint or
   soft equality standing in for the authoritative meter.
2. A force/torque sensor is located at the root-body origin on the vertical
   tether axis. Body and head remain fixed while wing joints, actuators, virtual
   hinges, and aerodynamic wing geometry remain active.
3. Ground contact and stimulus contact are zero. The rendered cylinders and
   textures never enter MuJoCo.
4. Engine coordinates are explicitly registered as `+x` forward, `+y`
   anatomical left, and `+z` up. The yaw projection comes from injected-axis
   calibration, never a hard-coded generalized-coordinate offset.
5. The equality validator reconstructs only the six rows whose `efc_type` and
   `efc_id` identify the tether weld, using `J^T f`. Aggregate
   `qfrc_constraint` is not an equality-only load-cell measurement.
6. Root fluid moment and total angular-momentum balance are diagnostics, not
   substitutes for the support reaction.

The sign contract is:

```text
attempted_fly_yaw_engine = -support_on_fly_yaw
reported_paper_torque = -attempted_fly_yaw_engine
                      = support_on_fly_yaw
```

The released paper-positive convention must map an injected clockwise
`1e-7 N·m` moment to `+1 dyne·cm` and the counterclockwise injection to
`-1 dyne·cm`.

Each sensor site is located at the existing physical `l_wing` or `r_wing`
hinge origin and adds no body, mass, joint, actuator, or degree of freedom.
Physical `l_wing` is anatomical left/body `+y`; physical `r_wing` is anatomical
right/body `-y`. These labels are independent of the inverted FAFB application
side labels.

For each raw sample calculate:

```text
wing_on_thorax_wrench = -parent_on_wing_sensor_wrench
moment_at_tether = moment_at_hinge
                 + (hinge_position - tether_position) × force
paper_positive_wing_torque = -moment_at_tether.z
wing_sum = left_wing_torque + right_wing_torque
nonwing_residual = authoritative_total_torque - wing_sum
```

Run full-fluid, left-only, and right-only non-integrating shadow models from the
authoritative model's named joint positions and velocities. They may call
`mj_forward` but may never step or mutate the authoritative state.

## 6. Stimulus, neural, motor, and sampling contract

- Figure 3a: 2 s, ±5°, 2.5 Hz, 0°→+90° transition beginning at 0.4 s through a
  0.4 s linear phase ramp.
- Figure 3b: the same timing with a signed −90° transition, corresponding to
  0°→270°.
- Figure 3c: 4 s, ±7.5°, 2.5 Hz, 0°→180° transition beginning at 1.2 s through
  a 0.4 s ramp.
- Use the locked 3°×3° binary texture, 120 wrapped azimuth columns, canonical
  registered-copy seed 123456, 12° opaque curved figure stripe centered at
  paper azimuth +30°, and the locked normalized luminance contrast.
- Use one unrecorded 0.4 s synchronous pre-roll and one unrecorded 0.4 s
  post-roll. Positions, not integrated velocities, are authoritative.
- Hold body/head sensory observations fixed. Circuit or motor changes must
  produce bit-identical stimulus time, figure angle, ground angle, and receptor
  luminance arrays.
- Run the canonical circuit at 400 Hz with its recalculated 2.5 ms
  discretization. The 800 Hz version is a convergence sensitivity only.
- Preserve the registered 1,684-cell circuit and the four-cell NOD1-only motor
  boundary. LLPC1 remains observable but does not enter the paper-mode motor
  path.
- Use deterministic trial-specific motor seeds. Canonical trials use a seeded,
  uniformly stratified distribution of 100 initial wingbeat phases. Fixed phase
  is a sensitivity control only.
- Publish 10 kHz physics with four 25 µs internal substeps, retaining 40 kHz raw
  torque. The 20 kHz convergence runs retain 80 kHz raw torque.
- Timestamp at completed integration states. Preserve every raw sample.
- Produce the synchronized 1 kHz channel with the locked delay-compensated
  linear-phase FIR: passband through 350 Hz, stopband from 500 Hz, ≤0.01 dB
  ripple, and ≥100 dB stopband attenuation.
- Produce `paper_comparison`, `lowpass_10hz`, `lowpass_25hz`, `lowpass_50hz`,
  and `wingbeat_averaged`. The latter four are sensitivity products and may not
  be selected after seeing which resembles the paper.
- Decimate only the browser projection to 200 Hz.

## 7. Calibration and release gates

The calibration receipt must cover signed moments of ±`1e-9`, ±`1e-8`,
±`1e-7`, and ±`2e-7 N·m`; known forces and lever arms; all three axes;
cross-axis leakage; 2.5 and 200 Hz injections; zero offset; exact unit
conversion; both wing sensors; and the sensor/tether transforms.

Required numerical gates:

| Gate | Limit |
|---|---|
| Load-cell gain error | ≤0.05% |
| Zero-load offset | ≤`1e-12 N·m` |
| Cross-axis leakage | ≤0.1% |
| Lever-arm error | ≤0.1% |
| 2.5 Hz phase error | ≤0.1° |
| Translation drift | <1 µm |
| Rotation drift | <`1e-4 rad` |
| Logged yaw rate | <`1e-3 rad/s` |
| Mechanical residual | `max(0.1% of peak, 1e-10 N·m)` |
| Fixed/equality mean and amplitude disagreement | ≤1% |
| Fixed/equality phase disagreement | ≤1° |
| Fixed/equality waveform NRMSE | ≤2% |
| Aerodynamic probe reconstruction | `max(0.1% of peak, 1e-10 N·m)` |
| 40/80 kHz and 400/800 Hz mean/amplitude change | ≤2% |
| 40/80 kHz and 400/800 Hz phase change | ≤2° |
| Convergence waveform NRMSE | ≤5% |
| Peak process RSS | <16 GiB |

Apply convergence gates independently to authoritative total, left wing, right
wing, and wing sum. Biological response sign or magnitude is not a release
gate.

## 8. Dry run and full 1,200-trial execution

The versioned matrix contains 12 sequential runs of 100 trials:

1. `fig3a-physics-20khz`
2. `fig3a-circuit-800hz`
3. `fig3a-canonical`
4. `fig3b-physics-20khz`
5. `fig3b-canonical`
6. `fig3c-physics-20khz`
7. `fig3c-canonical`
8. `fig3a-fixed-wingbeat-phase`
9. `fig3a-equality-validator`
10. `fig3a-independent-texture`
11. `fig3a-seed-123457`
12. `fig3a-seed-654321`

Do not reorder or parallelize these runs. Canonical runs consume matched
high-rate/circuit references produced earlier in the matrix.

First inspect the exact commands and storage gate without starting MuJoCo:

```bash
docker run --rm --init \
  --entrypoint python \
  -e MUJOCO_GL=egl \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source:ro" \
  -v "$PAPER_FGS_ARTIFACT_PARENT:/artifacts" \
  -v fly-fgs-assets-2-1-0:/opt/flygym-assets \
  fly-fgs-native:locked \
  -m fly_sensor2behavior.paper_fgs_release_cli \
  --output-root "/artifacts/$PAPER_FGS_RELEASE_NAME" \
  --source-root /fly_fgs_source \
  --dry-run
```

Run the release in a durable terminal such as `tmux`. Do not mount the Git
checkout writable during computation:

```bash
docker run --rm --init \
  --entrypoint python \
  -e MUJOCO_GL=egl \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source:ro" \
  -v "$PAPER_FGS_ARTIFACT_PARENT:/artifacts" \
  -v fly-fgs-assets-2-1-0:/opt/flygym-assets \
  fly-fgs-native:locked \
  -m fly_sensor2behavior.paper_fgs_release_cli \
  --output-root "/artifacts/$PAPER_FGS_RELEASE_NAME" \
  --source-root /fly_fgs_source
```

The runner preallocates one `.npy` memmap per native channel and run, writes and
flushes 4,096-sample blocks, then atomically records the trial receipt and
checksums in `release_state.json`. A trial interrupted before that commit is
restarted from deterministic pre-roll. A committed trial is immutable.

Inspect progress without running a trial:

```bash
docker run --rm --entrypoint python \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source:ro" \
  -v "$PAPER_FGS_ARTIFACT_PARENT:/artifacts" \
  fly-fgs-native:locked \
  -m fly_sensor2behavior.paper_fgs_release_cli \
  --output-root "/artifacts/$PAPER_FGS_RELEASE_NAME" \
  --source-root /fly_fgs_source \
  --status
```

Resume after interruption with the exact original mounts and configuration:

```bash
docker run --rm --init \
  --entrypoint python \
  -e MUJOCO_GL=egl \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source:ro" \
  -v "$PAPER_FGS_ARTIFACT_PARENT:/artifacts" \
  -v fly-fgs-assets-2-1-0:/opt/flygym-assets \
  fly-fgs-native:locked \
  -m fly_sensor2behavior.paper_fgs_release_cli \
  --output-root "/artifacts/$PAPER_FGS_RELEASE_NAME" \
  --source-root /fly_fgs_source \
  --resume
```

Resume verifies every checksum of every completed trial before advancing. If
the source, matrix, browser runtime, reference, dependency lock, model, or run
configuration digest differs, the runner must refuse the resume. Create a new
release name; do not alter the old journal.

## 9. Artifact layout and deep verification

Each matrix run contains:

```text
<run-name>/
  release_state.json
  array_specs.json
  arrays/*.npy
  trial_receipts/trial_000.json ... trial_099.json
  artifact/
    manifest.json
    scientific_arrays.npz
    analysis.json
    phase_locked_average.csv
    multichannel_phase_locked_average.csv
    trial_metrics.csv
    figure3_reference_for_protocol.json
    paper_overlay_and_residual.csv
    web_replay.json
```

The release root also contains its matrix journal and immutable
`release_manifest.json`. Raw memmaps and the compressed scientific artifact
remain on external storage and are never committed to Git.

After all runs finish, deep-verify all 1,200 trial receipts and raw array
checksums:

```bash
docker run --rm --init \
  --entrypoint python \
  -v "$PAPER_FGS_ARTIFACT_PARENT:/artifacts:ro" \
  fly-fgs-native:locked \
  /app/scripts/verify_scientific_release.py \
  --release-root "/artifacts/$PAPER_FGS_RELEASE_NAME"
```

This must report 12 verified runs, 1,200 verified trials, all three canonical
runs authoritative, and peak RSS below 16 GiB.

## 10. Figure 3 comparison

Use only the locked digitization of PDF page 5 rendered at 600 dpi. The
reference records panel crops, all axis calibration points, two independent
centerline traces, and per-sample uncertainty. Axis landmarks must round-trip
within 0.25 rendered pixels.

For authoritative whole-fly torque, report:

- raw physical overlay and residual in dyne·cm;
- pre-transition, post-transition, and delta mean;
- first-harmonic amplitude and phase against both ground position and velocity;
- RMSE, NRMSE, MAE, mean bias, and waveform correlation;
- diagnostic cross-correlation lag, clearly marked as unapplied;
- simulation SEM and digitization uncertainty as separate quantities.

The paper curve is a 100-sweep mean from one typical fly and has no reported
SEM. Do not manufacture a paper SEM or compare the paper trace with either
individual wing. Show all named filter sensitivities without declaring the
closest-looking one primary.

## 11. Browser replay publication

Publish only after deep verification succeeds. The publication pass reads the
completed external release and writes only compact v3 replays, manifests,
checksums, and the index into this static site. It rejects any Git-bound file
over 90 MiB and keeps raw arrays external.

Run this once with `fly_fgs_source` writable:

```bash
docker run --rm --init \
  --entrypoint python \
  -e MUJOCO_GL=egl \
  -v "$PAPER_FGS_SOURCE_ROOT:/fly_fgs_source" \
  -v "$PAPER_FGS_ARTIFACT_PARENT:/artifacts:ro" \
  -v fly-fgs-assets-2-1-0:/opt/flygym-assets \
  fly-fgs-native:locked \
  -m fly_sensor2behavior.paper_fgs_release_cli \
  --output-root "/artifacts/$PAPER_FGS_RELEASE_NAME" \
  --source-root /fly_fgs_source \
  --web-replay-root /fly_fgs_source/data \
  --publish-only
```

The publication step stages an immutable `paper_replays_v3_<digest>` bundle,
backs up an existing index with a UTC timestamp, and atomically replaces
`data/paper_replay_index.json`. Verify the static tests again after publication.
Do not push, deploy, or delete the previous public bundle unless the user
separately requests that external action.

## 12. Failure handling

- Preserve failed and partial release directories. Never delete, reuse, or
  rename them to imply success.
- A missing trial, changed checksum, partial artifact, calibration failure,
  convergence failure, solver warning, actuator clipping failure, contact,
  unexplained constraint row, or metrology failure makes that output
  non-authoritative.
- An incomplete trial may be deterministically restarted. A completed trial may
  not be overwritten.
- If an authoritative gate fails, diagnose it from the retained raw channels.
  Do not loosen a limit, rescale torque, edit the paper trace, or regenerate a
  receipt in place.
- A biologically different mean, sign, waveform, or phase is a model result, not
  an apparatus failure when all mechanical gates pass.

## 13. Completion report

Return a Markdown report containing all of the following:

- worker bundle ID, Docker image ID, dependency versions, compiled-model hash,
  source/runtime/reference/matrix hashes, and external artifact root;
- the live FlyWire manifest/status fetch timestamp or recorded endpoint failure,
  with FAFB-v783 and MANC provenance kept separate;
- all 12 run names, run IDs, statuses, 100 verified trials each, trial seeds, and
  wingbeat-phase modes;
- total, bilateral, lever-arm, axis, sign, unit, dynamic, and zero-load
  calibration results;
- body/head immobility, contact, solver, actuator, constraint, mechanical
  balance, aerodynamic closure, dual-meter, FIR, memory, and convergence gates;
- Figure 3a–c whole-fly metrics with simulation SEM and digitization uncertainty
  separate, plus unapplied diagnostic lag;
- left, right, wing-sum, non-wing residual, and aerodynamic diagnostic summaries
  explicitly labeled as simulation decompositions;
- every biological mismatch without fitted correction;
- `release_manifest.json`, all canonical artifact-manifest checksums, published
  replay bundle/index checksum, and the location of the retained previous index.

Do not call the release complete until the deep verifier reports all 1,200
trials and the completion report contains every item above.
