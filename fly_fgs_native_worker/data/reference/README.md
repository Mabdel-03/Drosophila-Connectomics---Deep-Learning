# Registered reference artifacts

## Paper fixed-load-cell calibration

`paper_fgs/torque_calibration.fixed_load_cell.v2.json` is the content-addressed
native calibration receipt for the structurally fixed FlyBody paper topology.
It records signed injections on all three axes, cross-axis leakage, ±0.5 mm
lever-arm trials, 2.5 and 200 Hz sinusoidal response, zero-load offset, exact
N·m↔dyne·cm conversion, the load-cell transform, solver configuration,
FlyGym/MuJoCo versions, and the compiled-model fingerprint. Its internal
receipt SHA-256 is
`c5ee51e95363eff95bff618d2783d20d15bd127ef3c99188f1a2b3f4858c07ea`.

This is an engineering metrology receipt for the implemented model. It is not
evidence that the model's neural, muscle, aerodynamic, or absolute biological
response matches the housefly data.

## FlyBody adapter compatibility

`flybody_analytic_wingbeat_smoke.v3.json` is the registered 5 ms execution trace from the
pinned Python 3.12 worker with FlyGym 2.1.0 and MuJoCo 3.9.0. It uses the
beat-wrapped analytic fallback wingbeat formula derived from upstream FlyBody at 218 Hz and a
0.05 ms physics timestep, with the released task's 0.2 ms controller clock
advanced before zero-order hold. Versions 1 and 2 are retained as historical
fixtures; v3 is the only version registered by the current benchmark.

The fixture's `0.05 ms` physics step is a compatibility-regression setting;
operational FlyBody episodes default to `0.1 ms`. Both keep the reviewed
`0.2 ms` prescribed control clock distinct from physics and neural clocks.

Current SHA-256:
`9e58e24af449d1744fe11a52f3ce8e49d63de3dc3d17ead557526c667cee7acc`

The exact v3 contract fingerprints:

- FlyGym/MuJoCo versions and dependency-record hashes;
- compiled model hash, mass, integrator, gravity, air properties, timestep, and
  counts (`68` bodies, `88` geoms, `103` joints, `85` meshes, `109` qpos,
  `108` qvel, `6` actuators, and `8` tendons);
- reviewed left/right yaw-roll-pitch DoF order and two wing-fluid geom names;
- beat-wrapped fallback waveform, `0.2 ms` advance-then-hold controller, four
  held `0.05 ms` samples, and exact first-command hold behavior;
- collision-disabled aerodynamic proxy geoms, `legs_only` ground-contact
  topology, and `released_policy_topology_equivalent: false`; this topology
  fingerprint is distinct from operational episode telemetry, whose counts
  describe pre-integration transition contact points rather than contact force;
- root/thorax reference and interval-applied root-fluid sample semantics.

Tolerance-bounded numeric summaries cover actuator torque, measured wing
excursion, root-fluid impulse, terminal root position, and quaternion. The
trace demonstrates exact-stack compatibility, six-axis actuation, passive
tendons, corrected fluid units, and finite whole-body dynamics. It is not a
reproduction of the released learned straight-flight or saccade policies, is
not driven by the neural/muscle scaffold, and remains
`exploratory; calibration: none`.

The trace uses the FlyBody free-joint root/thorax frame for body pose. It must
not be interpreted as whole-fly center of mass, desired virtual-hinge motion as
measured wing kinematics, or root-total fluid wrench as a per-wing force split.
The operational episode adapter records those references separately; this
standalone compatibility fixture does not promote the virtual hinge or any
neuromuscular parameter.

The historical v10 operational runs used FlyBody adapter v1.5.0 at source artifact
`sha256:239cf9b7c396ec75c5b344154c2c8ece54f07da4acd0ff97108693c596fd2098`
from model registry
`sha256:30381fcff006a3c6d1b43e010559b93b33db738097ca57b0ba274faf1dd66795`
in worker image
`sha256:3222b2d946b56bb039f0270000f3933dd9387ea14fa274f1b475fcbc70d9b4a6`.
Their 0.1 ms compiled-model fingerprint is
`sha256:bf32d4f573265399ebfc87a6dba18130f1169f23f16a3f1c2647368cf747ebe2`.
It is intentionally distinct from this compatibility fixture's
`sha256:f358f373862740cbcac66cdf3bafc4856992f906ea357c018d853f8a5989d2b0`
because the compiled fingerprint includes the fixture's 0.05 ms timestep.
These hashes remain historical fixture receipts; they are not the worker,
registry, report, or canonical-online artifact receipts for v1.18.0. The latter
must be generated from the final compiled run rather than copied from this
reference fixture.

Regenerate it with:

```bash
fly-s2b flybody-smoke --mode analytic-wingbeat --duration-s 0.005 \
  --output /tmp/flybody-smoke-raw.json
python scripts/register_flybody_smoke_fixture.py \
  --input /tmp/flybody-smoke-raw.json \
  --output data/reference/flybody_analytic_wingbeat_smoke.v3.json
```

The raw trace must come from the pinned worker. Registration recomputes
trace-derived summaries and adds the comparison contract; it does not make the
fixture a released-policy, calibration, or biological baseline.

## Female BANC/FANC wing-pathway evidence

`banc_fanc_wing_pathway_evidence.v1.json` is a normalized, content-addressed
structural snapshot built from exact-source BANC v888 paper-v2 aggregate edges,
BANC metadata, FANC materialization 840 wing premotor/MN tables, and the pinned
FANC DN annotation crosswalk.

Current SHA-256:
`91763c068c74520d9c9f1aadd0ef80a295bae01cefc95c9091014eb0a0859c21`

The registered structural invariants include:

- four BANC NOD1→contralateral-DNp26 edges totaling `487` raw synapses;
- `62` proofread BANC wing MNs, 31 per dataset side, with four mappings
  (`PSn_u` and `MNxm01` bilaterally) explicitly unresolved;
- 18 BANC DNp26→wing-MN edges totaling **`129`** raw synapses, correcting the
  earlier provisional count of 131;
- the one-sided FANC `1,784 × 29` wing premotor matrix: `7,289` nonzero pairs
  and `144,668` structural synapses after its published ≥3-pair threshold; and
- two explicitly annotated FANC DNp26 rows totaling `153` synapses onto named
  wing-MN columns.

BANC and FANC IDs remain separate. The fixture contains zero direct
individual-neuron joins, no premotor matching was performed, the premotor
identity crosswalk is `incomplete`, and no topological candidate was promoted
to a match. Counts are structural topology only: physiological sign, weight,
gain, and a complete DN→identified-premotor→MN→muscle route remain unresolved.
Regeneration and source-path requirements are documented in
[`../../scripts/README.md`](../../scripts/README.md).

## Canonical fly-FGS visual/circuit capture

`fly_fgs/source_manifest.v1.json` and
`fly_fgs/fixed_step_capture.v1.json.gz` lock the canonical upstream
[`/fly-fgs`](http://54.160.228.98/fly-fgs/) scene and neural circuit. The source
manifest SHA-256 is
`fef59e4b8abe5216081b6d64decd0d1775454d06e7631bb9af8ae409c1cd05c3`;
the deterministic-gzip capture SHA-256 is
`5612cc9c2218a917bf2ad939d6aaa2a8402db5f11fd2de36cd3168d9198cbd83`.
The manifest separately hashes the deployed page, fixed-step engine, circuit
bundle, and excluded `wing_dns` asset.

The registered program performs 48 static-scene pre-roll steps (`0.24 s`), then
stores 100 samples at `0.005 s` over half-open interval `[0, 0.5 s)`. The cell
axis has 1,684 entries: 1,457 T4a, 219 LLPC1, two vCH, two DCH, and four NOD1.
Exactly 1,441 T4a cells have registered retinotopic normalized-luminance input.
The capture has no T5, photoreceptor, or lamina cells. Because sample 0 is the
post-pre-roll state rather than the end of a published source interval, the
pipeline exports 99 causal `RetinalFrame`s for source indices 1–99.

The four exact NOD1 voltage arrays are the only `CircuitOutputTrace` inputs to
the motor bridge. Full voltage/activity state, the T4a retinal grid, pooled
T4a/LLPC1/vCH/DCH/NOD1 traces, and circuit topology are retained for replay and
audit only. Raw bundle `L`/`R` is an application compatibility label, not an
anatomical side; FAFB anatomy requires the separately checked soma-x rule.
Structural event counts are not physiological synaptic weights, and the bundle
permits synthetic fallback without recovering per-event origin.

The source page's `SCALE_M`, `FG_DN`, `DN_WMAX`, DN gain/rate logic, muscle
activation, wing amplitudes/angles, yaw, and toy body fields are forbidden at
intake. They are source receipts only and cannot become motor, muscle, hinge,
aerodynamic, or body state. The registered frozen-integrity evaluator checks
these exact bytes and projections; it does not constitute a second execution.
Run the Node capture twice to test runtime reproducibility. Even a passing
fixed-fixture or native-FlyBody integration gate establishes deterministic
software propagation only, not circuit physiology, stable flight, or
biological validation.

## Historical NOD1 browser/Python numerical parity

`nod1_browser_python_parity.v1.json.gz` is a deterministic-gzip capture of a
real Chromium Web Worker execution and the replay of the exact saved input in
the local Python solver. The fixture pins the deployed index, bundled main
JavaScript/CSS, worker asset, circuit response, complete source/morphology
inventory, materialization 783, four FlyWire root IDs as decimal strings, the
1,208-cell/5,188-edge circuit, and all 53,715 prepared events.

Current compressed SHA-256:
`a76794e630533822468971cbdbe2164a3d1ddce8a8a38811c9b9c1ad5b97a8a4`

For 400 scalar comparisons (100 time samples × four registered readouts), the
captured maximum readout difference is
`5.34057617157524e-08 V` against the preregistered `5e-05 V` limit; the maximum
summary relative difference is `7.318367272088506e-06` against the `0.01`
limit. The evaluator recomputes those metrics from the stored arrays and
rechecks all source receipts instead of trusting the fixture's stored pass
flag.

This is a frozen cross-runtime numerical regression only and is retained as a
historical circuit fixture now that fly-FGS is canonical. It does not validate
the circuit's biological accuracy, laterality, physiological weights,
visual-to-motor interpretation, muscles, or flight behavior. Regeneration
instructions and capture constraints are in [`../../scripts/README.md`](../../scripts/README.md).

The same exact compressed bytes are the registered input to
`pipeline.registered_nod1_to_flybody_vertical_slice` v1.2.0. That scheduled case
consumes all 100 samples across the complete 0.5 s capture and tests deterministic
software propagation plus a typed `MN-iv1-right` cut through native FlyBody. It
does not turn this numerical fixture into retinal evidence, a biological motor
mapping, or a stable-flight baseline.

In the final v10 catalog this fixture backs episode `frozen_browser_nod1`,
immutable run `nod1_visual_circuit_to_flight-61747ff116b3656c6fc8`, and replay
SHA-256 `d57558400e00e00d2e12ea3111705651dbec626016a8f22a73ecc0811166cc1b`.
Its run manifest has SHA-256
`ab346d45646340494a02c7f57542c98cc90abaf4be9371ba19e9b75a5aa2602d`
and contains 176 logical arrays in 469 chunks. The similarly prefixed run ending
`44c6451a9cf76d44a903` is the separate analytic-retinal/reduced-NOD1 episode and
does not consume this registered fixture.
