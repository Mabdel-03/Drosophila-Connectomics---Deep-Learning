# Paper tether-torque metrology

## Measurement chain

The authoritative `fixed-load-cell` topology compiles the FlyBody thorax/root
without a free joint. The root is therefore structurally welded to the world;
the head joints are omitted, while all six wing joints, actuators, tendons, and
ellipsoid-fluid aerodynamic geoms remain active. A site at the root-body origin
defines the vertical tether axis. MuJoCo force and torque sensors on that site
measure the parent-on-child support interaction in the site frame.

The recorded site rotation maps this wrench into engine coordinates: `+x` is
forward, `+y` is left, and `+z` is up. An injected-axis calibration confirms
that engine `+z` is a left-turn moment. Consequently:

```text
attempted_fly_yaw_engine = -support_on_fly_yaw
paper_positive_torque    = -attempted_fly_yaw_engine
                         =  support_on_fly_yaw
```

No amplitude, offset, latency, or transfer-function fit to the paper is used.
The compatibility field `fly_generated_yaw_moment_engine_Nm` is exactly the
negative support reaction and is not an independent moment estimator.

## Bilateral wing-root decomposition

The v3 paper topology adds massless force/torque sensor sites at the existing
`l_wing` and `r_wing` joint origins. These sites do not add bodies, joints,
mass, inertia, or control inputs. MuJoCo reports each parent-on-wing wrench in
its site frame. The runtime rotates it to engine coordinates, negates it to
obtain the wing-on-thorax wrench, and transports its moment to the canonical
tether origin:

```text
wing_on_thorax = -parent_on_wing
M_tether       = M_hinge + (hinge_position - tether_position) × F_wing
paper_torque   = -M_tether.z
```

The channel order is physical/anatomical left then right: `l_wing` is body
`+y`, and `r_wing` is body `-y`. This convention is independent of the raw
FAFB application-side labels. `wing_sum` is left plus right after applying the
same FIR to both sides; `nonwing_residual` is authoritative total minus that
filtered sum. The whole-fly root load cell remains authoritative and is never
replaced by the wing sum.

Three non-integrating shadow models receive the authoritative model's named
joint positions and velocities and run `mj_forward`: full fluid, left-wing
fluid only, and right-wing fluid only. Algebraic background removal yields the
two aerodynamic-only diagnostics. The probes cannot advance or modify the
authoritative state. Their bilateral sum must reconstruct full wing fluid yaw
within `max(0.1% of peak, 1e-10 N·m)`.

## Independent validator

`equality-reaction` retains the old free root and six-DoF weld only as a
validator. It selects the six constraint rows whose type is equality and whose
ID is the tether weld, zeros all other row forces, and invokes MuJoCo's
`mj_mulJacTVec` to reconstruct `Jᵀf`. A root-origin spatial Jacobian converts
that generalized reaction to a world wrench, so no `root_dof + 5` yaw
assumption remains. `comparison` advances both compiled models with identical
wing commands, motor seeds, clocks, and initial wingbeat phases; the fixed
load-cell channel remains authoritative.

## Timing and signal products

The canonical worker publishes at 10 kHz and performs four MuJoCo integration
steps per published sample. Every completed 25 µs state is retained as
`raw_internal` (40 kHz). The 20 kHz convergence run retains every 12.5 µs state
(80 kHz). Stimulus state and torque use the same completed-state timestamp.

Trials include an unrecorded 0.4 s synchronous pre-roll and a 0.4 s post-roll.
The padding supports an offline, odd-length Kaiser linear-phase FIR with delay
compensation. Its passband ends at 350 Hz, its stopband begins at 500 Hz, its
measured ripple is below 0.01 dB, and its measured stopband attenuation exceeds
100 dB at both raw rates. Exact 1 kHz sample-center timestamps form the
`paper_comparison` channel. `lowpass_10hz`, `lowpass_25hz`, `lowpass_50hz`, and
`wingbeat_averaged` are sensitivity products; none is selected based on paper
agreement, and none replaces raw samples.

The 100 initial wingbeat phases are uniformly stratified over a cycle and
permuted deterministically from the run seed. A fixed-phase run remains a
declared sensitivity control.

## Calibration and release gates

The content-hashed v3 calibration receipt includes both signs at ±1e-9, ±1e-8,
±1e-7, and ±2e-7 N·m on all three axes; cross-axis leakage; point forces at
±0.5 mm lever arms; 2.5 and 200 Hz sinusoidal injections; zero-load offset;
unit conversion; sensor transform; compiled-model fingerprint; worker
versions; timestep; and solver configuration. It requires ≤0.05% gain error,
≤1e-12 N·m zero yaw offset, ≤0.1% cross-axis leakage, ≤0.1% lever error, and
≤0.1° phase error at 2.5 Hz.

It also records both wing site transforms and positions, physical-side
registration, bilateral sensor-ID/address injection trials, exact side
isolation, signed `1e-7 N·m` yaw checks, and tether-origin lever-arm transport.
These sensor-buffer injections do not integrate the fly; they calibrate the
compiled sensor channels and every post-sensor coordinate/sign/unit operation.

The checked-in native receipt is
[`data/reference/paper_fgs/torque_calibration.fixed_load_cell.v2.json`](../data/reference/paper_fgs/torque_calibration.fixed_load_cell.v2.json).
Its internal SHA-256 covers every field except the digest itself. The receipt
is regenerated only in the pinned FlyGym/MuJoCo worker; each scientific run
also embeds its freshly computed receipt so a compiled-model change cannot
inherit this reference result silently.

A v3 browser replay is marked `authoritative_native_torque` only after total and
bilateral calibration,
apparatus, dual-meter, FIR, 40/80 kHz and 400/800 Hz convergence, 100-trial, and
paper-extraction checks pass. The rate-convergence checks apply independently
to total, left, right, and wing sum. A failure blocks that authority label but does not
delete results. Biological disagreement with the *Musca domestica* trace is
reported separately and is never repaired by rescaling the exploratory
Drosophila FAFB-v783/MANC/FlyBody model.

## Figure 3 comparison

`scripts/digitize_paper_figure3.py` verifies the supplied PDF SHA-256, renders
page 5 at 600 dpi, records panel crops and every visible axis tick, rectifies
the scan's small local geometric distortion through those ticks, and performs
an affine physical calibration in the rectified pixel frame. Each curve is
traced independently in both horizontal directions. Per-sample uncertainty is
the maximum of transformed half-line thickness, source-axis calibration RMS,
and half the disagreement between the traces.

Simulation means are interpolated to the paper timestamps without shifting or
warping. Outputs include overlays, residuals, pre/post/delta means, first
harmonics, RMSE, normalized RMSE, MAE, bias, correlation, simulation SEM, and
digitization uncertainty. The paper curve is explicitly a 100-sweep average
from one typical fly and has no reported SEM, acquisition rate, or filter.
Only the authoritative whole-fly tether channel is scored against Figure 3.
Per-wing and aerodynamic traces are simulation-only mechanical decompositions
because the paper contains no per-wing measurement.
