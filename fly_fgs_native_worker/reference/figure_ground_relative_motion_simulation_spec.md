# Figure-Ground Relative-Motion Experiment: Simulation Specification

## Purpose

This document is implementation context for recreating the physical stimulus, tether mechanics, and measured outputs of the figure-ground relative-motion experiment described in Reichardt, Poggio, and Hausen (1983).

The target is the original open-loop behavioral assay. A fly is held fixed at the center of independently moving visual layers, the wings remain active, and attempted yaw torque is measured while the fly is prevented from turning. An optional, separate intracellular-recording variant is specified near the end.

Biological preparation, husbandry, neural-circuit theory, and free-flight behavior are outside scope. Do not add body motion, closed-loop visual feedback, or unreported apparatus dynamics to the canonical replication.

Primary sources:

- [1983 experiment and circuitry paper](https://link.springer.com/article/10.1007/BF00595226), the main source for this specification.
- [1979 Part I experimental paper](https://link.springer.com/article/10.1007/BF00337434), the source of several time-averaged protocols reproduced in the 1983 paper.
- [1989 primary follow-up](https://link.springer.com/article/10.1007/BF00200799) and its [full primary-paper PDF](https://core.ac.uk/download/pdf/15980582.pdf), used only where explicitly labeled as supporting later apparatus information.

## Provenance labels

Every parameter that is not an ordinary mathematical conversion has one of these labels:

| Label | Meaning |
|---|---|
| `R83` | Explicitly reported in the 1983 paper. |
| `R79` | Explicitly reported in the 1979 protocol as reproduced or cited by the 1983 paper. |
| `S89` | Reported in the later 1989 primary paper. It is supporting evidence, not proof of the 1983 value. |
| `DERIVED` | Computed exactly from reported values. |
| `DEFAULT` | Recommended implementation choice filling an unreported detail. It must remain configurable. |
| `UNKNOWN` | The papers do not determine the value. Do not silently invent or hard-code it as historical fact. |
| `FIGURE_READ` | Read from plotted markers or traces rather than stated numerically in prose or a caption. |

Normative words are used as follows:

- `MUST`: required for the canonical replication.
- `SHOULD`: strongly recommended for reproducibility or fidelity.
- `MAY`: optional extension or ambiguity-control run.

## 1. Minimal faithful replication

Implement this configuration first. It is the main dynamic behavioral protocol, corresponding to Figure 3a.

| Component | Canonical value | Provenance |
|---|---:|---|
| Assay topology | Open-loop tethered flight | `R83` |
| Body | Fixed in position and orientation | `R83` |
| Head | Rigidly fixed relative to thorax | `R83` |
| Active mechanics | Wings generate flight forces and moments | `R83` |
| Required output | Attempted yaw torque about the vertical body axis | `R83` |
| Figure | Vertical random-dot stripe | `R83` |
| Figure center | `+30 deg` azimuth | `R83` |
| Figure width | `12 deg` | `R83` |
| Ground | Full `360 deg` random-dot panorama | `R83` |
| Texture cells | Binary black/white, `3 deg x 3 deg` | `R83` |
| Mean surface luminance | Approximately `700 cd/m^2` | `R83` |
| Figure amplitude | `+/-5 deg`, peak amplitude | `R83` |
| Ground amplitude | `+/-5 deg`, peak amplitude | `R83` |
| Frequency | `2.5 Hz` | `R83` |
| Initial relative phase | `0 deg` | `R83` |
| Phase transition begins | `t = 0.4 s` | `R83` |
| Target relative phase | `+90 deg`, figure relative to ground | `R83` |
| Displayed duration | `2.0 s` | `R83` |
| Repetitions | `100`, stimulus-phase locked | `R83` |
| Torque sign | Positive is attempted right turn | `R83` |
| Historical torque unit | `dyne cm` | `R83` |
| SI conversion | `1 dyne cm = 1e-7 N m` | `DERIVED` |

The simulation MUST prescribe the visual motion independently of fly output. The body MUST remain fixed. The primary returned data MUST be the yaw-torque time series, not body yaw or yaw velocity.

## 2. Coordinate system and signs

### 2.1 Paper angular convention

Use horizontal angular coordinate `psi`:

- `psi = 0 deg` is the forward symmetry direction between the eyes. `R83`
- `psi > 0` is toward the right visual field. `R83`
- `psi < 0` is toward the left visual field. `R83`
- Increasing `psi` is clockwise when viewed from above.
- On the right eye, increasing stimulus angle is progressive motion; decreasing angle is regressive motion. The paper illustrates this as motion from `-5 deg` to `+5 deg` and the reverse. `R83`

Use the half-open wrap interval `[-180 deg, 180 deg)` for stored azimuths. Use an unwrapped angle internally when differentiating a moving layer so the wrap seam cannot produce false velocities.

### 2.2 Recommended physics-engine frame

Use a right-handed frame:

- `+x`: body forward, corresponding to `psi = 0 deg`.
- `+y`: body left.
- `+z`: upward along the body vertical axis and cylinder axis.

A ray at paper azimuth `psi` has horizontal direction

```text
d(psi) = [cos(psi), -sin(psi), 0]
```

where the trigonometric functions receive radians in code.

Positive right-handed torque about `+z` is a left turn. The paper-positive yaw torque is therefore

```text
tau_paper = -tau_fly_z_engine
```

where `tau_fly_z_engine` is the moment exerted by the fly on its body in the engine frame.

If the engine instead reports the support-on-fly constraint reaction, then under a static, well-centered constraint:

```text
tau_paper ~= +tau_support_on_fly_z_engine
```

Physics APIs differ in which body receives the reported reaction. A sign unit test is mandatory.

### 2.3 Attraction-normalized torque

Always preserve raw paper-sign torque. For single-figure comparisons, a second derived channel MAY normalize the sign so positive means attraction toward either a right-side or left-side figure:

```text
tau_attraction = sign(figure_mean_azimuth_deg) * tau_paper
```

Do not use this transform for bilateral two-figure trials, and never overwrite the raw channel.

## 3. Physical and mechanical setup

### 3.1 Assay topology

The canonical assay is open loop. `R83`

- The fly is suspended at the common center of two concentric, vertical visual cylinders.
- The body is fixed in space.
- The head is rigid relative to the thorax. A stationary pattern is therefore stationary on the retina.
- The wings continue flying and generate aerodynamic forces and moments.
- A torque compensator balances and measures flight torque about the vertical body axis.
- The compensator electronics output a voltage proportional to compensated torque.
- The fly does not control either cylinder in this experiment.

The visual cylinders provide optical stimulation only. They MUST NOT collide with the fly, drag the fly, impose body rotation, create programmed airflow, or apply a mechanical torque to the body.

### 3.2 Canonical constraint model

The simulation MUST:

1. Place the body reference point at the common cylinder center.
2. Align the body vertical axis with the cylinder rotation axis.
3. Constrain all body translations.
4. Constrain body roll, pitch, and yaw.
5. Keep the head-to-thorax transform constant.
6. Leave wing joints and the wing/aerodynamic model active.
7. Record the vertical-axis constraint reaction and/or independently summed wing-generated yaw moment.
8. Keep figure and ground motion independent of every fly output.

An ideal rigid constraint is the canonical `DEFAULT`. The paper does not report torque-meter stiffness, damping, inertia, controller gains, bandwidth, residual angular compliance, analog filtering, or latency. A compliant torsional spring-damper MAY be added only as a separate sensitivity analysis.

### 3.3 Force and moment accounting

When wings are physically resolved, compute the fly-generated yaw moment as

```text
tau_fly_z = z_hat dot sum_i((r_i - r_axis) cross F_i + M_i)
```

where:

- `r_axis` lies on the vertical body/tether axis.
- `F_i` is an aerodynamic force on a wing element or rigid wing.
- `M_i` is any directly modeled aerodynamic moment.

With zero body angular acceleration:

```text
tau_fly_z + tau_support_on_fly_z + tau_other_z = 0
```

Gravity and vertical support should make no appreciable yaw moment in a centered setup. Calibrate or log any non-wing yaw bias separately.

If the fly agent does not resolve wings, a physical yaw-moment command MAY be used as a surrogate output. Label it `surrogate_yaw_moment`. Do not substitute action probability, differential-wing command, desired yaw rate, or body yaw angle for physical torque unless an explicit, validated conversion produces units of `N m`.

### 3.4 Historically reported actuator arrangement

The physical apparatus used two independently controlled cylinders. `R83`

- The inner cylinder was directly driven by one servomotor.
- The outer cylinder was belt-driven by a second servomotor.
- Each shaft carried a ring potentiometer used for position feedback.
- Cylinder angular displacement was proportional to command voltage under position control.
- The motors are described as `400 Hz servomotors`.
- Behavioral torque was accumulated online with a Princeton Signal Averager Model 4202. The instrument name does not determine the unreported sampling rate or analog filter.

Do not reinterpret `400 Hz` as a display refresh rate, data-acquisition rate, simulation rate, or measured closed-loop bandwidth. Those rates are `UNKNOWN`.

The virtual implementation SHOULD prescribe analytic angular trajectories and record both commanded and realized angles. A hardware-faithful actuator model is optional because its inertia, gearing, backlash, tracking error, and transient response are unreported.

### 3.5 Cylinder radii and physical sizes

The 1983 paper does not report the radii, separation, height, or wall material of the figure-ground cylinders. `UNKNOWN`

For an angular renderer, radius has no effect on the retinal azimuth of an observer exactly at the common center. Angular geometry is therefore authoritative.

If physical radii are required, use the following configurable starting values only as a supporting later-apparatus inference:

```text
figure_layer_radius_m = 0.035
ground_layer_radius_m = 0.036
```

These values are `DEFAULT` based on `S89`, not `R83`. The later paper reports visual elements at radii of `35 mm` and `36 mm`, but does not explicitly assign those radii to the 1983 figure-ground cylinders.

Useful derived sizes are:

| Angular size | At 35 mm radius | At 36 mm radius | Provenance |
|---:|---:|---:|---|
| `3 deg` arc | `1.833 mm` | `1.885 mm` | `DERIVED` |
| `10 deg` arc | `6.109 mm` | `6.283 mm` | `DERIVED` |
| `12 deg` arc | `7.330 mm` | `7.540 mm` | `DERIVED` |
| `360 deg` circumference | `219.91 mm` | `226.19 mm` | `DERIVED` |

For the main stimulus at `+/-5 deg` and `2.5 Hz`, a `35 mm` radius gives `3.054 mm` peak surface displacement and `47.98 mm/s` peak tangential speed. A `36 mm` radius gives `3.142 mm` and `49.35 mm/s`. These are derived from assumed radii and are not primary stimulus requirements.

### 3.6 Air and gravity

- `DEFAULT`: Use quiescent air and the ordinary gravity settings required by the wing/body model.
- The tether or body constraint supports the body and cancels translations and rotations.
- Do not make cylinder rotation drive ambient airflow.
- Temperature, pressure, humidity, air density, and ambient flow are `UNKNOWN`.

## 4. Visual scene geometry and rendering

### 4.1 Observer placement

The visual origin MUST lie on the cylinder axis at the center of rotation. `R83`

If two eye cameras or compound-eye samplers are used, place them according to the fly model, but preserve the paper angular convention at the body center. Keep body and head transforms constant during the trial.

### 4.2 Layer order

Use this conceptual radial order:

1. Fly/eye at the origin.
2. Figure layer on the inner cylinder.
3. Optional stationary screens or masks.
4. Ground texture on the outer cylinder.

Within the figure mask, the figure MUST visually replace or occlude the ground. Outside the mask, the ground MUST remain visible. Treat unused portions of the inner layer as optically transparent. This transparency is an implementation model of the required retinal result; cylinder material is `UNKNOWN`.

### 4.3 Standard figure and ground

The main figure is a vertical stripe with:

```text
mean_center_azimuth_deg = +30
angular_width_deg = 12
```

The rest-position horizontal footprint is `[+24 deg, +36 deg]`. Under `+/-5 deg` motion, its outer envelope is `[+19 deg, +41 deg]`. `DERIVED`

The main ground covers the full `360 deg` panorama. `R83`

The figure's exact vertical height and the cylinder's vertical extent are `UNKNOWN`. The canonical `DEFAULT` is:

- Make the figure span the entire rendered vertical field.
- Make the ground extend beyond every eye ray by a safety margin.
- Ensure no top or bottom cylinder edge is visible.

### 4.4 Angular ray-to-cylinder mapping

For an ideal cylindrical renderer, use paper azimuth

```text
psi_deg = wrap180(rad2deg(atan2(-ray_y, ray_x)))
```

and elevation

```text
elevation_deg = rad2deg(atan2(ray_z, hypot(ray_x, ray_y)))
```

The visual textures are defined in angular coordinates. This avoids making an unreported radius affect texel size.

### 4.5 No unintended visual cues

The canonical renderer MUST avoid cues absent from the intended stimulus:

- no cast shadows;
- no specular highlights;
- no texture-dependent surface normals;
- no automatic exposure;
- no tone-mapping changes over time;
- no motion blur;
- no temporal antialiasing history;
- no visible cylinder seam;
- no visible top or bottom edge;
- no depth-of-field change during motion;
- no lighting modulation caused by cylinder angle.

Use emissive or uniformly illuminated Lambertian surfaces. If a photoreceptor optics model supplies blur, render sharp binary scene geometry and let that model perform the blur. Do not add arbitrary display filtering and then also blur in the eye model.

## 5. Random-dot pattern generator

### 5.1 Reported pattern

Figure and ground use binary black/white random-dot patterns, called Julesz patterns. Each pixel subtends:

```text
pixel_width_deg = 3
pixel_height_deg = 3
```

`R83`

A `12 deg` figure is therefore four horizontal texture cells wide when its mask and the master grid are aligned. A full `360 deg` ground contains 120 horizontal cells. `DERIVED`

### 5.2 Canonical registered-copy construction

The papers describe figure and ground using language including "same texture," "identical texture," and "statistically equivalent." The fact that the boundary can disappear during synchronous motion most strongly supports a registered-copy implementation, but the exact physical pixel maps are unavailable.

Use `registered_copy` as the canonical `DEFAULT`:

1. Generate one binary master texture `T_ground(psi, elevation)` on a `3 deg x 3 deg` angular grid.
2. At rest, define the figure as an opaque `12 deg` mask that reveals a copied patch from that same master texture.
3. Align the figure and ground texture coordinates when their angular motions are synchronous.
4. Move the figure mask and its copied texture rigidly with the figure layer.
5. When figure and ground have equal positions, pixels inside and outside the figure boundary match in angular coordinates.

Also implement `independent_matched_statistics` as an ambiguity-control mode. It uses independent figure and ground random seeds with the same cell size, luminance levels, and black probability.

The relationship mode and all seeds MUST be stored in metadata.

### 5.3 Texture grid definition

Recommended deterministic implementation:

```text
ground_col = floor(wrap360(texture_azimuth_deg - grid_origin_azimuth_deg) / 3)
ground_row = floor((elevation_deg - grid_origin_elevation_deg) / 3)
value = texture_seeded[ground_row, ground_col]
```

Requirements:

- The horizontal grid MUST wrap seamlessly after 120 columns.
- The master texture MUST be static in the local coordinates of its cylinder.
- Texture offsets MUST be moved by geometry or angular coordinate transformation, not regenerated each frame.
- Use nearest-neighbor lookup at the scene-texture level.
- Store `grid_origin_azimuth_deg` and `grid_origin_elevation_deg`.
- Store the generated bitmap or a versioned generator name plus exact seed.

The historical grid origin, seam location, random-number generator, random seed, and black-pixel probability are `UNKNOWN`.

Use `black_probability = 0.5` as a configurable `DEFAULT`. A sensitivity run SHOULD verify that results are not an artifact of the particular texture seed.

### 5.4 Precise compositing rule

Let:

- `alpha_g(t)` be ground angular displacement.
- `alpha_f(t)` be figure angular displacement.
- `psi_f0` be figure mean center.
- `W_f` be figure width.
- `T_g` be the master ground texture.
- `T_f` be either the same master texture or an independent texture.

For each ray azimuth `psi`:

```text
ground_coord = wrap360(psi - alpha_g(t))
figure_local = wrap180(psi - (psi_f0 + alpha_f(t)))

if abs(figure_local) <= W_f / 2 and ray_elevation is inside figure vertical extent:
    figure_coord = wrap360(psi - alpha_f(t))
    luminance = T_f(figure_coord, elevation)
else:
    luminance = T_g(ground_coord, elevation)
```

In `registered_copy` mode, set `T_f = T_g`. When `alpha_f = alpha_g`, this rule makes the texture continuous across the moving figure boundary.

Define one edge-inclusion rule, such as left-inclusive/right-exclusive, and use it consistently. Do not allow overlapping pixels or a one-pixel transparent gap along figure edges.

### 5.5 Luminance and contrast

The reported average brightness at the cylinder surfaces is approximately:

```text
mean_luminance_cd_m2 = 700
```

`R83`

The 1983 paper does not state black luminance, white luminance, or contrast. A later primary paper from the same laboratory reports `78%` contrast for `3 deg x 3 deg` random-dot textures at approximately the same mean luminance. `S89`

Recommended supporting `DEFAULT`:

```text
contrast_definition = Michelson
michelson_contrast = 0.78
white_probability = 0.5
mean_luminance_cd_m2 = 700
```

Assuming the arithmetic midpoint of black and white luminance is `700 cd/m^2`:

```text
L_white = 1246 cd/m^2
L_black = 154 cd/m^2
```

`DERIVED` from `S89` plus the stated assumptions.

If absolute luminance cannot be represented, preserve the ratio

```text
L_black / L_white = (1 - 0.78) / (1 + 0.78) = 0.1235955
```

and document the display-to-luminance mapping. Gamma, spectral power distribution, surface reflectance, and residual lamp flicker are `UNKNOWN`.

The physical setup used four DC-current fluorescent ring bulbs. `R83` The simulation SHOULD emulate the resulting uniform, stable surface luminance rather than modeling lamp geometry unless lamp nonuniformity is explicitly under study.

## 6. Motion generator

### 6.1 Exact steady-state motion

The stimulus is sinusoidal in angular position. `R83`

Define the ground displacement and figure center as:

```text
omega = 2 * pi * frequency_hz

alpha_g(t) = A_g * sin(omega * t + phi_g)
psi_f(t)   = psi_f0 + A_f * sin(omega * t + phi_f)
relative_phase = phi_f - phi_g
```

Set `phi_g = 0` unless a protocol states otherwise. Angles in the configuration are degrees; convert once at the numerical boundary and use radians internally.

`A_f` and `A_g` are peak amplitudes, not peak-to-peak values.

For the standard stimulus:

```text
frequency_hz = 2.5
period_s = 0.4
omega_rad_s = 15.7079632679
A_f_deg = 5
A_g_deg = 5
```

### 6.2 Velocity and acceleration

For either sinusoidal layer:

```text
angular_velocity(t) = omega * A * cos(omega * t + phi)
angular_acceleration(t) = -omega^2 * A * sin(omega * t + phi)
```

| Peak amplitude | Peak speed | Peak acceleration | Provenance |
|---:|---:|---:|---|
| `0.5 deg` | `7.854 deg/s` | `123.37 deg/s^2` | `DERIVED` |
| `1 deg` | `15.708 deg/s` | `246.74 deg/s^2` | `DERIVED` |
| `2.5 deg` | `39.270 deg/s` | `616.85 deg/s^2` | `DERIVED` |
| `3 deg` | `47.124 deg/s` | `740.22 deg/s^2` | `DERIVED` |
| `5 deg` | `78.540 deg/s` | `1233.70 deg/s^2` | `DERIVED` |
| `7 deg` | `109.956 deg/s` | `1727.18 deg/s^2` | `DERIVED` |
| `7.5 deg` | `117.810 deg/s` | `1850.55 deg/s^2` | `DERIVED` |
| `10 deg` | `157.080 deg/s` | `2467.40 deg/s^2` | `DERIVED` |

### 6.3 Relative motion checks

For equal amplitudes `A` and phase difference `Phi`:

```text
delta(t) = alpha_f(t) - alpha_g(t)
relative_displacement_amplitude = 2 * A * abs(sin(Phi / 2))
peak_relative_speed = omega * relative_displacement_amplitude
```

For `A = 5 deg`, `f = 2.5 Hz`:

| Relative phase | Relative displacement amplitude | Peak relative speed |
|---:|---:|---:|
| `0 deg` | `0 deg` | `0 deg/s` |
| `90 deg` | `7.0711 deg` | `111.072 deg/s` |
| `180 deg` | `10 deg` | `157.080 deg/s` |
| `270 deg` | `7.0711 deg` | `111.072 deg/s` |

For unequal amplitudes:

```text
relative_amplitude = sqrt(A_f^2 + A_g^2 - 2*A_f*A_g*cos(Phi))
```

These equations are mandatory unit tests for the realized-angle traces.

### 6.4 Phase transitions

The paper gives transition start times and target phases, but it does not report the exact behavioral-actuator trajectory, acceleration limit, position error, or whether plotted position traces are command or measured ring-potentiometer signals. `UNKNOWN`

The plotted transitions are smooth. Several matching model captions say a transition beginning at `0.4 s` is complete at `0.8 s`, one stimulus period later. This supports, but does not prove, a one-period behavioral transition.

Implement two explicit modes:

1. `phase_command_step`: change the phase command at the reported switch time and pass it through an explicitly configured servo model. Do not teleport rendered geometry.
2. `one_period_phase_ramp`: continuously change relative phase from the old value to the signed target over `0.4 s`. Use this as the canonical `DEFAULT` when no servo model is available.

For the ramp mode:

```text
u = clamp((t - transition_start_s) / 0.4, 0, 1)
relative_phase(t) = old_phase + u * signed_phase_delta
```

Use signed targets:

- paper `90 deg` condition: `+90 deg`;
- paper `270 deg` condition: `-90 deg`;
- paper `180 deg` condition: `+180 deg`.

A smoothstep ramp MAY be tested, but it changes the phase trajectory and must be labeled separately. Never silently smooth or interpolate.

Always store:

- requested target phase;
- signed phase delta;
- transition start and duration;
- commanded figure and ground angles;
- realized figure and ground angles.

Analyze steady-state phase metrics outside the transition interval. Retain the transition data because the original paper emphasizes response dynamics.

### 6.5 Simulation and rendering rates

The historical stimulus update rate, acquisition sample rate, and filters are unreported. `UNKNOWN`

Recommended modern `DEFAULT` values:

```text
physics_rate_hz >= 2000
stimulus_and_eye_update_rate_hz >= 400
raw_torque_logging_rate_hz >= 1000
```

These are engineering choices, not historical values. Analytic motion evaluated at each sample is preferred. If the engine cannot meet them, run a convergence study and show that the torque and retinal-input metrics change negligibly when rates are doubled.

## 7. Canonical behavioral protocol

### 7.1 Main `0 to 90 deg` trial, Figure 3a

```text
trial_duration_s = 2.0
figure_type = textured_vertical_stripe
figure_mean_azimuth_deg = +30
figure_width_deg = 12
ground_extent_deg = 360
figure_amplitude_deg = 5
ground_amplitude_deg = 5
frequency_hz = 2.5
initial_relative_phase_deg = 0
transition_start_s = 0.4
target_relative_phase_deg = 90
transition_duration_s = 0.4  # DEFAULT, inferred rather than directly reported
repetitions = 100
```

Timeline:

| Time | State |
|---:|---|
| `0.0 to 0.4 s` | Figure and ground oscillate synchronously. |
| `0.4 to 0.8 s` | Canonical inferred transition from `0` to `+90 deg`. |
| `0.8 to 2.0 s` | Three full periods at steady `+90 deg` relative phase. |

The paper's expected response is initially oscillatory around approximately zero mean. After relative motion begins, mean torque becomes positive for the right-side figure and a structured `2.5 Hz` waveform appears. This is a behavioral-model validation target, not a renderer-only acceptance condition.

### 7.2 Main `0 to 270 deg` trial, Figure 3b

Use the same settings as Figure 3a, except:

```text
target_relative_phase_deg = 270
signed_phase_delta_deg = -90
```

The expected mean is again positive, while the within-period waveform is approximately reversed relative to the `90 deg` condition.

### 7.3 Main `0 to 180 deg` trial, Figure 3c

```text
trial_duration_s = 4.0
figure_amplitude_deg = 7.5
ground_amplitude_deg = 7.5
frequency_hz = 2.5
transition_start_s = 1.2
target_relative_phase_deg = 180
transition_duration_s = 0.4  # DEFAULT
repetitions = 100
```

The reported response remains oscillatory with approximately zero time average and no clear response-phase shift after the stimulus transition.

## 8. Additional stimulus protocols

These are valuable cross-checks after the canonical Figure 3 implementation works.

### 8.1 Black-stripe phase sweep, Figure 2

| Parameter | Value | Provenance |
|---|---:|---|
| Figure | Solid black vertical stripe | `R79`/`R83` |
| Figure width | `3 deg` | `R79`/`R83` |
| Vertical placement | Lower part of panorama | `R79`/`R83`; exact height `UNKNOWN` |
| Mean azimuth | Mirrored trials at `+30 deg` and `-30 deg` | `R79`/`R83` |
| Ground | Movable random-dot panorama | `R79`/`R83` |
| Stationary screen | White, `12 deg` wide, between stripe and ground | `R79`/`R83` |
| Figure amplitude | `+/-1 deg` | `R79`/`R83` |
| Ground amplitude | `+/-1 deg` | `R79`/`R83` |
| Frequency | `2.5 Hz` | `R79`/`R83` |
| Relative phases | `0, 30, ..., 360 deg` | `FIGURE_READ` |
| Output | Time-averaged yaw torque | `R79`/`R83` |
| Aggregation | 10 independent flies per point; SEM | `R79`/`R83` |

Render the stationary white screen as a fixed `12 deg` angular patch behind the moving `3 deg` stripe and in front of the random ground. Its exact radial distance, center, height, and boundary construction are `UNKNOWN`. Centering it on the stripe's mean azimuth and spanning the same configured vertical range is the `DEFAULT`.

Expected response: strong attraction near `90 deg` and `270 deg`, near-zero response at `0 deg` and `180 deg`.

### 8.2 Contralateral half-ground, Figure 4

```text
figure_center_deg = +30
figure_width_deg = 12
ground_azimuth_interval_deg = [-180, 0]
edge_screen_width_deg = 10
figure_amplitude_deg = 5
ground_amplitude_deg = 5
frequency_hz = 2.5
initial_phase_deg = 0
transition_start_s = 1.2
target_phase_deg = 90
trial_duration_s = 4.0
repetitions = 100
```

The `180 deg` moving ground lies in the visual hemifield opposite the figure. Both moving ground boundaries are covered by stationary `10 deg` screens so the moving edge itself does not become a separate cue. The screen color and exact placement are `UNKNOWN`; stationary uniform masks centered on the two boundary azimuths are the `DEFAULT`.

Use a stationary uniform field at mean luminance outside the half-ground and figure masks. This is a `DEFAULT`; the exact appearance of the unused visual field is not reported.

Expected response: the raw mean is negative because the half ground dominates, but it becomes less negative after the relative-phase transition, indicating attraction toward the right-side figure. Use baseline-subtracted torque for this protocol.

### 8.3 Ground stop and restart, Figure 6

```text
figure_center_deg = +30
figure_width_deg = 12
ground_extent_deg = 360
figure_amplitude_deg = 5
ground_amplitude_deg = 5
frequency_hz = 2.5
pre_record_state = synchronous_motion
ground_stop_time_s = 0
ground_restart_time_s = 12  # paper says approximately 12 s
record_end_s = 22
repetitions = 100
```

At `t = 0`, freeze the ground at its instantaneous angle while the figure continues. Do not reset the ground angle to zero. At approximately `12 s`, restart the ground in synchrony with the figure using a position-continuous transition.

Expected response: positive attraction develops after the ground stops, approaches a stationary level in about `5 s`, and returns toward the synchronous baseline after ground motion resumes. The oscillatory torque amplitude during figure-only motion is approximately equal to the amplitude during synchronous figure-plus-ground motion.

### 8.4 Figure-width and amplitude sweep, Figures 7 and 8

Use the Figure 6 procedure and sweep:

```text
figure_widths_deg = [0, 6, 12, 24, 36, 48]
equal_figure_ground_amplitudes_deg = [0.5, 1, 3, 5, 7, 10]
frequency_hz = 2.5
```

The `6 deg` width is read from the first nonzero plotted marker; the caption only states the overall `0 to 48 deg` range. `FIGURE_READ`

For each setting, measure during a declared steady figure-only interval:

1. First-harmonic or oscillation amplitude of yaw torque, corresponding to Figure 7.
2. Mean attraction torque, corresponding to Figure 8.

Each plotted point was based on 100 measurements. `R83`

Expected response: most width-dependent growth occurs by `12 deg`; further width has comparatively little effect. Mean and oscillatory responses increase with motion amplitude.

### 8.5 Ipsilateral half-ground, Figure 14

```text
figure_center_deg = +30
figure_width_deg = 12
ground_azimuth_interval_deg = [0, 180]
opposite_hemifield = illuminated_uniform_no_contrast
edge_screen_width_deg = 10
figure_amplitude_deg = 5
ground_amplitude_deg = 5
frequency_hz = 2.5
initial_phase_deg = 0
transition_start_s = 0.8
target_phase_deg = 90
trial_duration_s = 2.4
repetitions = 100
```

The figure and moving half-ground lie in the same visual hemifield. The other hemifield is illuminated but lacks contrast. The Figure 14 text places the experimental phase transition at `0.8 s`; use that value instead of inheriting the Figure 4 start time.

Expected response: positive raw mean before the transition, followed by a further positive increase and approximately two response peaks per stimulus period.

### 8.6 Unequal-amplitude phase sweep, Figure 16

Use the black-stripe geometry of Figure 2 with:

```text
figure_amplitude_deg = 5
ground_amplitude_deg = 2.5
frequency_hz = 2.5
relative_phases_deg = [0, 30, ..., 360]  # FIGURE_READ
```

Each point averages 10 independent flies. Expected attraction remains positive even at `0 deg` and `180 deg` because unequal amplitudes create relative motion.

### 8.7 Ground-to-figure amplitude-ratio sweep, Figure 18

Use the black-stripe geometry of Figure 2 with:

```text
figure_amplitude_deg = 1
ground_to_figure_amplitude_ratios = [0, 0.5, 1, 2, 3, 4, 6]
relative_phases_deg = [0, 180]
```

The ratios are read from the graph. `FIGURE_READ` Normalize the mean figure-only response at ratio `0` to `1`. Each point averages 10 independent flies; average SEM is approximately `+/-0.1` relative units. `R83`

Expected response is near zero around equal amplitudes and becomes negative when ground amplitude exceeds figure amplitude.

### 8.8 Two-figure bilateral protocol, Figure 20

```text
extended_textured_ground = false
right_figure_center_deg = +40
left_figure_center_deg = -40
right_figure_width_deg = 12
left_figure_width_deg = 12
right_figure_amplitude_deg = 5
left_figure_amplitude_deg = 5
frequency_hz = 2.5
initial_relative_phase_deg = 0
transition_start_s = 0.4
right_relative_to_left_target_phase_deg = 90
transition_duration_s = 0.4
trial_duration_s = 2.0
repetitions = 100
```

The appearance of the otherwise unused panorama and the texture relation between the stripes are `UNKNOWN`. Use a uniform field at mean luminance and registered-copy stripe textures as the `DEFAULT`, then run an independent-texture control.

Expected response: mean torque remains near zero, while oscillation amplitude after the phase change is approximately twice the synchronous amplitude. The rising segment contains a shallow saddle.

## 9. Measured behavioral output

### 9.1 Historical measured variable

The primary measured output is wing-generated flight torque about the vertical body axis under compensation. `R83`

It is:

- an attempted yaw torque;
- a physical moment;
- measured while the body is fixed;
- positive for an attempted right turn;
- negative for an attempted left turn;
- stored as a stimulus-locked time series and commonly averaged over 100 sweeps.

It is not:

- body yaw angle;
- body yaw rate;
- a free-flight trajectory;
- lift or thrust in this assay;
- wing kinematics;
- an internal steering command;
- simultaneously recorded neuronal membrane potential.

### 9.2 Units

Store torque internally in `N m` and export the historical unit:

```text
tau_dyne_cm = tau_Nm * 1e7
tau_Nm = tau_dyne_cm * 1e-7
```

| Historical value | SI value |
|---:|---:|
| `0.1 dyne cm` | `1e-8 N m` |
| `0.5 dyne cm` | `5e-8 N m` |
| `1.0 dyne cm` | `1e-7 N m` |
| `1.6 dyne cm` | `1.6e-7 N m` |

Some displayed traces approach approximately `1.6 dyne cm`. This is not a reported instrument limit.

### 9.3 Required raw channels

At minimum, save:

```text
time_s
trial_id
stimulus_interval

figure_angle_command_deg
figure_angle_realized_deg
ground_angle_command_deg
ground_angle_realized_deg
relative_displacement_realized_deg
relative_phase_command_deg
relative_phase_realized_deg

fly_generated_yaw_moment_engine_Nm
support_on_fly_yaw_reaction_engine_Nm
reported_yaw_torque_Nm
reported_yaw_torque_dyne_cm

body_x_m
body_y_m
body_z_m
body_roll_deg
body_pitch_deg
body_yaw_deg
body_yaw_rate_deg_s
```

Body pose channels are diagnostics. They should remain within constraint tolerance and are not historical response variables.

Useful optional diagnostics:

```text
left_wing_force_xyz_N
right_wing_force_xyz_N
left_wing_yaw_moment_Nm
right_wing_yaw_moment_Nm
constraint_force_xyz_N
constraint_moment_xyz_Nm
wingbeat_phase
controller_action
render_timestamp_s
physics_timestamp_s
```

### 9.4 Required metadata

Save enough metadata to reproduce every retinal input:

```text
spec_version
protocol_id
trial_id
agent_or_individual_id

master_texture_seed
figure_texture_seed
ground_texture_seed
texture_relationship_mode
texture_generator_name_and_version
black_probability
pixel_width_deg
pixel_height_deg
grid_origin_azimuth_deg
grid_origin_elevation_deg

figure_mean_azimuth_deg
figure_width_deg
figure_vertical_extent_deg
ground_start_azimuth_deg
ground_end_azimuth_deg
screen_centers_deg
screen_widths_deg

figure_amplitude_deg
ground_amplitude_deg
frequency_hz
initial_relative_phase_deg
target_relative_phase_deg
signed_phase_delta_deg
phase_transition_mode
phase_transition_start_s
phase_transition_duration_s

mean_luminance_cd_m2
black_luminance_cd_m2
white_luminance_cd_m2
contrast_definition
contrast_value

figure_layer_radius_m
ground_layer_radius_m
render_rate_hz
stimulus_update_rate_hz
physics_timestep_s
torque_sampling_rate_hz
filter_definition
tether_model
engine_name_and_version
```

## 10. Behavioral analysis

Never discard single-trial data after producing the historical sweep average.

### 10.1 Phase-locked averaging

- Start every repeat at the same ground phase.
- Align repeats to the realized stimulus, not only wall-clock time.
- Average torque sample-by-sample after resampling to a common stimulus phase grid if necessary.
- Reproduce `100` sweeps unless the protocol specifies a different aggregation.
- Report mean, standard deviation, and SEM across repetitions or independent agents.

The historical acquisition sample rate and filter are `UNKNOWN`. Record unfiltered raw simulation torque first. Apply any comparison filter as a named, versioned post-processing step.

### 10.2 Mean torque

For a declared window of duration `T_w`:

```text
mean_tau = (1 / T_w) * integral(tau(t), window)
```

Use an integer number of realized stimulus periods. Report the window endpoints and whether it is synchronous, transitional, or steady relative-motion data.

For phase-transition protocols, also calculate:

```text
delta_mean_tau = mean_tau_steady_relative - mean_tau_steady_synchronous
```

This baseline subtraction is essential for half-ground trials with nonzero pre-transition mean torque.

### 10.3 First harmonic

For torque with its window mean removed:

```text
C1 = (2 / T_w) * integral((tau(t) - mean_tau) * exp(-i * 2*pi*f*t), window)
first_harmonic_amplitude = abs(C1)
first_harmonic_phase = angle(C1)
```

When the realized ground is not a perfect analytic sinusoid, estimate phase using the actual ground-angle fundamental rather than nominal time.

Report torque phase relative to both:

1. ground position;
2. ground velocity, which is `+90 deg` ahead of sinusoidal position.

This prevents a common `90 deg` convention error. The paper's Figure 5 plots phase relative to ground position while discussing velocity-sensitive responses.

### 10.4 Settling and waveform features

For the ground-stop protocol, calculate:

- baseline mean before stop;
- steady figure-only mean;
- time to enter and remain within a declared band around the steady mean;
- mean after synchronous restart;
- first-harmonic amplitude in figure-only and synchronous intervals.

For the `90 deg` and `270 deg` protocols, preserve full phase-binned waveforms. Plateau, peak, saddle, and harmonic-distortion features are informative, but they should not be hard apparatus acceptance criteria because fine waveform details varied across comparative behavioral records.

## 11. Optional intracellular-output variant

This is a separate assay. Do not attach this output to the behavioral torque trial as if both were recorded simultaneously.

### 11.1 Physical and visual setup

| Parameter | Value | Provenance |
|---|---:|---|
| Body and head | Immobilized | `R83` |
| Recorded cell | Right equatorial horizontal cell in the lobula plate | `R83` |
| Ground panorama | `-105 deg <= psi <= +105 deg` | `R83` |
| Rear panorama | Open for electrode access | `R83` |
| Figure center | `+40 deg` | `R83` |
| Figure width | `10 deg` | `R83` |
| Frequency | `2.5 Hz` | `R83` |
| Amplitude | Usually `+/-4 to +/-5 deg`; principal plot `+/-5 deg` | `R83` |
| Repetitions | `20` | `R83` |

The standard full program is:

1. `5 s` synchronous figure-ground motion.
2. `5 s` relative motion at `0`, `90`, `180`, or `270 deg`.
3. `2 s` stationary pattern.

The plotted records contain the last synchronous period and the first three relative-motion periods. The phase switch shown at `0.4 s` in the cropped plot is the alignment point in that excerpt, not the start of the full `12 s` program.

For the separate figure/ground contribution test corresponding to Figure 27, use the same geometry, amplitude, frequency, and recording configuration in three trials:

1. Figure moves while ground remains stationary.
2. Ground moves while figure remains stationary.
3. Figure and ground move synchronously.

### 11.2 Measured output

Record intracellular graded membrane potential:

```text
Vm_mV
Vrest_mV
delta_Vm_mV = Vm_mV - Vrest_mV
```

- Positive `delta_Vm` is depolarization.
- Negative `delta_Vm` is hyperpolarization.
- For the right recorded cell, clockwise/progressive horizontal motion depolarizes and counterclockwise/regressive motion hyperpolarizes.
- Superimposed spikes from contralateral stimulation were excluded from the graded-potential analysis.
- The paper's `-51 mV` resting potential is one example record, not a universal target.

Minimum data channels:

```text
time_s
trial_id
stimulus_interval
figure_angle_realized_deg
ground_angle_realized_deg
Vm_mV
Vrest_mV
delta_Vm_mV
spike_event_or_mask
```

Expected qualitative checks:

- figure-only and ground-only movement can produce membrane-potential amplitudes of comparable order despite their different visual areas;
- simultaneous figure-plus-ground response is smaller than the arithmetic sum of the separate responses;
- progressive motion produces depolarization and regressive motion produces hyperpolarization.

## 12. Reference configuration

The following YAML is the canonical starting point. Values labeled `DEFAULT` in comments are implementation choices, not recovered 1983 facts.

```yaml
spec_version: "1.0"
protocol_id: "R83_Fig3a_0_to_90"

coordinate_system:
  handedness: "right"
  x_axis: "body_forward"
  y_axis: "body_left"
  z_axis: "up"
  paper_azimuth_zero: "body_forward"
  paper_positive_azimuth: "toward_right_visual_field"
  paper_positive_torque: "attempted_right_turn"

assay:
  topology: "open_loop_tethered_flight"
  body_translation_locked: true
  body_roll_locked: true
  body_pitch_locked: true
  body_yaw_locked: true
  head_rigid_to_thorax: true
  wings_active: true
  stimulus_feedback_from_fly: false
  cylinder_collision_with_fly: false
  cylinder_airflow_coupling: false
  tether_model: "ideal_rigid_6dof"   # DEFAULT

geometry:
  observer_at_common_center: true
  figure_layer_radius_m: 0.035        # DEFAULT based on later apparatus
  ground_layer_radius_m: 0.036        # DEFAULT based on later apparatus
  figure_vertical_extent: "full_visible_field"  # DEFAULT
  ground_vertical_extent: "eye_fov_plus_margin" # DEFAULT
  show_cylinder_top_bottom_edges: false

lighting:
  mean_luminance_cd_m2: 700
  illumination_model: "uniform_emissive"        # DEFAULT
  michelson_contrast: 0.78                       # DEFAULT based on later apparatus
  white_luminance_cd_m2: 1246                    # DERIVED under stated assumptions
  black_luminance_cd_m2: 154                     # DERIVED under stated assumptions
  auto_exposure: false
  shadows: false
  specular: false
  motion_blur: false

texture:
  type: "binary_random_dot"
  pixel_width_deg: 3
  pixel_height_deg: 3
  black_probability: 0.5                         # DEFAULT
  relationship_mode: "registered_copy"          # DEFAULT
  master_seed: 123456                            # DEFAULT; replace and record
  figure_seed: null                              # master seed used in registered mode
  ground_seed: 123456
  grid_origin_azimuth_deg: 0                     # DEFAULT
  grid_origin_elevation_deg: 0                   # DEFAULT
  horizontal_wrap: true
  scene_texture_filter: "nearest"

figure:
  type: "textured_vertical_stripe"
  mean_azimuth_deg: 30
  width_deg: 12
  amplitude_deg: 5
  frequency_hz: 2.5
  initial_phase_deg: 0

ground:
  type: "full_textured_panorama"
  start_azimuth_deg: -180
  end_azimuth_deg: 180
  amplitude_deg: 5
  frequency_hz: 2.5
  initial_phase_deg: 0

phase_transition:
  target_paper_phase_deg: 90
  signed_phase_delta_deg: 90
  start_s: 0.4
  duration_s: 0.4                            # DEFAULT inferred from plotted/model transition
  mode: "one_period_phase_ramp"              # DEFAULT

trial:
  duration_s: 2.0
  repetitions: 100
  ground_start_phase_deg: 0
  phase_locked_repeats: true

rates:
  physics_rate_hz: 2000                      # DEFAULT
  stimulus_eye_update_rate_hz: 400           # DEFAULT
  raw_torque_logging_rate_hz: 1000           # DEFAULT

output:
  primary: "reported_yaw_torque_Nm"
  also_export_dyne_cm: true
  save_single_trials: true
  save_phase_locked_average: true
  save_commanded_and_realized_stimulus: true
  save_constraint_diagnostics: true
```

## 13. Renderer pseudocode

```text
initialize deterministic texture generator
T_ground = generate_binary_angular_texture(master_seed, 3 deg, 3 deg)

if relationship_mode == registered_copy:
    T_figure = T_ground
else:
    T_figure = generate_binary_angular_texture(figure_seed, 3 deg, 3 deg)

for each simulation time t:
    phase_rel = phase_schedule(t)

    alpha_ground_cmd = A_ground * sin(omega*t)
    alpha_figure_cmd = A_figure * sin(omega*t + phase_rel)

    alpha_ground_actual = ground_actuator(alpha_ground_cmd)
    alpha_figure_actual = figure_actuator(alpha_figure_cmd)

    for each eye ray:
        psi = paper_azimuth(ray)
        elev = elevation(ray)

        ground_coord = wrap360(psi - alpha_ground_actual)
        figure_local = wrap180(psi - (figure_mean + alpha_figure_actual))

        if inside_figure_mask(figure_local, elev):
            figure_coord = wrap360(psi - alpha_figure_actual)
            L = sample(T_figure, figure_coord, elev)
        else if inside_stationary_screen(psi, elev):
            L = stationary_screen_luminance
        else if inside_ground_mask(psi, elev):
            L = sample(T_ground, ground_coord, elev)
        else:
            L = uniform_mean_luminance

        deliver_luminance_to_eye(ray, L)

    integrate wing and body physics with body pose constrained
    read or calculate fly-generated yaw moment
    convert to paper-positive yaw torque
    log stimulus, torque, pose diagnostics, and timestamps
```

For protocols with a figure in front of a stationary screen, the figure branch must remain ahead of the screen branch, as shown above.

## 14. Acceptance tests

Separate apparatus tests from behavioral-model tests. A renderer can pass apparatus fidelity even if the simulated fly does not reproduce the biological torque response.

### 14.1 Coordinate and geometry tests

1. A ray at paper `psi = +30 deg` has direction `[cos(30 deg), -sin(30 deg), 0]`.
2. A stationary `12 deg` figure centered at `+30 deg` covers `+24 to +36 deg`.
3. Under `+/-5 deg` motion, the full figure envelope is `+19 to +41 deg`.
4. A full ground covers all azimuth rays without a visible seam.
5. The observer remains at the common rotation center.
6. No top or bottom cylinder edge is visible to any eye sample.

### 14.2 Texture tests

1. Every texture cell spans exactly `3 deg x 3 deg` in the angular texture definition.
2. A full panorama has 120 horizontal columns.
3. A grid-aligned `12 deg` figure spans four horizontal cells.
4. The same seed reproduces exactly the same texture bitmap.
5. In `registered_copy` mode with synchronous equal motion, pixels match across both figure boundaries.
6. Moving the figure relative to the ground moves its mask and texture rigidly, without regenerating pixels.
7. Mean luminance is invariant under cylinder rotation to numerical tolerance.

### 14.3 Motion tests

For `A = 5 deg`, `f = 2.5 Hz`:

1. Period is `0.4 s`.
2. Peak amplitude is `5 deg`, not `10 deg`.
3. Peak speed is `78.5398 deg/s` or `1.37078 rad/s`.
4. Peak acceleration is `1233.70 deg/s^2` or `21.5321 rad/s^2`.
5. At `0 deg` relative phase, equal-amplitude realized displacements are identical.
6. At `90 deg`, relative displacement amplitude is `7.0711 deg`.
7. At `180 deg`, relative displacement amplitude is `10 deg`.
8. At `270 deg`, using signed `-90 deg` produces the mirror transition while the steady sinusoid is equivalent to `270 deg`.
9. A ground stop freezes its instantaneous angle rather than resetting it.
10. Commanded and realized angles are both stored.

### 14.4 Mechanical tests

1. Body translation, roll, pitch, and yaw remain below declared solver tolerances.
2. Head-to-thorax transform is constant.
3. Wing joints remain active.
4. Visual cylinders exert no collision or aerodynamic forces on the fly.
5. With the body fixed, `abs(tau_fly_z + tau_support_z)` is below `max(1e-3 * peak_abs_tau_fly_z, 1e-10 N m)`, unless another logged yaw moment explains the residual.
6. Two trials with radically different fly outputs have identical figure and ground angle traces. This proves open-loop isolation.

### 14.5 Torque sign and unit tests

Apply a known clockwise/right-turn fly moment with magnitude `1e-7 N m`.

Expected:

```text
reported_yaw_torque_Nm = +1e-7
reported_yaw_torque_dyne_cm = +1.0
```

Apply the opposite moment and expect `-1.0 dyne cm`.

Verify exact round-trip conversion:

```text
1e-7 N m -> 1 dyne cm -> 1e-7 N m
```

### 14.6 Analysis tests

Feed the analysis code:

```text
tau(t) = 0.3 + 0.2*sin(2*pi*2.5*t + 30 deg)  dyne cm
```

over an integer number of periods. Recover:

- mean `0.3 dyne cm`;
- first-harmonic amplitude `0.2 dyne cm`;
- phase `30 deg` relative to position.

### 14.7 Behavioral-model validation targets

These test the fly/controller model, not just the apparatus:

1. Synchronous figure and full ground produce oscillatory torque with approximately zero mean.
2. `90 deg` and `270 deg` relative phase with a right-side figure produce positive mean torque.
3. The within-cycle `90 deg` and `270 deg` waveforms have reversed temporal organization.
4. The main `180 deg` condition has approximately zero mean attraction.
5. Stopping the ground while the figure continues causes positive attraction that approaches a stationary level in roughly `5 s`.
6. Restarting synchronous ground motion returns mean torque toward baseline.
7. Figure-width response grows strongly by `12 deg` and changes less for larger widths.
8. The two-figure bilateral trial retains near-zero mean while its oscillatory amplitude increases after the phase offset.

## 15. Unreported values that must stay configurable

Do not present any of the following as recovered historical facts:

- exact 1983 cylinder radii or radial separation;
- cylinder height, wall thickness, material, or transparency;
- exact figure vertical height;
- exact lower-panorama placement of the `3 deg` black stripe;
- pattern grid origin and seam location;
- original pixel bitmap, random seed, random-number generator, or black probability;
- whether all protocols used a registered copied texture or an independent statistically equivalent texture;
- black and white luminance in 1983;
- 1983 pattern contrast;
- spectrum, color temperature, gamma, surface reflectance, lamp nonuniformity, or residual flicker;
- exact stationary-screen luminance, vertical extent, radial distance, and edge placement;
- appearance outside the moving regions in half-ground and two-figure protocols;
- torque-compensator stiffness, damping, inertia, bandwidth, controller gains, latency, calibration constant, noise, and filtering;
- exact body pitch, roll, or tether attachment point;
- servo gear ratio, backlash, acceleration limits, tracking error, and phase-transition trajectory;
- stimulus update rate, rendering rate, acquisition sample rate, and analog filter;
- adaptation duration, inter-trial interval, and phase-sweep averaging-window duration;
- temperature, pressure, humidity, air density, and ambient airflow.

Any selected value for one of these fields MUST be tagged `DEFAULT`, stored in run metadata, and included in sensitivity testing if it could alter retinal input or measured torque.

## 16. Implementation order

1. Implement the coordinate convention and sign tests.
2. Implement a stimulus-only angular renderer with deterministic textures.
3. Verify registered-copy camouflage during synchronous motion.
4. Implement analytic figure and ground motion and all relative-phase unit tests.
5. Add the rigid open-loop body constraint and torque extraction.
6. Run the canonical Figure 3a, 3b, and 3c protocols.
7. Add the Figure 6 stop/restart test.
8. Add half-ground, width/amplitude, black-stripe, and two-figure variants.
9. Add the optional intracellular-output configuration only if the fly model exposes the required neuron.
10. Perform rate, texture-seed, texture-relationship, phase-transition, and assumed-radius sensitivity runs.

The final canonical simulation should be describable in one sentence: prescribed angular figure and ground motion is presented to a fixed, head-stabilized flying agent; the only primary response is its attempted yaw torque about the constrained vertical body axis.
