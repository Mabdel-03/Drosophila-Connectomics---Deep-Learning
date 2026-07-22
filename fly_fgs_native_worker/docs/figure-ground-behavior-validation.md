# Proposal: a paper-grounded figure–ground validation for `fly-sensor2behavior`

## Decision

The Fox et al. (2014) dataset can support a meaningful external validation of the complete visual-to-muscle path, but the strongest test is **not** a single “turned left/turned right” comparison. The recommended benchmark is to reproduce the paper's figure and ground **spatiotemporal action fields (STAFs)** from simulated wing steering, then hold out the paper's closed-loop fixation and triangle-wave experiments as endpoint tests.

This is a good ground-truth test because the two independent motion-noise inputs let us identify, from one behavioral output, separate transfer functions for figure tracking and wide-field stabilization. A model that merely turns toward a bar, or that bypasses the retina with a hand-written steering rule, should fail the spatial, temporal, control, and causal checks below.

The current repository is not ready to claim a pass. It already has useful pieces—content-addressed fly-FGS circuit execution, individual NOD1 voltages, a causal motor bridge, muscle/wing mechanics, and a pinned FlyBody/MuJoCo backend—but its canonical episode does not reproduce the paper's stimulus or virtual-tether preparation, and several biological stages remain explicitly uncalibrated. That is useful: this benchmark would provide a concrete target for closing those gaps rather than retroactively labeling the existing 0.5 s fixture as behaviorally valid.

## Claim under test

The proposed claim is deliberately narrow:

> Given only a paper-matched retinal stimulus and initial state, the frozen model produces left/right wing steering through its simulated retina, neural pathway, motor neurons, muscles, and mechanics such that the population-level figure–ground response is statistically equivalent to the reported fly response.

The test would **not** by itself prove that every internal neuron, synaptic weight, muscle parameter, or aerodynamic force is correct. Behavioral agreement is necessary for end-to-end validity, not sufficient for mechanistic identification. Stage-specific neural and mechanical validation should remain separate gates.

## End-to-end test topology

```text
paper-matched LED panorama
  -> compound-eye sampling / phototransduction
  -> motion pathway and figure/ground circuit
  -> NOD1 and descending pathway
  -> premotor neurons and identified wing motor neurons
  -> individual muscles
  -> tethered FlyBody wings and aerodynamics
  -> simulated left/right wingbeat amplitudes
  -> delta wingbeat amplitude and yaw reaction torque
  -> the paper's exact STAF and closed-loop analyses
```

Only the visual scene is an allowed input. The paper's measured steering trace, figure trajectory, STAF, or fitted kernel must never be injected into the controller or motor path.

## What the paper gives us

The source is Fox, Aptekar, Zolotova, Shoemaker and Frye, “Figure–ground discrimination behavior in *Drosophila*. I. Spatial organization of wing-steering responses,” *Journal of Experimental Biology* 217, 558–569 (2014), DOI [`10.1242/jeb.097220`](https://doi.org/10.1242/jeb.097220). The [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3922833/) is open under CC BY 3.0.

### Preparation and stimulus contract

The important details to reproduce are:

- Adult female flies, 3–5 days after eclosion, rigidly tethered in a cylindrical `32 x 96` pixel green LED arena.
- Arena pixel width: `3.75 deg`; wing signals digitized at `1,000 Hz`.
- Behavioral output: the difference between left and right wingbeat amplitudes, or delta wingbeat amplitude (`Delta WBA`), which is proportional to yaw torque. The paper's output is in uncalibrated volts.
- The standard figure is a `30 deg`-wide vertical bar spanning `-60 to +60 deg` elevation.
- Figure and ground contain matched, random vertical one-pixel stripes, band-pass filtered so most uniform elements are `2–4 pixels` (`7.5–15 deg`) wide, with `50%` average contrast.
- The figure is defined by motion relative to an otherwise visually matched ground; it is not merely a dark bar on a bright field.
- STAF inputs are two independent `127`-element maximum-length sequences. Each element commands a one-pixel (`3.75 deg`) left or right position step for either the figure or ground.
- STAF sampling uses 24 starting positions at `15 deg` intervals, while the moving figure lets the analysis fill all 96 azimuthal arena positions. The sequence is repeated three times in `15.6 s`; original and inverted figure-sequence trials are subtracted. The main STAF cohort is `N=27` flies.
- The exact m-sequence identities and explicit element update interval are not supplied in the article text. The raw stimulus command stream should therefore be obtained from the authors; inferring timing only from `15.6 s / (3 x 127)` is not acceptable for the final ground-truth version.

### Published behavioral targets

| Target | Paper protocol | Reported result | Proposed use |
|---|---|---|---|
| Closed-loop fixation | 20 s trials; figure under negative feedback; last 10 s scored; `N=16` | Vector strength `0.93` on static ground, `0.61` with one-pixel counter-rotation; both Rayleigh `P<0.05` | Held-out closed-loop endpoint |
| Figure STAF | Independent figure/ground m-sequences; `N=27` | Largest figure response in frontal field, weak in rear | Primary spatial-transfer test |
| Ground STAF | Same trials | Ground response suppressed when figure is frontal and strong when figure is rearward | Primary discrimination test |
| Gray-window control | Figure replaced by an equal-mean-luminance gray patch moving with ground | Gray occlusion does not reproduce the central ground-response suppression | Primary mechanism-negative control |
| Size tuning | Widths `7.5, 15, 30, 45, 60, 90, 120, 180 deg`; `N=24` except `N=18` at 30 and 180 deg | Strong width-dependent reorganization; ground STAF amplitude increases for large figures | Secondary out-of-distribution test |
| Temporal kernels | Overdamped-oscillator fits to STAF kernels | Figure/ground rise constants differ by less than 10 ms; decay constants differ by more than 200 ms; figure has the larger DC position component | Primary dynamics test after raw-data recovery |
| Triangle-wave response | 30 deg peak-to-peak, front/rear, 0.5 and 1.2 Hz; 50 trials from 10 flies | Combined STAF prediction `r=0.46–0.58` | Held-out predictive test |
| Front/rear allocation | Same triangle-wave trials | Front: response vs figure `r=0.57`, vs ground `r=0.17`; rear: figure `r=-0.22`, ground `r=0.58` | Held-out figure/ground allocation test |

The PMC record has no attached machine-readable supplement. Values printed in the article are valid targets, but the heat maps and temporal distributions should not be treated as precise raw data without an explicit recovery step.

## Recommended benchmark

### 1. Reproduce the experiment as a virtual tether, not as ordinary free flight

The thorax should be fixed at the arena center while both wings, muscles, hinge mechanics, and aerodynamic loading remain active. This matches the preparation and prevents free-body rotation from changing the comparison being made. For open-loop STAF trials, the rendered panorama follows the prescribed stimulus only. For closed-loop trials, the measured simulated steering signal updates the figure exactly as in the paper.

At every wingbeat, derive:

- left and right stroke amplitude from measured wing joint trajectories;
- `Delta WBA_sim = amplitude_left - amplitude_right`, with the sign convention fixed before looking at test results;
- independently, the constraint/reaction yaw torque on the tether and the aerodynamic yaw torque.

`Delta WBA_sim` is the primary paper-matched observable. Torque is a valuable SI-unit consistency check, but should not silently replace the paper's measurement. The relationship between the two should have the expected sign and be approximately monotonic over the tested range.

Because the paper reports Delta WBA in uncalibrated volts, absolute amplitude cannot be compared without a sensor conversion. Use one of these tracks and report which one was used:

1. **Strict predictive track:** evaluate only scale-free STAF shape, timing, spatial allocation, and closed-loop position statistics; do not fit behavior.
2. **One-scalar calibration track:** fit one fixed multiplicative volts-per-degree conversion on a declared calibration subset, then freeze it. No neural, motor, muscle, or controller parameter may be fit on the behavior test set.

An intercept is unnecessary for cross-correlations after baseline removal. A time shift must not be fitted; latency is part of the biological prediction.

### 2. Run the same system-identification experiment

For each virtual fly:

1. Render the paper-matched random-stripe ground and 30 deg figure on the 96-column azimuth grid and the paper's vertical extent.
2. Use the authors' two exact 127-element m-sequences and texture seeds if recovered. Keep figure and ground sequences independent.
3. Run the 24 randomized starting positions, three repeats, and the original/inverted figure-sequence pair.
4. Record the scene, every retinal sample and availability time, selected circuit states, all motor and muscle states, measured wing joints, `Delta WBA_sim`, reaction torque, and any clipping/saturation.
5. Resample only the final observable at 1 kHz for the paper-matched analysis. Preserve native-rate traces as the authoritative source.
6. Apply the authors' analysis: direct circular cross-correlation of Delta WBA with the ground sequence; cross-correlation of the time derivative of Delta WBA with the figure sequence followed by integration; subtract the inverted-sequence responses; divide by the `3.75 deg` impulse magnitude.
7. Use unsmoothed STAFs for every metric. A four-pixel box filter may be used only for a display matching the paper.

The result is one figure STAF and one ground STAF per virtual fly on an azimuth-by-lag grid. This is the primary validation object.

### 3. Treat biological variation as part of the target

A single deterministic rollout cannot reproduce a population statistic such as vector strength. Define a virtual population before opening held-out targets, drawing uncertain but biologically meaningful parameters such as optical alignment, response latency, membrane parameters, neuromuscular gain, wing stiffness, and sensor/motor noise from frozen distributions. Do not add arbitrary output noise solely to match the reported spread.

Use the same cohort sizes as the paper for each comparison. Randomness should be hierarchical and reproducible: a virtual-fly seed controls stable individual parameters, while a nested trial seed controls within-fly noise and texture/sequence order.

### 4. Use a biological noise ceiling instead of arbitrary tolerances

The ideal acceptance rule requires the authors' per-fly traces or STAFs:

1. Repeatedly split the biological flies into equally sized reference and pseudo-test cohorts.
2. Compute the distance between the two biological cohort means. This is the finite-sample biological disagreement distribution.
3. Compare an equally sized virtual cohort with a biological reference cohort using the same distance.
4. Pass equivalence only if the model-to-biology distance is inside the pre-registered `95%` biological disagreement limit.

Use a vector of distances rather than one forgiving aggregate:

- centered two-dimensional correlation for STAF shape;
- normalized RMSE after applying only the pre-frozen output scale;
- onset, rise, decay, peak, and DC-component errors from the paper's kernel fit;
- front-versus-rear response ratios;
- the ground-suppression specificity contrast described below.

For reported correlations, compare Fisher-`z` transformed values and use a hierarchical bootstrap over fly, then trial. For closed-loop fixation, bootstrap flies and compare the simulated vector strength with the biological confidence interval; also require the correct ordering (`static > counter-rotating`) and significant nonuniform frontal fixation in both conditions.

If raw data cannot be recovered, digitized curves plus digitization uncertainty can support an explicitly **provisional reproduction score**, but not a calibrated/validated promotion. The printed scalar values can still serve as directional and gross-magnitude smoke tests.

### 5. Make the gray-window contrast a hard causal discriminator

Define a ground-notch statistic on the unsmoothed ground STAF, using a preregistered frontal azimuth band and the first 100 ms, matching the paper's summary window:

```text
ground notch = mean rear ground response - mean frontal ground response
suppression specificity = ground notch with moving figure - ground notch with gray window
```

The simulated moving figure should create a positive ground notch, while the equal-luminance gray window should not create a comparable notch. The suppression-specificity confidence interval must be positive. This rejects a model that appears to discriminate figure from ground only because the bar masks frontally important ground pixels.

Also run these software-causal checks:

- remove relative motion while keeping luminance and masked area matched;
- shuffle the two m-sequences and confirm that the corresponding recovered kernel collapses;
- replay the exact same retinal frames twice and require deterministic equality in deterministic mode;
- cut the declared visual-to-NOD1 path and require loss of the relevant motor response;
- cut the relevant NOD1-to-motor or motor-to-muscle path and require loss of Delta WBA despite preserved upstream activity;
- mirror the stimulus and require the signed response to mirror without changing its magnitude distribution beyond numerical tolerance.

These checks validate causal dependence and bookkeeping. They do not replace the biological-equivalence tests.

### 6. Seal endpoint tests until the STAF pipeline is frozen

After the model, uncertainty distributions, m-sequence analysis, and any one-scalar output calibration are frozen, run:

#### Held-out test A: triangle waves

Use the paper's 30 deg peak-to-peak trajectories at 0.5 and 1.2 Hz with the figure centered in front or rear. Convolve the previously estimated, unsmoothed figure and ground STAFs with the new inputs; do not refit kernels.

Require:

- combined prediction-to-simulated-response correlations consistent with the paper's `0.46–0.58` range after accounting for biological uncertainty;
- front figure dominance (`r_figure > r_ground`) with targets near `0.57` and `0.17`;
- rear ground dominance (`r_ground > r_figure`) with targets near `0.58` and `-0.22`;
- no condition-specific gain or time shift.

This tests whether the identified transfer functions generalize to a different temporal stimulus.

#### Held-out test B: closed-loop fixation

Run five 20 s trials per virtual fly for `N=16` virtual flies, interleaving the same 5 s active-bar periods. Score only the last 10 s. In the opposing condition, every one-pixel figure displacement drives a one-pixel ground displacement in the opposite direction.

Compute circular vector strength

```text
R = magnitude(mean(exp(i * figure_azimuth)))
```

and the Rayleigh test exactly as in the paper. The primary comparison is equivalence to the biological bootstrap intervals around the reported `0.93` and `0.61`, not exact equality to those two rounded point estimates.

This test requires a live loop from measured wing response back to the display. Replaying a precomputed NOD1 trace or a paper-derived bar trajectory is not an end-to-end closed-loop test.

#### Held-out test C: width sweep

Use the eight published widths without changing parameters. This is a hard extrapolation test because the response to large figures is not a simple monotonic extension of the 30 deg case. Preserve the paper's cohort sizes and compare the complete figure and ground STAF profiles, not only peak amplitude.

## Pass/fail structure

Do not collapse the benchmark into a single weighted score. A release can claim figure–ground behavioral validity only if all of these gates pass:

1. **Reproduction gate:** exact stimulus assets, clocks, sequence hashes, analysis implementation, and native traces are content-addressed and rerunnable.
2. **Causality gate:** no future reads or behavioral target injection; declared cuts and mirror tests have the expected effects.
3. **Primary STAF gate:** figure and ground maps fall within the biological noise ceiling for spatial shape and temporal dynamics.
4. **Suppression-control gate:** the moving-figure notch is present and is significantly stronger than the gray-window notch.
5. **Held-out prediction gate:** frozen STAFs generalize to the triangle-wave responses.
6. **Closed-loop gate:** the virtual population reproduces frontal fixation and the degradation under opposing ground motion.
7. **Mechanics sanity gate:** wing amplitudes, reaction torque, actuator limits, contacts, and energy remain finite and physiologically plausible; no hidden clamp or controller saturation explains the result.

A model that passes software causality but lacks raw biological data should remain `software_correct` or `provisional_behavioral_reproduction`, not `biologically_validated`.

## Data acquisition plan

The order of preference is:

1. Ask the authors for the per-fly 1 kHz Delta WBA traces, the exact figure/ground command streams, m-sequence generators/seeds, random-stripe patterns, trial metadata, sign convention, LED update timing, and any analysis scripts or per-fly STAF matrices.
2. If raw traces are unavailable, request at least the per-fly STAFs and fitted parameters behind Figs 4–8.
3. Digitize published panels only as a documented fallback. Store the original panel, axis/color calibration, digitizer version, operator, extracted points, and repeated-digitization error. The article's CC BY 3.0 license permits reuse with attribution.
4. Keep rounded printed targets in a separate manifest; do not pretend they have unreported precision or confidence intervals.

The exact stimulus stream is as important as the response data. A visually similar random bar is insufficient for strict numerical comparison because m-sequence identity, pattern content, and display timing affect finite-trial kernels.

## Fit/validation split that avoids leakage

Before inspecting held-out outputs, freeze a protocol manifest with software and data hashes.

Recommended split:

- **Engineering-only:** manufactured causality, mirrored input, sequence shuffle, deterministic replay, unit and clock checks. No biological targets.
- **Optional calibration:** one global Delta WBA output scale and predeclared population distributions, using only a designated subset of the 30 deg m-sequence cohort. The subset must be grouped by fly, never by randomly mixing time samples from the same fly.
- **Development validation:** remaining flies from the 30 deg STAF cohort and the gray-window control.
- **Sealed tests:** triangle waves, closed-loop counter-rotation, and the width sweep.

For the strongest claim, publish both the no-behavior-fit result and the one-scalar-calibrated result. Any tuning of neural, motor, muscle, or mechanics parameters on the paper should change the claim from external validation to model calibration with held-out validation.

## Current repository gap analysis

The assessment below is based on the checked-in state inspected on 2026-07-18, especially the [project README](../README.md), [fly-FGS source manifest](../data/reference/fly_fgs/source_manifest.v1.json), and current runtime/adapter interfaces.

| Stage | Present now | Missing for this benchmark |
|---|---|---|
| Stimulus | Analytic 1-D grating/figure-ground scenes and a re-entrant fly-FGS scene/body input | Exact 32 x 96 random band-pass stripe arena, paper m-sequences, gray-window control, elevation mask, width sweep, and 15.6/20 s protocols |
| Retina | Analytic 1-D normalized-luminance sampler; fly-FGS endpoint luminance at 1,441 T4a coordinates | Calibrated compound-eye optics, photoreceptors, radiometry, photon noise, and paper LED temporal response |
| Visual circuit | Captured 1,684-cell fly-FGS runtime: 1,457 T4a, 219 LLPC1, 2 vCH, 2 DCH, 4 NOD1 | T5, photoreceptor and lamina stages; calibrated physiological synaptic weights; provenance for every synthetic fallback |
| Motor path | Four exact NOD1 voltages can enter a causal contralateral DNp26 to ipsilateral iv2/i1/iv1/b3 bridge | Completed evidence-backed DN to premotor to motor-neuron graph and calibrated rates, gains, phases, signs, and latencies |
| Muscle/hinge | Individual muscle state, six-axis virtual hinge, and FlyBody torque adapter | Held-out physiological calibration and validated muscle-to-wing mapping |
| Mechanics | Pinned FlyBody/MuJoCo free-flight worker with measured six-axis wing state and aerodynamic wrench | Paper-matched rigid virtual tether and a reviewed Delta WBA extraction from wing cycles |
| Feedback | Re-entrant fly-FGS circuit input accepts heading and figure/ground velocity; reduced analytic yaw loop tests causal feedback | One authoritative online loop connecting rendered paper scene through full circuit/muscles/FlyBody measurement back to the display |
| Existing fly-FGS fixture | Deterministic 0.5 s, 5 ms circuit capture; 24 deg x 180 deg figure; grating disabled; only four NOD1 voltages drive downstream output | It is not one of the paper's trials and must not be scored as a paper reproduction |

The most important near-term blockers are therefore not plotting or score computation. They are (1) recovering the exact behavior/stimulus data, (2) constructing a paper-matched virtual tether and Delta WBA observable, and (3) running the current re-entrant circuit and streaming motor/physics path as one live causal loop.

## Connectome and provenance constraints

The current fly-FGS source manifest identifies FlyWire FAFB materialization `783`, the `139,255`-neuron proofread universe, nanometre skeleton/synapse coordinates, and decimal-string root IDs. It also states that structural counts are not physiological weights, does not encode an autapse policy or cleft-score threshold for the captured bundle, and permits synthetic fallback synapses without per-event provenance accounting. Those limitations must be carried into every benchmark artifact.

Do not join FAFB root IDs directly to MANC, BANC, or FANC identifiers. FAFB application side labels are not anatomical truth; the locked anatomical convention is higher soma `x` = fly-left and lower soma `x` = fly-right. Any laterality-dependent score must resolve anatomy using that convention rather than the raw application label.

The public FlyWire service manifest/status was checked on 2026-07-18. No new root IDs, partner counts, or synapse weights were inferred for this proposal. If the circuit graph is later rebuilt, record materialization, neuron universe, autapse policy, cleft-score threshold, coordinate source/units, transmitter/sign assumptions, synthetic-edge policy, and every cross-atlas mapping in the validation manifest.

## Minimal useful first milestone

A practical first milestone, before acquiring all raw data, would be a **provisional paper-reproduction run** with these constraints:

1. Implement the exact visual geometry and a documented m-sequence reconstruction.
2. Run a virtual-tether output through the real retina-to-NOD1-to-muscle-to-FlyBody path.
3. Produce unsmoothed figure and ground STAFs plus the gray-window control.
4. Evaluate only scale-free spatial complementarity, sign/mirror symmetry, causal cuts, and the published front/rear triangle-wave ordering.
5. Label the result provisional and block biological promotion until raw per-fly data and exact stimulus streams are obtained.

That milestone would already be substantially more informative than the existing canonical sweep: it would expose whether failures originate in visual encoding, figure/ground separation, motor transformation, mechanics, or the live feedback boundary, while keeping the final validation target scientifically honest.

## References

- Fox JL, Aptekar JW, Zolotova NM, Shoemaker PA, Frye MA. 2014. [Figure–ground discrimination behavior in *Drosophila*. I. Spatial organization of wing-steering responses](https://journals.biologists.com/jeb/article/217/4/558/12927/Figure-ground-discrimination-behavior-in). *J Exp Biol* 217:558–569. DOI: [`10.1242/jeb.097220`](https://doi.org/10.1242/jeb.097220).
- [Open-access full text and article metadata at PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3922833/).
- Local implementation scope: [README](../README.md), [fly-FGS source manifest](../data/reference/fly_fgs/source_manifest.v1.json), [evaluation framework](evaluation.md), and [calibration roadmap](calibration-roadmap.md).
