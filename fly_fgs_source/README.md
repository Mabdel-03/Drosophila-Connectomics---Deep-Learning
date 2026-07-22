# `/fly-fgs/` source snapshot

This directory is a local snapshot of the deployed [Drosophila Figure–Ground Circuit](http://54.160.228.98/fly-fgs/) page and its browser-side dependencies, fetched on 2026-07-20 UTC.

Contents:

- `index.html` — main Three.js flight page and its inline control/visualization code.
- `map/index.html` — linked retinotopic T4a/LLPC1 activity-map page.
- `lib/fgmodel.js` — conductance-based point/cable circuit engine.
- `lib/three.min.js` and `lib/OrbitControls.js` — vendored Three.js dependencies loaded by the main page.
- `data/bundle_fg.json` — prepared FAFB-derived circuit cells, cable morphologies, and synaptic events.
- `data/wing_dns.json` — MANC `male-cns:v1.0` DN-to-wing-muscle table used by the page's downstream controller.
- `provenance/` — live FlyWire manifest/status responses and the file inventory below.

The page now starts in the paper-assay mode defined by
[`figure_ground_relative_motion_simulation_spec.md`](../figure_ground_relative_motion_simulation_spec.md):
the Figure 3a/3b/3c motions are analytic and open loop, the body stays fixed,
and the renderer and T4a samplers share one deterministic 3° random-dot
compositor.  The original moving-bar, closed-loop tracking application remains
available under **legacy tracking**.

Without a registered native artifact, paper mode is explicitly a live
arbitrary-unit circuit preview. A `data/paper_replay_index.json` can register
one immutable replay and manifest per Figure 3 protocol (the original single
replay filenames remain a compatibility fallback). Only artifacts marked
`authoritative_native_torque` after load-cell calibration, equality-meter,
sampling, convergence, and apparatus gates can display physical units. The UI
then shows selected raw trials, mean ± simulation SEM, the digitized paper
trace with a separate uncertainty band, all named filter sensitivities,
harmonic metrics, and measured FlyBody wing-joint animation. Schema-v3 replays
also show left/right complete wing-root torque, their sum, the whole-fly-minus-
wing residual, and optional aerodynamic-only diagnostics in a dedicated chart.
Figure 3 scoring remains restricted to authoritative whole-fly tether torque
because the paper contains no per-wing trace. V2 total-only replays remain
readable and label wing channels unavailable. No paper SEM, amplitude fit,
offset fit, time shift, or time warp is implied.

The **Circuit neurons** tab is live in both paper modes. It retains the original
retina, T4a/T5a, cable-voltage, LLPC1, and descending-neuron display; adds exact
per-cell voltage/activity readouts for the eight vCH/DCH/NOD1 cable cells; and
lets a user click a T4a/T5a or LLPC1 map point to inspect its root ID and state.
The browser muscle preview uses only the four NOD1 cells, the versioned
FAFB-v783 soma-side receipt, and the original type-level MANC motor table. It is
labelled arbitrary-unit and never replaces native replay torque or wing traces.

To inspect the page locally, serve this directory over HTTP so the relative JSON and JavaScript paths work:

```bash
cd fly_fgs_source
python3 -m http.server 8000
```

Then open `http://127.0.0.1:8000/`. The snapshot does not include the separate `map/` deployment's external `nod1_sim` link, and no source license was declared in the captured assets. The data bundle is a prepared, downsampled model input; it is not a complete export of the FlyWire connectome.

Run the static-browser unit and integration tests with:

```bash
node --test tests/*.test.mjs
```

The main report is one directory above: [FLY_FGS_MODEL_REPORT.md](../FLY_FGS_MODEL_REPORT.md).
