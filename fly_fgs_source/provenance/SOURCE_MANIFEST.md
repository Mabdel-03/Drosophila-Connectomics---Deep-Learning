# Source snapshot manifest

- Page URL: `http://54.160.228.98/fly-fgs/`
- Map URL: `http://54.160.228.98/fly-fgs/map/`
- Fetch date: `2026-07-20 UTC`
- Dataset: FlyWire FAFB, materialization `783`
- Neuron universe: proofread `139255` (as reported by the live manifest)
- Main circuit data: prepared bundle, not a complete full-synapse export
- Downstream motor data: MANC `male-cns:v1.0`; separate identifier space from FAFB
- License: not declared in the captured assets

## Files and SHA-256

| Local file | Source URL | SHA-256 |
|---|---|---|
| `../index.html` | `http://54.160.228.98/fly-fgs/` | `f19434ee872984035fafbfb492538609cf458111ddb64ad9f998e6c371187932` |
| `../map/index.html` | `http://54.160.228.98/fly-fgs/map/` | `2f83566bfeaae599d7249d178241c1d21e0f74d436b7dfa1831589540937858f` |
| `../lib/three.min.js` | `http://54.160.228.98/fly-fgs/lib/three.min.js` | `9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2` |
| `../lib/OrbitControls.js` | `http://54.160.228.98/fly-fgs/lib/OrbitControls.js` | `02bb4ade710f3e607329e37a21f098bc3ac70eb6e33daf8a65e79f4db785e7b2` |
| `../lib/fgmodel.js` | `http://54.160.228.98/fly-fgs/lib/fgmodel.js?v=13` | `f6de20850d463399b308642222e0ef6dadf02129bf27440fd7f2d5cf7d45396f` |
| `../data/bundle_fg.json` | `http://54.160.228.98/fly-fgs/data/bundle_fg.json` | `dc7ad294e54df7f4b8d4556f36e95b2676e19135d150557b8a16cfc17a4ac0dc` |
| `../data/wing_dns.json` | `http://54.160.228.98/fly-fgs/data/wing_dns.json` | `b3ae85e6151228919617a87907730705ec3acb8d8b441465ce182926d6c32390` |

The main page and circuit assets match the workspace's content-addressed `fly-fgs-live-2026-07-18` receipt. Root IDs in the bundle are decimal strings. Raw application `L/R` labels are compatibility labels, not independently resolved anatomical hemispheres. The bundle does not encode a cleft-score threshold or an asserted autapse policy and permits synthetic fallback contacts without per-event provenance.

## Local paper-mode extension

The original hashes above document the captured `/fly-fgs/` baseline; they are
not hashes of the subsequently extended local page. Paper-mode monitoring adds
`lib/paper_motor_monitor.js`, `data/nod1_laterality.v1.json`, and tests. The
laterality receipt resolves only the four exact NOD1 roots from FAFB-v783 soma
coordinates (higher x = anatomical left). vCH/DCH labels remain unresolved, and
FAFB identifiers are never merged with MANC `male-cns:v1.0` body identifiers.

The live manifest/status were rechecked on 2026-07-21 UTC: materialization 783,
139,255 proofread roots, 144,248 catalog records, and 139,259 full skeleton
files. Structural counts are used only as relative exploratory weights; no
cleft-score threshold or autapse inference was added by this UI extension.

The physical-torque replay contract is now `paper_fgs_web_replay.v3`; v2
total-only replays remain readable. V3 accepts only checksum-bound
`authoritative_native_torque` artifacts produced by the structurally fixed
MuJoCo root load cell and cross-validated against the equality-row-only weld
reaction. It adds calibrated left/right complete wing-root wrench
decompositions, wing sum, non-wing residual, and validated aerodynamic-only
diagnostics. Figure 3 scoring remains restricted to the authoritative
whole-fly support torque because the paper has no per-wing trace. Named
low-pass and wingbeat-average products are sensitivity views; the unfitted
`paper_comparison` product remains primary. The browser continues to run the
FAFB-v783/MANC exploratory preview in parallel and never relabels it as native
torque.

The native execution source, locked paper inputs, resumable 1,200-trial release
runner, and complete operating instructions are stored outside this publicly
served directory in `../../fly_fgs_native_worker` and
`../AGENT_INSTRUCTIONS.md`, respectively. Raw native arrays are external
scientific artifacts and are never part of this captured browser-source
manifest.
