# Scientific fixture capture scripts

## Official FlyBody released-flight bundle intake

`intake_flybody_release.py` is a non-downloading, fail-closed intake for a
user-supplied extraction of Figshare DOI `10.25378/janelia.25309105` v4. It
locks archive file IDs `51196859` (flight imitation) and `44815195` (trained
policies), upstream commit `d015e9bfe441bd90ae431bac24c55cb74bdbce26`, and
license `GPL-3.0+`. The script recursively hashes every regular file and rejects
symlinks, non-regular or empty files, hard-linked duplicates, path collisions,
missing WPG/reference sentinels, and incomplete TensorFlow SavedModels. It has
no network or download operation.

Create an unreviewed candidate receipt outside the extracted tree:

```bash
python scripts/intake_flybody_release.py intake \
  --root /path/to/extracted-flybody-release \
  --roles /path/to/reviewed-path-role-map.json \
  --output /path/to/receipts/flybody-release-candidate.json
```

The optional role map is a JSON object from exact relative paths to `wpg`,
`reference`, `policy`, `normalization`, `inference`, `runtime`, or `supporting`.
Known WPG/reference sentinels and the `trained-fly-policies/flight/` tree are
classified automatically; explicit overrides are appropriate only when a
reviewer has identified normalization, inference, or runtime files. Every
candidate is permanently labelled `candidate_unreviewed`.

Readiness requires a separately authored `reviewed_expected` receipt containing
the exact complete path/hash/byte/role inventory, review decision receipt,
normalization mode, observation/action and inference code hashes, controller
clock, dependency lock, container digest, versions, platform, and physics step:

```bash
python scripts/intake_flybody_release.py verify \
  --root /path/to/extracted-flybody-release \
  --candidate /path/to/receipts/flybody-release-candidate.json \
  --expected /path/to/separately-reviewed-expected.json \
  --roles /path/to/reviewed-path-role-map.json
```

No expected receipt or upstream artifact hash is fabricated or checked into
this repository. Even an exact readiness match only prepares an offline
released-policy reproduction; its report always states that it does not modify,
pass, or unblock `physics.timestep_convergence`.

## Static web-release audit

`audit_web_release.py` is the fail-closed release gate for the generated replay
site. By default it audits `web/dist`: every manifest URL must resolve to a
contained local file; episode IDs must agree; each immutable run sidecar and
chunk digest is verified; every logical array is reconstructed through the
project artifact reader and checked for shape, dtype, and finite values; and
the validation report is byte-locked and canonically rebound to its exact
benchmark registry.

Run the static audit after `npm run build`:

```bash
python scripts/audit_web_release.py
```

The optional browser gate starts a loopback-only HTTP server in the same Python
process, hides `crypto.subtle` so the application's bundled SHA-256 fallback is
actually exercised, and checks the initial FlyBody replay, validation counts,
plot window, comparison disclosure, channel rows, requests, and browser errors:

```bash
python scripts/audit_web_release.py --browser \
  --expected-episodes 9 \
  --expected-neural-channels 29 \
  --expected-muscle-channels 14 \
  --expected-pass 14 --expected-blocked 4 --expected-fail 0 \
  --screenshot /tmp/fly-s2b-web-release.png
```

The auditor infers the evaluation ID, scientific result counts, initial
episode ID, and plot window from the attached artifacts. The optional expected
counts are release-policy assertions, not substitutes for the report binding.
It prints one compact JSON summary and exits nonzero on the first integrity or
browser failure.

## Canonical run publication

`publish_canonical_web_run.py` is the only supported bridge from a completed
canonical online run into `web/public`. It verifies the artifact's
`manifest.sha256`, the replay-to-run identity and manifest digest, and the
artifact-owned replay projection digest before copying anything. The immutable
run is installed under its content-derived ID, the replay is copied as a
separate derived file, and the public episode index is replaced last. A crash
can therefore leave only unreferenced bytes, not a manifest pointing at a
partial run.

```bash
python scripts/publish_canonical_web_run.py \
  --artifact-dir /tmp/canonical-online-baseline \
  --replay /tmp/canonical-online-baseline.web.json \
  --episode-id canonical-online-baseline \
  --condition "moving figure; no force-stage perturbation" \
  --description "Live fly-FGS circuit through the causal actuator and native FlyBody." \
  --color '#2dd4bf' \
  --position 0
```

An exact retry is idempotent. Reusing an episode ID for different bytes fails
unless `--replace-episode` is supplied explicitly. Publication does not alter
the run's scientific status, validation report, or projection, and the static
release auditor must still pass after the web build.

## Canonical fly-FGS fixed-step capture

`capture_fly_fgs_fixed_step.mjs` is the authoritative offline executor for the
canonical [`/fly-fgs`](http://54.160.228.98/fly-fgs/) scene/circuit boundary.
It verifies every content-addressed asset in
`data/reference/fly_fgs/source_manifest.v1.json`, imports only the registered
engine and circuit bundle, and refuses the page and `wing_dns` receipt as
executable inputs. Thus the source `SCALE_M`, DN gains/tables, muscle state,
wing equations, yaw, and toy body cannot enter the capture.

The registered program runs a 48-step (`0.24 s`) static-scene pre-roll and then
100 fixed 5 ms samples over `[0, 0.5 s)`. It records all 1,684 voltage/activity
states, the exact 1,441-point T4a luminance grid, pooled circuit traces, and four
NOD1 voltages. It has no T5, photoreceptor, or lamina model. Only those four
NOD1 voltages are eligible for the downstream motor bridge; all other arrays
are visualization/audit data. Raw bundle `L/R` remains a non-anatomical
application label.

Regenerate uncompressed candidates and prove actual Node reproducibility with
two independent executions:

```bash
node scripts/capture_fly_fgs_fixed_step.mjs \
  --output /tmp/fly-fgs-capture-a.json
node scripts/capture_fly_fgs_fixed_step.mjs \
  --output /tmp/fly-fgs-capture-b.json
sha256sum /tmp/fly-fgs-capture-a.json /tmp/fly-fgs-capture-b.json
cmp /tmp/fly-fgs-capture-a.json /tmp/fly-fgs-capture-b.json

gzip -n -9 -c /tmp/fly-fgs-capture-a.json \
  > /tmp/fixed_step_capture.v1.json.gz
sha256sum /tmp/fixed_step_capture.v1.json.gz
```

The reviewed registered hashes are source manifest
`fef59e4b8abe5216081b6d64decd0d1775454d06e7631bb9af8ae409c1cd05c3`
and compressed capture
`5612cc9c2218a917bf2ad939d6aaa2a8402db5f11fd2de36cd3168d9198cbd83`.
Never overwrite the registered fixture merely because a new execution differs:
audit the source change and create a new manifest/fixture/registry version.
`fly_fgs.fixed_step_integrity` validates the frozen bytes and projections; it
does not rerun Node and must not be reported as runtime reproducibility.

## Historical NOD1 browser/Python parity

`capture_nod1_browser_parity.py` captures an actual Chromium Web Worker run
from the frozen NOD1 simulator and reruns the exact browser-saved configuration
through its local Python backend. It deliberately does not call the public
time-advancing `/api/simulate` route.

Prerequisites:

```bash
python -m pip install playwright==1.60.0
playwright install chromium
```

Against the deployed reverse-proxy route (the container's direct root does not
rewrite the Vite base path):

```bash
python scripts/capture_nod1_browser_parity.py \
  --app-url http://54.160.228.98/nod1_sim_fix/ \
  --legacy-source /home/ec2-user/figure-ground-lab-nod1-sim-fix \
  --output /tmp/nod1-browser-python-parity.v1.json

# Create byte-reproducible compressed fixture bytes after reviewing the raw JSON.
gzip -n -9 -c /tmp/nod1-browser-python-parity.v1.json \
  > data/reference/nod1_browser_python_parity.v1.json.gz
sha256sum data/reference/nod1_browser_python_parity.v1.json.gz
```

The script refuses source drift by default. It verifies the hashes in
`data/manifests/legacy_nod1_v0.5.0.json`, records the circuit response,
prepared-bundle, request, browser worker, configuration, deployed index HTML,
bundled main JavaScript/CSS, complete application-source inventory, and
morphology receipts. It also checks the exact 1,208-cell/5,188-edge circuit and
53,715-event inventory, and preserves FlyWire root IDs as decimal strings.
Allowing a diagnostic drift capture cannot yield a passing fixture; the
registered evaluator independently requires strict source receipts and
recomputes differences from the stored arrays rather than trusting a stored
pass flag.

The currently registered gzip has SHA-256
`a76794e630533822468971cbdbe2164a3d1ddce8a8a38811c9b9c1ad5b97a8a4`.
Do not replace it in place after any deployed/source/configuration change:
capture a new fixture, review every receipt, and update the benchmark registry
as a new claim. The fixture establishes only cross-runtime numerical agreement.
It cannot validate circuit biology, legacy side labels, physiological synaptic
strengths, motor output, muscles, or behavior.

For a quick JavaScript-only preflight, Node can execute the pure numerical loop
after bundling its extensionless TypeScript imports:

```bash
/home/ec2-user/figure-ground-lab-nod1-sim-fix/node_modules/.bin/esbuild \
  /home/ec2-user/figure-ground-lab-nod1-sim-fix/src/sim/loop.ts \
  --bundle --platform=node --format=esm --outfile=/tmp/nod1-loop.mjs
node -e "import('/tmp/nod1-loop.mjs').then(m => console.log(Object.keys(m)))"
```

That Node check is not browser parity and must never be registered as such.

## Female BANC/FANC structural evidence

`extract_banc_fanc_wing_evidence.py` reads five exact upstream snapshots and
refuses any byte-count or SHA-256 drift before normalizing the female BANC v888
and FANC v840 wing-pathway evidence. With the reviewed sources at their default
temporary paths:

```bash
python scripts/extract_banc_fanc_wing_evidence.py \
  --output /tmp/banc_fanc_wing_pathway_evidence.v1.json
sha256sum /tmp/banc_fanc_wing_pathway_evidence.v1.json
```

The expected digest is
`91763c068c74520d9c9f1aadd0ef80a295bae01cefc95c9091014eb0a0859c21`.
The extractor recomputes all aggregate rows and emits decimal identifiers as
strings. Its registered BANC DNp26→wing-MN total is 129 synapses across 18
edges, not the earlier provisional 131. It deliberately performs no BANC/FANC
premotor matching: the crosswalk remains incomplete, matched-pair lists stay
empty, and topological candidates may not be promoted to identities. A new
source release requires a new fixture version rather than a drift override.

## FlyBody compatibility-fixture registration

`register_flybody_smoke_fixture.py` converts the raw pinned-worker smoke trace
into the current v3 fixture. It recomputes trace-derived summaries and adds an
exact/numeric comparison contract:

```bash
fly-s2b flybody-smoke --mode analytic-wingbeat --duration-s 0.005 \
  --output /tmp/flybody-smoke-raw.json
python scripts/register_flybody_smoke_fixture.py \
  --input /tmp/flybody-smoke-raw.json \
  --output /tmp/flybody_analytic_wingbeat_smoke.v3.json
sha256sum /tmp/flybody_analytic_wingbeat_smoke.v3.json
```

The expected v3 digest is
`9e58e24af449d1744fe11a52f3ce8e49d63de3dc3d17ead557526c667cee7acc`.
Exact fields fingerprint the worker versions/dependency records, compiled model
topology/options, wing/fluid identities, `0.2 ms` advance-then-hold control,
`0.05 ms` fixture step, collision-disabled fluid proxies, legs-only contact,
root-force sampling, and released-policy non-equivalence. Numeric summaries
have narrow platform tolerances. This is a compatibility registration, not a
stable-flight, policy, muscle-calibration, or biological fixture.
