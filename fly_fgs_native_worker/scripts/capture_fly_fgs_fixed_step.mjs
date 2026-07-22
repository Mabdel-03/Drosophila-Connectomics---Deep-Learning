#!/usr/bin/env node
/**
 * Deterministically execute the content-addressed fly-FGS circuit boundary.
 *
 * This runner deliberately loads only fgmodel.js and bundle_fg.json.  The
 * captured page and wing-DN file are verified as source receipts but never
 * imported into the executable model.  In particular, none of the legacy
 * DN, motor, muscle, wing, yaw, or toy-body equations can enter this output.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MANIFEST = path.resolve(
  SCRIPT_DIR,
  "../data/reference/fly_fgs/source_manifest.v1.json",
);
const FIXTURE_KIND = "fly_fgs_fixed_step_circuit_capture";
const SCHEMA_VERSION = "1.0.0";

function fail(message) {
  throw new Error(message);
}

function requireCondition(condition, message) {
  if (!condition) fail(message);
}

function parseArguments(argv) {
  const options = {
    manifest: DEFAULT_MANIFEST,
    output: null,
    includeFullCellState: true,
    durationS: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--manifest" || argument === "--output" || argument === "--duration-s") {
      requireCondition(index + 1 < argv.length, `${argument} requires a value`);
      const value = argv[++index];
      if (argument === "--manifest") options.manifest = path.resolve(value);
      else if (argument === "--output") options.output = path.resolve(value);
      else options.durationS = Number(value);
    } else if (argument === "--include-full-cell-state") {
      options.includeFullCellState = true;
    } else if (argument === "--omit-full-cell-state") {
      options.includeFullCellState = false;
    } else if (argument === "--help") {
      process.stdout.write(
        "Usage: capture_fly_fgs_fixed_step.mjs [--manifest PATH] [--output PATH] "
          + "[--duration-s SECONDS] [--include-full-cell-state|--omit-full-cell-state]\n",
      );
      process.exit(0);
    } else {
      fail(`unknown argument: ${argument}`);
    }
  }
  if (options.durationS !== null) {
    requireCondition(
      Number.isFinite(options.durationS) && options.durationS > 0,
      "--duration-s must be a positive finite number",
    );
  }
  return options;
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function sortedValue(value) {
  if (Array.isArray(value)) return value.map(sortedValue);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, sortedValue(value[key])]),
    );
  }
  return value;
}

function canonicalBytes(value) {
  return Buffer.from(JSON.stringify(sortedValue(value)), "utf8");
}

function stablePrettyJson(value) {
  return `${JSON.stringify(sortedValue(value), null, 2)}\n`;
}

function exactKeys(value, expected, label) {
  requireCondition(value !== null && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  requireCondition(JSON.stringify(actual) === JSON.stringify(wanted), `${label} fields do not match the registered contract`);
}

function finiteNumber(value, label) {
  requireCondition(typeof value === "number" && Number.isFinite(value), `${label} must be finite`);
  return value;
}

function resolveRegularAsset(manifestPath, entry) {
  exactKeys(
    entry,
    ["asset_id", "path", "sha256", "bytes", "media_type", "role", "eligible_circuit_input"],
    `asset ${entry?.asset_id ?? "unknown"}`,
  );
  requireCondition(typeof entry.path === "string" && entry.path.length > 0, "asset path is required");
  requireCondition(!path.isAbsolute(entry.path), `asset ${entry.asset_id} path must be relative`);
  const manifestDirectory = path.dirname(manifestPath);
  const resolved = path.resolve(manifestDirectory, entry.path);
  const relative = path.relative(manifestDirectory, resolved);
  requireCondition(relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative), `asset ${entry.asset_id} escapes the manifest directory`);
  const stat = fs.lstatSync(resolved);
  requireCondition(stat.isFile() && !stat.isSymbolicLink(), `asset ${entry.asset_id} must be a regular non-symlink file`);
  const bytes = fs.readFileSync(resolved);
  requireCondition(bytes.length === entry.bytes, `asset ${entry.asset_id} byte count mismatch`);
  requireCondition(sha256(bytes) === entry.sha256, `asset ${entry.asset_id} SHA-256 mismatch`);
  return { entry, path: resolved, bytes };
}

function validateManifest(manifestPath) {
  const manifestBytes = fs.readFileSync(manifestPath);
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  exactKeys(
    manifest,
    [
      "schema_version",
      "snapshot_id",
      "source",
      "dataset",
      "assets",
      "circuit_inventory",
      "execution_contract",
      "rejected_downstream_fields",
      "claim_scope",
    ],
    "source manifest",
  );
  requireCondition(manifest.schema_version === SCHEMA_VERSION, "unsupported source-manifest schema version");
  requireCondition(Array.isArray(manifest.assets), "source manifest assets must be an array");
  const assets = new Map();
  for (const entry of manifest.assets) {
    requireCondition(typeof entry.asset_id === "string" && !assets.has(entry.asset_id), "asset IDs must be unique strings");
    assets.set(entry.asset_id, resolveRegularAsset(manifestPath, entry));
  }
  for (const required of ["page_snapshot", "circuit_engine", "circuit_bundle", "wing_dns_excluded"]) {
    requireCondition(assets.has(required), `source manifest is missing ${required}`);
  }
  const eligible = [...assets.values()].filter(({ entry }) => entry.eligible_circuit_input);
  requireCondition(
    eligible.length === 2
      && eligible.some(({ entry }) => entry.asset_id === "circuit_engine")
      && eligible.some(({ entry }) => entry.asset_id === "circuit_bundle"),
    "only the circuit engine and circuit bundle may be eligible circuit inputs",
  );
  requireCondition(
    assets.get("page_snapshot").entry.eligible_circuit_input === false
      && assets.get("wing_dns_excluded").entry.eligible_circuit_input === false,
    "page and wing-DN assets must remain receipt-only",
  );
  return {
    manifest,
    manifestSha256: sha256(manifestBytes),
    assets,
  };
}

function validateBundle(bundle, manifest) {
  exactKeys(
    bundle,
    ["cableIndices", "cables", "cells", "dt", "events", "ids", "meta", "modelParams"],
    "circuit bundle",
  );
  const inventory = manifest.circuit_inventory;
  requireCondition(bundle.dt === manifest.execution_contract.dt_s, "bundle timestep mismatch");
  requireCondition(Array.isArray(bundle.ids) && bundle.ids.length === inventory.cell_count, "bundle ID inventory mismatch");
  requireCondition(Array.isArray(bundle.cells) && bundle.cells.length === inventory.cell_count, "bundle cell inventory mismatch");
  requireCondition(bundle.meta.nCells === inventory.cell_count, "bundle meta cell count mismatch");
  requireCondition(inventory.event_count === inventory.raw_event_count, "manifest event_count must name the raw bundle count");
  requireCondition(bundle.meta.nEvents === inventory.raw_event_count, "bundle meta raw-event count mismatch");
  requireCondition(bundle.meta.nCables === inventory.cable_count, "bundle meta cable count mismatch");
  requireCondition(JSON.stringify(bundle.cableIndices) === JSON.stringify(inventory.cable_indices), "bundle cable indices mismatch");
  requireCondition(
    JSON.stringify(sortedValue(bundle.modelParams))
      === JSON.stringify(sortedValue(inventory.model_parameters)),
    "bundle model-parameter mapping mismatch",
  );
  requireCondition(
    sha256(canonicalBytes(bundle.modelParams))
      === inventory.model_parameters_js_sorted_json_sha256,
    "bundle JavaScript-sorted model-parameter digest mismatch",
  );
  requireCondition(sha256(canonicalBytes(bundle.ids)) === inventory.cell_ids_canonical_sha256, "bundle cell-ID digest mismatch");
  const typeCounts = {};
  for (let index = 0; index < bundle.cells.length; index += 1) {
    const cell = bundle.cells[index];
    requireCondition(cell.id === bundle.ids[index], `bundle cell/ID mismatch at index ${index}`);
    typeCounts[cell.type] = (typeCounts[cell.type] ?? 0) + 1;
  }
  requireCondition(JSON.stringify(sortedValue(typeCounts)) === JSON.stringify(sortedValue(inventory.cell_type_counts)), "bundle cell-type inventory mismatch");
  requireCondition((typeCounts.T5 ?? 0) === inventory.t5_cell_count, "bundle T5 inventory mismatch");
  exactKeys(bundle.events, ["gunit", "post", "postc", "pre", "prec", "sign"], "bundle events");
  for (const [name, vector] of Object.entries(bundle.events)) {
    requireCondition(Array.isArray(vector) && vector.length === inventory.raw_event_count, `bundle event vector ${name} mismatch`);
  }
  const llpcIndices = new Set();
  bundle.cells.forEach((cell, index) => {
    if (String(cell.type ?? "").toUpperCase().includes("LLPC1")) llpcIndices.add(index);
  });
  let removedLlpc1ToLlpc1 = 0;
  for (let index = 0; index < inventory.raw_event_count; index += 1) {
    if (llpcIndices.has(bundle.events.pre[index]) && llpcIndices.has(bundle.events.post[index])) {
      removedLlpc1ToLlpc1 += 1;
    }
  }
  requireCondition(
    removedLlpc1ToLlpc1 === inventory.removed_llpc1_to_llpc1_event_count,
    "LLPC1-to-LLPC1 removed-event count mismatch",
  );
  requireCondition(
    inventory.raw_event_count - removedLlpc1ToLlpc1 === inventory.effective_runtime_event_count,
    "effective runtime-event count mismatch",
  );
  const byRoot = new Map();
  bundle.cells.forEach((cell, index) => {
    if (cell.type === "Nod1") byRoot.set(String(cell.id).replace(/^r/, ""), { cell, index });
  });
  requireCondition(
    JSON.stringify([...byRoot.keys()]) === JSON.stringify(inventory.nod1_root_ids),
    "bundle NOD1 root inventory/order mismatch",
  );
  for (const rootId of inventory.nod1_root_ids) {
    requireCondition(byRoot.get(rootId).cell.side === inventory.nod1_raw_app_sides[rootId], `bundle NOD1 app side mismatch for ${rootId}`);
  }
  return byRoot;
}

function appendPooledTrace(target, readout) {
  const pairs = [
    ["nod1_activity", "nod1L", "nod1R", 1],
    ["vch_activity", "vchL", "vchR", 1],
    ["dch_activity", "dchL", "dchR", 1],
    ["t4_activity", "t4L", "t4R", 1],
    ["llpc1_activity", "llpcL", "llpcR", 1],
  ];
  for (const [name, leftKey, rightKey, scale] of pairs) {
    target[name].raw_app_L.push(finiteNumber(readout[leftKey], `${name}.raw_app_L`) * scale);
    target[name].raw_app_R.push(finiteNumber(readout[rightKey], `${name}.raw_app_R`) * scale);
  }
  for (const [name, leftKey, rightKey] of [
    ["nod1_voltage_v", "nod1L", "nod1R"],
    ["vch_voltage_v", "vchL", "vchR"],
    ["dch_voltage_v", "dchL", "dchR"],
  ]) {
    target[name].raw_app_L.push(finiteNumber(readout.mv[leftKey], `${name}.raw_app_L`) / 1000);
    target[name].raw_app_R.push(finiteNumber(readout.mv[rightKey], `${name}.raw_app_R`) / 1000);
  }
}

function executeCapture(validated, options) {
  const { manifest, assets, manifestSha256 } = validated;
  const engineAsset = assets.get("circuit_engine");
  const bundleAsset = assets.get("circuit_bundle");
  const bundle = JSON.parse(bundleAsset.bytes.toString("utf8"));
  const nod1ByRoot = validateBundle(bundle, manifest);
  const require = createRequire(import.meta.url);
  delete require.cache[engineAsset.path];
  const engine = require(engineAsset.path);
  for (const method of manifest.execution_contract.engine_api) {
    requireCondition(typeof engine[method] === "function", `circuit engine API is missing ${method}`);
  }

  const contract = manifest.execution_contract;
  const dtS = finiteNumber(contract.dt_s, "execution dt_s");
  const durationS = options.durationS ?? finiteNumber(contract.duration_s, "execution duration_s");
  const stepRatio = durationS / dtS;
  const sampleCount = Math.round(stepRatio);
  requireCondition(Math.abs(stepRatio - sampleCount) <= 1e-12, "duration must be an integer multiple of circuit dt");
  if (options.durationS === null) {
    requireCondition(sampleCount === contract.sample_count, "registered sample count mismatch");
  }
  const configured = contract.options;
  const state = engine.init(bundle, {
    dt: dtS,
    figW: configured.figure_width_deg,
    figH: configured.figure_height_deg,
    figOn: configured.figure_enabled,
    gratingOn: configured.grating_enabled,
    gratingPeriod: configured.grating_period_deg,
    inputScale: configured.input_scale,
    scale: configured.retinal_texture_scale,
    llpcBoost: configured.llpc_boost,
  });
  state.ablateWf = configured.ablate_vch_dch;
  requireCondition(
    state.nEv === manifest.circuit_inventory.effective_runtime_event_count,
    "engine runtime-event count differs from the registered post-filter count",
  );

  const timeS = [];
  const nod1VoltageV = Object.fromEntries(
    manifest.circuit_inventory.nod1_root_ids.map((rootId) => [rootId, []]),
  );
  const pooled = Object.fromEntries(
    [
      "nod1_activity",
      "nod1_voltage_v",
      "vch_activity",
      "vch_voltage_v",
      "dch_activity",
      "dch_voltage_v",
      "t4_activity",
      "llpc1_activity",
    ].map((name) => [name, { raw_app_L: [], raw_app_R: [] }]),
  );
  const schedule = {
    figure_world_azimuth_deg: [],
    figure_velocity_deg_s: [],
    figure_retinal_azimuth_deg: [],
    heading_deg: [],
    heading_velocity_deg_s: [],
    grating_velocity_deg_s: [],
    grating_world_displacement_deg: [],
  };
  const retinalAzimuthDeg = Array.from({ length: 180 }, (_, index) => -140 + (280 * index) / 179);
  const retinalLuminance = [];
  const retinotopicT4a = bundle.cells
    .map((cell, index) => ({ cell, index }))
    .filter(({ cell }) => (
      cell.type === "T4a"
      && cell.retinotopic === true
      && Number.isFinite(cell.retinoAzimuth)
      && Number.isFinite(cell.retinoElevation)
    ));
  requireCondition(
    retinotopicT4a.length === manifest.circuit_inventory.retinotopic_t4a_input_count,
    "retinotopic T4a input-grid count mismatch",
  );
  const retinalInput = {
    cell_indices: retinotopicT4a.map(({ index }) => index),
    cell_ids: retinotopicT4a.map(({ cell }) => cell.id),
    azimuth_deg: retinotopicT4a.map(({ cell }) => cell.retinoAzimuth),
    elevation_deg: retinotopicT4a.map(({ cell }) => cell.retinoElevation),
    luminance: [],
    unit: "1",
    sampling_semantics: (
      "exact endpoint lumNow value at each retinotopic T4a circuit coordinate; "
      + "analytic scene input, not calibrated compound-eye optics or biological ommatidia"
    ),
  };
  const fullCellState = options.includeFullCellState
    ? { cell_ids: [...bundle.ids], voltage_v: [], activity: [] }
    : null;

  let figureAzimuthDeg = configured.figure_initial_world_azimuth_deg;
  let figureDirection = configured.figure_speed_deg_s >= 0 ? 1 : -1;
  const figureSpeedMagnitude = Math.abs(configured.figure_speed_deg_s);
  const headingDeg = configured.heading_deg;
  const headingVelocityDegS = configured.heading_velocity_deg_s;

  const preRollSteps = contract.pre_roll_steps;
  requireCondition(
    Number.isInteger(preRollSteps) && preRollSteps > 0,
    "registered pre-roll step count must be a positive integer",
  );
  requireCondition(
    Math.abs(preRollSteps * dtS - contract.pre_roll_duration_s) <= 1e-15,
    "registered pre-roll duration does not match its fixed-step count",
  );
  for (let preRollIndex = 0; preRollIndex < preRollSteps; preRollIndex += 1) {
    engine.step(state, {
      heading: headingDeg,
      headingVel: 0,
      figWorldAz: figureAzimuthDeg,
      figVelWorld: 0,
      groundVelWorld: 0,
    });
  }

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    let figureVelocityDegS = figureDirection * figureSpeedMagnitude;
    const gratingVelocityDegS = configured.grating_enabled
      ? configured.grating_velocity_deg_s
      : 0;
    if (sampleIndex > 0) {
      figureAzimuthDeg += figureVelocityDegS * dtS;
      if (figureAzimuthDeg > configured.figure_bounce_limit_deg) {
        figureAzimuthDeg = configured.figure_bounce_limit_deg;
        figureDirection = -1;
      }
      if (figureAzimuthDeg < -configured.figure_bounce_limit_deg) {
        figureAzimuthDeg = -configured.figure_bounce_limit_deg;
        figureDirection = 1;
      }
      figureVelocityDegS = figureDirection * figureSpeedMagnitude;
      engine.step(state, {
        heading: headingDeg,
        headingVel: headingVelocityDegS,
        figWorldAz: figureAzimuthDeg,
        figVelWorld: figureVelocityDegS,
        groundVelWorld: gratingVelocityDegS,
      });
    }

    const sampleTimeS = sampleIndex * dtS;
    timeS.push(sampleTimeS);
    const readout = engine.readout(state);
    appendPooledTrace(pooled, readout);
    for (const rootId of manifest.circuit_inventory.nod1_root_ids) {
      const voltage = finiteNumber(state.voltage[nod1ByRoot.get(rootId).index], `NOD1 ${rootId} voltage`);
      nod1VoltageV[rootId].push(voltage);
    }
    schedule.figure_world_azimuth_deg.push(configured.figure_enabled ? figureAzimuthDeg : null);
    schedule.figure_velocity_deg_s.push(configured.figure_enabled ? figureVelocityDegS : 0);
    schedule.figure_retinal_azimuth_deg.push(configured.figure_enabled ? figureAzimuthDeg - headingDeg : null);
    schedule.heading_deg.push(headingDeg);
    schedule.heading_velocity_deg_s.push(headingVelocityDegS);
    schedule.grating_velocity_deg_s.push(gratingVelocityDegS);
    schedule.grating_world_displacement_deg.push(state.groundWorldDisp);
    retinalLuminance.push(retinalAzimuthDeg.map((azimuth) => engine.lumNow(state, azimuth, 0)));
    retinalInput.luminance.push(
      retinotopicT4a.map(({ cell }) => engine.lumNow(state, cell.retinoAzimuth, cell.retinoElevation)),
    );
    if (fullCellState !== null) {
      fullCellState.voltage_v.push(Array.from(state.voltage));
      fullCellState.activity.push(Array.from(state.act));
    }
  }

  const t4Values = [
    ...pooled.t4_activity.raw_app_L,
    ...pooled.t4_activity.raw_app_R,
  ];
  const nod1Values = Object.values(nod1VoltageV).flat();
  const invariantObservations = {
    t4_activity_excursion: Math.max(...t4Values) - Math.min(...t4Values),
    nod1_voltage_excursion_v: Math.max(...nod1Values) - Math.min(...nod1Values),
  };
  if (contract.capture_invariants.requires_nonzero_t4_activity_excursion) {
    requireCondition(
      invariantObservations.t4_activity_excursion > 1e-12,
      "canonical fly-FGS capture has a dead T4 activity trace",
    );
  }
  if (contract.capture_invariants.requires_nonzero_nod1_voltage_excursion) {
    requireCondition(
      invariantObservations.nod1_voltage_excursion_v > 1e-12,
      "canonical fly-FGS capture has a dead NOD1 voltage trace",
    );
  }

  return {
    schema_version: SCHEMA_VERSION,
    fixture_kind: FIXTURE_KIND,
    snapshot_id: manifest.snapshot_id,
    source_receipt: {
      source_manifest_sha256: manifestSha256,
      circuit_engine_sha256: engineAsset.entry.sha256,
      circuit_bundle_sha256: bundleAsset.entry.sha256,
    },
    execution: {
      dt_s: dtS,
      duration_s: durationS,
      sample_count: sampleCount,
      sample_interval: `half-open [0, ${durationS} s)`,
      sample_semantics: contract.sample_semantics,
      scheduler: contract.scheduler,
      pre_roll_steps: preRollSteps,
      pre_roll_duration_s: contract.pre_roll_duration_s,
      pre_roll_mode: contract.pre_roll_mode,
      invariant_observations: invariantObservations,
      time_s: timeS,
    },
    circuit_inventory: manifest.circuit_inventory,
    stimulus: {
      configuration: configured,
      schedule,
      retinal_display: {
        azimuth_deg: retinalAzimuthDeg,
        luminance: retinalLuminance,
        unit: "normalized luminance (1)",
      },
      retinal_input: retinalInput,
    },
    nod1_voltage_v: nod1VoltageV,
    pooled_readout_traces: pooled,
    rejected_downstream_fields: manifest.rejected_downstream_fields,
    full_cell_state: fullCellState,
  };
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  const validated = validateManifest(options.manifest);
  const capture = executeCapture(validated, options);
  const output = stablePrettyJson(capture);
  if (options.output === null) process.stdout.write(output);
  else fs.writeFileSync(options.output, output, { encoding: "utf8", flag: "w" });
}

try {
  main();
} catch (error) {
  process.stderr.write(`fly-FGS fixed-step capture failed: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
