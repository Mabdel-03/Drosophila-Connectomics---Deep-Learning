#!/usr/bin/env node
/**
 * Stateful RPC boundary for the content-addressed fly-FGS circuit engine.
 *
 * Only the two manifest assets marked as executable circuit inputs are read:
 * fgmodel.js and bundle_fg.json.  Receipt-only HTML and the excluded legacy
 * wing/DN source are intentionally never opened by this process.
 *
 * stdin/stdout are newline-delimited strict JSON.  Public controls use SI
 * radians even though the captured engine's private scene convention is in
 * degrees.  The process performs the registered 48-step static pre-roll and
 * exposes sample 0 before any caller-controlled transition.
 */

import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MANIFEST = path.resolve(
  SCRIPT_DIR,
  "../data/reference/fly_fgs/source_manifest.v1.json",
);
const CHECKPOINT_SCHEMA_VERSION = "1.0.0";
const PROTOCOL_VERSION = "1.0.0";

function fail(message) {
  throw new Error(message);
}

function requireCondition(condition, message) {
  if (!condition) fail(message);
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

function exactKeys(value, expected, label) {
  requireCondition(
    value !== null && typeof value === "object" && !Array.isArray(value),
    `${label} must be an object`,
  );
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  requireCondition(
    JSON.stringify(actual) === JSON.stringify(wanted),
    `${label} fields do not match the protocol`,
  );
}

function finiteNumber(value, label) {
  requireCondition(
    typeof value === "number" && Number.isFinite(value),
    `${label} must be a finite number`,
  );
  return value;
}

function parseArguments(argv) {
  const options = {
    manifest: DEFAULT_MANIFEST,
    manifestSha256: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--manifest" || argument === "--manifest-sha256") {
      requireCondition(index + 1 < argv.length, `${argument} requires a value`);
      const value = argv[++index];
      if (argument === "--manifest") options.manifest = path.resolve(value);
      else options.manifestSha256 = value;
    } else if (argument === "--help") {
      process.stdout.write(
        "Usage: fly_fgs_runtime_rpc.mjs --manifest PATH --manifest-sha256 HEX\n",
      );
      process.exit(0);
    } else {
      fail(`unknown argument: ${argument}`);
    }
  }
  requireCondition(
    typeof options.manifestSha256 === "string"
      && /^[0-9a-f]{64}$/.test(options.manifestSha256),
    "--manifest-sha256 must be a lowercase SHA-256 digest",
  );
  return options;
}

function regularAssetPath(manifestPath, entry) {
  requireCondition(
    typeof entry.path === "string" && entry.path.length > 0,
    `asset ${entry.asset_id} path is required`,
  );
  requireCondition(
    !path.isAbsolute(entry.path),
    `asset ${entry.asset_id} path must be relative`,
  );
  const directory = path.dirname(manifestPath);
  const resolved = path.resolve(directory, entry.path);
  const relative = path.relative(directory, resolved);
  requireCondition(
    relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative),
    `asset ${entry.asset_id} escapes the manifest directory`,
  );
  const stat = fs.lstatSync(resolved);
  requireCondition(
    stat.isFile() && !stat.isSymbolicLink(),
    `asset ${entry.asset_id} must be a regular non-symlink file`,
  );
  return resolved;
}

function loadRuntimeInputs(options) {
  requireCondition(os.endianness() === "LE", "runtime requires little-endian float64 encoding");
  const manifestStat = fs.lstatSync(options.manifest);
  requireCondition(
    manifestStat.isFile() && !manifestStat.isSymbolicLink(),
    "source manifest must be a regular non-symlink file",
  );
  const manifestBytes = fs.readFileSync(options.manifest);
  requireCondition(
    sha256(manifestBytes) === options.manifestSha256,
    "source manifest SHA-256 mismatch",
  );
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
  requireCondition(Array.isArray(manifest.assets), "manifest assets must be an array");
  const byId = new Map();
  for (const entry of manifest.assets) {
    requireCondition(
      entry !== null && typeof entry === "object" && !Array.isArray(entry),
      "asset receipt must be an object",
    );
    requireCondition(
      typeof entry.asset_id === "string" && !byId.has(entry.asset_id),
      "asset IDs must be unique strings",
    );
    byId.set(entry.asset_id, entry);
  }
  const eligible = manifest.assets.filter((entry) => entry.eligible_circuit_input === true);
  requireCondition(
    eligible.length === 2
      && eligible.some((entry) => entry.asset_id === "circuit_engine")
      && eligible.some((entry) => entry.asset_id === "circuit_bundle"),
    "only circuit_engine and circuit_bundle may be executable inputs",
  );
  requireCondition(
    byId.get("page_snapshot")?.eligible_circuit_input === false
      && byId.get("wing_dns_excluded")?.eligible_circuit_input === false,
    "receipt-only downstream assets must remain ineligible",
  );

  // Deliberately resolve and read only the two eligible assets.  Ineligible
  // receipts may be absent from a runtime-only deployment without changing
  // circuit execution.
  const loaded = {};
  for (const entry of eligible) {
    const assetPath = regularAssetPath(options.manifest, entry);
    const bytes = fs.readFileSync(assetPath);
    requireCondition(bytes.length === entry.bytes, `${entry.asset_id} byte count mismatch`);
    requireCondition(sha256(bytes) === entry.sha256, `${entry.asset_id} SHA-256 mismatch`);
    loaded[entry.asset_id] = { entry, path: assetPath, bytes };
  }

  const bundle = JSON.parse(loaded.circuit_bundle.bytes.toString("utf8"));
  const inventory = manifest.circuit_inventory;
  requireCondition(
    Array.isArray(bundle.cells) && bundle.cells.length === inventory.cell_count,
    "bundle cell inventory mismatch",
  );
  requireCondition(
    Array.isArray(bundle.ids) && bundle.ids.length === inventory.cell_count,
    "bundle ID inventory mismatch",
  );
  requireCondition(
    bundle.dt === manifest.execution_contract.dt_s,
    "bundle timestep mismatch",
  );
  requireCondition(
    bundle.meta?.nEvents === inventory.raw_event_count,
    "bundle raw-event inventory mismatch",
  );
  const nod1ByRoot = new Map();
  bundle.cells.forEach((cell, index) => {
    if (cell.type === "Nod1") {
      nod1ByRoot.set(String(cell.id).replace(/^r/, ""), { cell, index });
    }
  });
  requireCondition(
    JSON.stringify([...nod1ByRoot.keys()]) === JSON.stringify(inventory.nod1_root_ids),
    "bundle NOD1 inventory/order mismatch",
  );
  const retinotopicT4a = bundle.cells
    .map((cell, index) => ({ cell, index }))
    .filter(({ cell }) => (
      cell.type === "T4a"
      && cell.retinotopic === true
      && Number.isFinite(cell.retinoAzimuth)
      && Number.isFinite(cell.retinoElevation)
    ));
  requireCondition(
    retinotopicT4a.length === inventory.retinotopic_t4a_input_count,
    "retinotopic T4a inventory mismatch",
  );

  const require = createRequire(import.meta.url);
  delete require.cache[loaded.circuit_engine.path];
  const engine = require(loaded.circuit_engine.path);
  for (const method of manifest.execution_contract.engine_api) {
    requireCondition(typeof engine[method] === "function", `engine API is missing ${method}`);
  }
  return {
    manifest,
    manifestSha256: options.manifestSha256,
    engineSha256: loaded.circuit_engine.entry.sha256,
    bundleSha256: loaded.circuit_bundle.entry.sha256,
    bundle,
    engine,
    nod1ByRoot,
    retinotopicT4a,
  };
}

function encodeFloat64(array) {
  requireCondition(array instanceof Float64Array, "checkpoint array must be Float64Array");
  return {
    dtype: "float64_le",
    length: array.length,
    data_base64: Buffer.from(array.buffer, array.byteOffset, array.byteLength).toString("base64"),
  };
}

function decodeFloat64(encoded, label) {
  exactKeys(encoded, ["dtype", "length", "data_base64"], label);
  requireCondition(encoded.dtype === "float64_le", `${label} dtype mismatch`);
  requireCondition(
    Number.isInteger(encoded.length) && encoded.length >= 0,
    `${label} length must be a non-negative integer`,
  );
  requireCondition(typeof encoded.data_base64 === "string", `${label} data must be base64 text`);
  const bytes = Buffer.from(encoded.data_base64, "base64");
  requireCondition(bytes.length === encoded.length * 8, `${label} byte length mismatch`);
  // Buffer storage is not guaranteed to be 8-byte aligned.  Copy into an
  // owned ArrayBuffer before creating the Float64Array view.
  const storage = new ArrayBuffer(bytes.length);
  new Uint8Array(storage).set(bytes);
  const result = new Float64Array(storage);
  requireCondition(
    Array.from(result).every(Number.isFinite),
    `${label} contains non-finite values`,
  );
  return result;
}

function copyInto(target, encoded, label) {
  const values = decodeFloat64(encoded, label);
  requireCondition(values.length === target.length, `${label} shape mismatch`);
  target.set(values);
}

function parseControl(value) {
  exactKeys(
    value,
    [
      "heading_rad",
      "heading_velocity_rad_s",
      "figure_world_azimuth_rad",
      "figure_velocity_rad_s",
      "ground_velocity_rad_s",
    ],
    "scene/body control",
  );
  const result = {};
  for (const key of Object.keys(value)) result[key] = finiteNumber(value[key], key);
  return result;
}

function controlToPrivateDegrees(control) {
  const factor = 180 / Math.PI;
  return {
    heading: control.heading_rad * factor,
    headingVel: control.heading_velocity_rad_s * factor,
    figWorldAz: control.figure_world_azimuth_rad * factor,
    figVelWorld: control.figure_velocity_rad_s * factor,
    groundVelWorld: control.ground_velocity_rad_s * factor,
  };
}

function finiteTree(value, label) {
  if (typeof value === "number") return finiteNumber(value, label);
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, finiteTree(child, `${label}.${key}`)]),
    );
  }
  fail(`${label} must contain only finite numbers and objects`);
}

class Runtime {
  constructor(inputs) {
    this.inputs = inputs;
    this.state = null;
    this.sampleIndex = null;
    this.lastControl = null;
  }

  initialControl() {
    const configured = this.inputs.manifest.execution_contract.options;
    return {
      heading_rad: configured.heading_deg * Math.PI / 180,
      heading_velocity_rad_s: 0,
      figure_world_azimuth_rad: configured.figure_initial_world_azimuth_deg * Math.PI / 180,
      figure_velocity_rad_s: 0,
      ground_velocity_rad_s: 0,
    };
  }

  resetState() {
    const { manifest, bundle, engine } = this.inputs;
    const contract = manifest.execution_contract;
    const configured = contract.options;
    const state = engine.init(bundle, {
      dt: contract.dt_s,
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
      "engine runtime-event inventory mismatch",
    );
    const initial = this.initialControl();
    const privateControl = controlToPrivateDegrees(initial);
    for (let index = 0; index < contract.pre_roll_steps; index += 1) {
      engine.step(state, privateControl);
    }
    this.state = state;
    this.sampleIndex = 0;
    this.lastControl = initial;
  }

  requireInitialized() {
    requireCondition(this.state !== null, "runtime is not initialized");
  }

  sample(options = {}) {
    this.requireInitialized();
    const allowed = ["include_full_cell_state", "include_retinal_input"];
    requireCondition(
      options !== null && typeof options === "object" && !Array.isArray(options),
      "sample options must be an object",
    );
    requireCondition(
      Object.keys(options).every((key) => allowed.includes(key)),
      "sample options contain an unknown field",
    );
    const includeFull = options.include_full_cell_state === true;
    const includeRetinal = options.include_retinal_input === true;
    requireCondition(
      options.include_full_cell_state === undefined || typeof options.include_full_cell_state === "boolean",
      "include_full_cell_state must be boolean",
    );
    requireCondition(
      options.include_retinal_input === undefined || typeof options.include_retinal_input === "boolean",
      "include_retinal_input must be boolean",
    );
    const { manifest, engine, nod1ByRoot, retinotopicT4a } = this.inputs;
    const readout = finiteTree(engine.readout(this.state), "pooled_readout");
    const nod1 = {};
    for (const rootId of manifest.circuit_inventory.nod1_root_ids) {
      nod1[rootId] = finiteNumber(
        this.state.voltage[nod1ByRoot.get(rootId).index],
        `NOD1 ${rootId} voltage`,
      );
    }
    const result = {
      sample_index: this.sampleIndex,
      measurement_time_s: this.sampleIndex * manifest.execution_contract.dt_s,
      availability_time_s: this.sampleIndex * manifest.execution_contract.dt_s,
      nod1_voltage_v: nod1,
      pooled_readout: readout,
      last_control: this.lastControl,
      full_cell_state_eligible_motor_input: false,
    };
    if (includeRetinal) {
      result.retinal_input_luminance = retinotopicT4a.map(({ cell }) => (
        finiteNumber(
          engine.lumNow(this.state, cell.retinoAzimuth, cell.retinoElevation),
          `retinal luminance ${cell.id}`,
        )
      ));
    }
    if (includeFull) {
      result.full_cell_voltage_v = Array.from(this.state.voltage);
      result.full_cell_activity = Array.from(this.state.act);
    }
    return result;
  }

  initialize(params) {
    exactKeys(params, ["sample_options"], "initialize params");
    this.resetState();
    return this.sample(params.sample_options);
  }

  advance(params) {
    exactKeys(params, ["control", "sample_options"], "advance params");
    this.requireInitialized();
    const maxIndex = this.inputs.manifest.execution_contract.sample_count - 1;
    requireCondition(this.sampleIndex < maxIndex, "registered circuit sample interval is exhausted");
    const control = parseControl(params.control);
    this.inputs.engine.step(this.state, controlToPrivateDegrees(control));
    this.sampleIndex += 1;
    this.lastControl = control;
    return this.sample(params.sample_options);
  }

  dynamicStatePayload() {
    this.requireInitialized();
    const cableV = {};
    for (const index of this.state.cableIndices) {
      cableV[String(index)] = encodeFloat64(this.state.cable_v[index]);
    }
    return {
      sample_index: this.sampleIndex,
      last_control: this.lastControl,
      scalars: {
        dF: finiteNumber(this.state.dF, "dF"),
        dG: finiteNumber(this.state.dG, "dG"),
        groundWorldDisp: finiteNumber(this.state.groundWorldDisp, "groundWorldDisp"),
        figAz: finiteNumber(this.state.figAz, "figAz"),
        figVelRet: finiteNumber(this.state.figVelRet, "figVelRet"),
        groundVelRet: finiteNumber(this.state.groundVelRet, "groundVelRet"),
        ablateWf: this.state.ablateWf === true,
      },
      arrays: {
        voltage: encodeFloat64(this.state.voltage),
        act: encodeFloat64(this.state.act),
        ev_s: encodeFloat64(this.state.ev_s),
        ev_dep: encodeFloat64(this.state.ev_dep),
        ext_lp: encodeFloat64(this.state.ext_lp),
        external: encodeFloat64(this.state.external),
      },
      cable_v: cableV,
      ring: {
        dF: encodeFloat64(this.state.ring.dF),
        dG: encodeFloat64(this.state.ring.dG),
        head: this.state.ring.head,
        len: this.state.ring.len,
        capacity: this.state.ring.N,
      },
    };
  }

  checkpoint() {
    const payload = {
      schema_version: CHECKPOINT_SCHEMA_VERSION,
      protocol_version: PROTOCOL_VERSION,
      snapshot_id: this.inputs.manifest.snapshot_id,
      source_manifest_sha256: this.inputs.manifestSha256,
      circuit_engine_sha256: this.inputs.engineSha256,
      circuit_bundle_sha256: this.inputs.bundleSha256,
      dt_s: this.inputs.manifest.execution_contract.dt_s,
      float_encoding: "float64_le_base64",
      dynamic_state: this.dynamicStatePayload(),
    };
    return {
      ...payload,
      payload_sha256: sha256(canonicalBytes(payload)),
    };
  }

  restore(params) {
    exactKeys(params, ["checkpoint", "sample_options"], "restore params");
    const checkpoint = params.checkpoint;
    exactKeys(
      checkpoint,
      [
        "schema_version",
        "protocol_version",
        "snapshot_id",
        "source_manifest_sha256",
        "circuit_engine_sha256",
        "circuit_bundle_sha256",
        "dt_s",
        "float_encoding",
        "dynamic_state",
        "payload_sha256",
      ],
      "checkpoint",
    );
    const payload = { ...checkpoint };
    delete payload.payload_sha256;
    requireCondition(
      checkpoint.payload_sha256 === sha256(canonicalBytes(payload)),
      "checkpoint payload SHA-256 mismatch",
    );
    requireCondition(checkpoint.schema_version === CHECKPOINT_SCHEMA_VERSION, "checkpoint schema mismatch");
    requireCondition(checkpoint.protocol_version === PROTOCOL_VERSION, "checkpoint protocol mismatch");
    requireCondition(checkpoint.snapshot_id === this.inputs.manifest.snapshot_id, "checkpoint snapshot mismatch");
    requireCondition(checkpoint.source_manifest_sha256 === this.inputs.manifestSha256, "checkpoint manifest mismatch");
    requireCondition(checkpoint.circuit_engine_sha256 === this.inputs.engineSha256, "checkpoint engine mismatch");
    requireCondition(checkpoint.circuit_bundle_sha256 === this.inputs.bundleSha256, "checkpoint bundle mismatch");
    requireCondition(checkpoint.dt_s === this.inputs.manifest.execution_contract.dt_s, "checkpoint timestep mismatch");
    requireCondition(checkpoint.float_encoding === "float64_le_base64", "checkpoint encoding mismatch");

    const dynamic = checkpoint.dynamic_state;
    exactKeys(dynamic, ["sample_index", "last_control", "scalars", "arrays", "cable_v", "ring"], "dynamic state");
    requireCondition(
      Number.isInteger(dynamic.sample_index)
        && dynamic.sample_index >= 0
        && dynamic.sample_index < this.inputs.manifest.execution_contract.sample_count,
      "checkpoint sample index is outside the registered interval",
    );
    const lastControl = parseControl(dynamic.last_control);
    exactKeys(
      dynamic.scalars,
      ["dF", "dG", "groundWorldDisp", "figAz", "figVelRet", "groundVelRet", "ablateWf"],
      "checkpoint scalars",
    );
    requireCondition(typeof dynamic.scalars.ablateWf === "boolean", "checkpoint ablateWf must be boolean");
    for (const key of ["dF", "dG", "groundWorldDisp", "figAz", "figVelRet", "groundVelRet"]) {
      finiteNumber(dynamic.scalars[key], `checkpoint ${key}`);
    }
    exactKeys(dynamic.arrays, ["voltage", "act", "ev_s", "ev_dep", "ext_lp", "external"], "checkpoint arrays");
    exactKeys(dynamic.ring, ["dF", "dG", "head", "len", "capacity"], "checkpoint ring");

    this.resetState();
    copyInto(this.state.voltage, dynamic.arrays.voltage, "checkpoint voltage");
    copyInto(this.state.act, dynamic.arrays.act, "checkpoint activity");
    copyInto(this.state.ev_s, dynamic.arrays.ev_s, "checkpoint synapse state");
    copyInto(this.state.ev_dep, dynamic.arrays.ev_dep, "checkpoint depression state");
    copyInto(this.state.ext_lp, dynamic.arrays.ext_lp, "checkpoint input lowpass");
    copyInto(this.state.external, dynamic.arrays.external, "checkpoint external drive");
    const expectedCableKeys = this.state.cableIndices.map(String).sort();
    requireCondition(
      JSON.stringify(Object.keys(dynamic.cable_v).sort()) === JSON.stringify(expectedCableKeys),
      "checkpoint cable inventory mismatch",
    );
    for (const index of this.state.cableIndices) {
      copyInto(this.state.cable_v[index], dynamic.cable_v[String(index)], `checkpoint cable ${index}`);
    }
    requireCondition(dynamic.ring.capacity === this.state.ring.N, "checkpoint ring capacity mismatch");
    requireCondition(
      Number.isInteger(dynamic.ring.head)
        && dynamic.ring.head >= 0
        && dynamic.ring.head < this.state.ring.N,
      "checkpoint ring head is invalid",
    );
    requireCondition(
      Number.isInteger(dynamic.ring.len)
        && dynamic.ring.len >= 0
        && dynamic.ring.len <= this.state.ring.N,
      "checkpoint ring length is invalid",
    );
    copyInto(this.state.ring.dF, dynamic.ring.dF, "checkpoint ring figure");
    copyInto(this.state.ring.dG, dynamic.ring.dG, "checkpoint ring ground");
    this.state.ring.head = dynamic.ring.head;
    this.state.ring.len = dynamic.ring.len;
    for (const [key, value] of Object.entries(dynamic.scalars)) {
      if (key !== "ablateWf") this.state[key] = value;
    }
    this.state.ablateWf = dynamic.scalars.ablateWf;
    this.sampleIndex = dynamic.sample_index;
    this.lastControl = lastControl;
    return this.sample(params.sample_options);
  }

  stateDigest() {
    const dynamic = this.dynamicStatePayload();
    return sha256(canonicalBytes(dynamic));
  }

  finalize(params) {
    exactKeys(params, [], "finalize params");
    this.requireInitialized();
    return {
      final_sample_index: this.sampleIndex,
      final_measurement_time_s: this.sampleIndex * this.inputs.manifest.execution_contract.dt_s,
      state_sha256: this.stateDigest(),
    };
  }
}

function readyPayload(inputs) {
  return {
    event: "ready",
    protocol_version: PROTOCOL_VERSION,
    node_version: process.version,
    snapshot_id: inputs.manifest.snapshot_id,
    source_manifest_sha256: inputs.manifestSha256,
    circuit_engine_sha256: inputs.engineSha256,
    circuit_bundle_sha256: inputs.bundleSha256,
    dt_s: inputs.manifest.execution_contract.dt_s,
    sample_count: inputs.manifest.execution_contract.sample_count,
    pre_roll_steps: inputs.manifest.execution_contract.pre_roll_steps,
    motor_input_policy: "four individual NOD1 voltage channels only",
    full_cell_state_eligible_motor_input: false,
    executable_asset_ids: ["circuit_engine", "circuit_bundle"],
  };
}

function strictRequest(line) {
  const request = JSON.parse(line);
  exactKeys(request, ["id", "method", "params"], "RPC request");
  requireCondition(Number.isInteger(request.id) && request.id > 0, "request id must be a positive integer");
  requireCondition(typeof request.method === "string", "request method must be a string");
  requireCondition(
    request.params !== null && typeof request.params === "object" && !Array.isArray(request.params),
    "request params must be an object",
  );
  return request;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const inputs = loadRuntimeInputs(options);
  const runtime = new Runtime(inputs);
  process.stdout.write(`${JSON.stringify(readyPayload(inputs))}\n`);
  const reader = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of reader) {
    if (line.trim() === "") continue;
    let id = null;
    try {
      const request = strictRequest(line);
      id = request.id;
      let result;
      if (request.method === "initialize") result = runtime.initialize(request.params);
      else if (request.method === "advance") result = runtime.advance(request.params);
      else if (request.method === "checkpoint") {
        exactKeys(request.params, [], "checkpoint params");
        result = runtime.checkpoint();
      } else if (request.method === "restore") result = runtime.restore(request.params);
      else if (request.method === "finalize") result = runtime.finalize(request.params);
      else if (request.method === "shutdown") {
        exactKeys(request.params, [], "shutdown params");
        process.stdout.write(`${JSON.stringify({ id, ok: true, result: { shutdown: true } })}\n`);
        return;
      } else fail(`unknown RPC method: ${request.method}`);
      process.stdout.write(`${JSON.stringify({ id, ok: true, result })}\n`);
    } catch (error) {
      process.stdout.write(`${JSON.stringify({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      })}\n`);
    }
  }
}

try {
  await main();
} catch (error) {
  process.stderr.write(
    `fly-FGS runtime failed: ${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
}
