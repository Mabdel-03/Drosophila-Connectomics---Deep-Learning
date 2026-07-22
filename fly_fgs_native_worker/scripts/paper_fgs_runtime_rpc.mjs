#!/usr/bin/env node
/** Fixed-step paper-stimulus boundary for the captured fly-FGS circuit. */

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_SOURCE_ROOT = path.resolve(SCRIPT_DIR, "../../fly_fgs_source");
const PROTOCOL_VERSION = "1.0.0";
const DEFAULT_DT_S = 0.0025;

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function exactKeys(value, expected, label) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  requireCondition(JSON.stringify(actual) === JSON.stringify(wanted), `${label} fields do not match protocol`);
}

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function finite(value, label) {
  requireCondition(typeof value === "number" && Number.isFinite(value), `${label} must be finite`);
  return value;
}

function parseArguments(argv) {
  let sourceRoot = DEFAULT_SOURCE_ROOT;
  let dtS = DEFAULT_DT_S;
  let expectedManifest = null;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--source-root") {
      requireCondition(index + 1 < argv.length, "--source-root requires a path");
      sourceRoot = path.resolve(argv[++index]);
    } else if (argv[index] === "--dt-s") {
      requireCondition(index + 1 < argv.length, "--dt-s requires a value");
      dtS = Number(argv[++index]);
      requireCondition(dtS === 0.0025 || dtS === 0.00125, "--dt-s must select 400 or 800 Hz");
    } else if (argv[index] === "--expected-manifest") {
      requireCondition(index + 1 < argv.length, "--expected-manifest requires a path");
      expectedManifest = path.resolve(argv[++index]);
    } else if (argv[index] === "--help") {
      process.stdout.write("Usage: paper_fgs_runtime_rpc.mjs [--source-root PATH] [--dt-s 0.0025|0.00125]\n");
      process.exit(0);
    } else {
      throw new Error(`unknown argument: ${argv[index]}`);
    }
  }
  return { sourceRoot, dtS, expectedManifest };
}

function verifyExpectedManifest(inputs, manifestPath) {
  if (manifestPath === null) return null;
  const bytes = fs.readFileSync(manifestPath);
  const manifest = JSON.parse(bytes.toString("utf8"));
  requireCondition(manifest.schema_version === "1.0.0", "paper runtime manifest schema mismatch");
  for (const [name, receipt] of Object.entries(inputs.receipts)) {
    const expected = manifest.assets && manifest.assets[name];
    requireCondition(
      expected && expected.bytes === receipt.bytes && expected.sha256 === receipt.sha256,
      `${name} does not match paper runtime manifest`,
    );
  }
  requireCondition(
    Object.keys(manifest.assets).length === Object.keys(inputs.receipts).length,
    "paper runtime manifest asset inventory mismatch",
  );
  return {
    path: manifestPath,
    bytes: bytes.length,
    sha256: sha256(bytes),
    runtime_id: manifest.runtime_id,
  };
}

function regularFile(root, relative) {
  const resolved = path.resolve(root, relative);
  const relation = path.relative(root, resolved);
  requireCondition(relation && !relation.startsWith("..") && !path.isAbsolute(relation), "asset escapes source root");
  const stat = fs.lstatSync(resolved);
  requireCondition(stat.isFile() && !stat.isSymbolicLink(), `${relative} must be a regular file`);
  return resolved;
}

function loadInputs(sourceRoot) {
  const assets = {
    paper_stimulus: regularFile(sourceRoot, "lib/paper_stimulus.js"),
    circuit_engine: regularFile(sourceRoot, "lib/fgmodel.js"),
    circuit_bundle: regularFile(sourceRoot, "data/bundle_fg.json"),
    protocol_document: regularFile(sourceRoot, "data/paper_protocols.json"),
  };
  const receipts = {};
  for (const [name, assetPath] of Object.entries(assets)) {
    const bytes = fs.readFileSync(assetPath);
    receipts[name] = { path: assetPath, bytes: bytes.length, sha256: sha256(bytes) };
  }
  const require = createRequire(import.meta.url);
  delete require.cache[assets.paper_stimulus];
  const paper = require(assets.paper_stimulus);
  globalThis.PaperFGS = paper;
  delete require.cache[assets.circuit_engine];
  const engine = require(assets.circuit_engine);
  requireCondition(typeof engine.stepPaper === "function", "paper circuit engine API is unavailable");
  const bundle = JSON.parse(fs.readFileSync(assets.circuit_bundle, "utf8"));
  const protocols = JSON.parse(fs.readFileSync(assets.protocol_document, "utf8"));
  requireCondition(bundle.ids.length === 1684, "paper runtime circuit inventory mismatch");
  const nod1 = bundle.cells
    .map((cell, index) => ({ cell, index }))
    .filter(({ cell }) => cell.type === "Nod1");
  const retinotopic = bundle.cells
    .map((cell, index) => ({ cell, index }))
    .filter(({ cell }) => cell.type === "T4a" && cell.retinotopic === true
      && Number.isFinite(cell.retinoAzimuth) && Number.isFinite(cell.retinoElevation));
  requireCondition(nod1.length === 4 && retinotopic.length === 1441, "paper runtime readout inventory mismatch");
  return { sourceRoot, receipts, paper, engine, bundle, protocols, nod1, retinotopic };
}

class Runtime {
  constructor(inputs, dtS) {
    this.inputs = inputs;
    this.dtS = dtS;
    this.state = null;
    this.protocol = null;
    this.scene = null;
    this.sampleIndex = -1;
    this.stimulus = null;
  }

  configure(params) {
    exactKeys(params, ["protocol_id", "texture_relationship_mode", "texture_seed", "stimulus_time_s", "sample_options"], "initialize params");
    requireCondition(["registered_copy", "independent_matched_statistics"].includes(params.texture_relationship_mode), "unsupported texture mode");
    requireCondition(Number.isInteger(params.texture_seed) && params.texture_seed >= 0 && params.texture_seed <= 0xffffffff, "texture_seed must be uint32");
    this.protocol = this.inputs.paper.resolveProtocol(this.inputs.protocols, params.protocol_id);
    this.scene = this.inputs.paper.createScene({
      relationship_mode: params.texture_relationship_mode,
      master_seed: params.texture_seed,
      ground_seed: params.texture_seed,
      figure_seed: params.texture_seed === 123456
        ? 654321
        : (params.texture_seed ^ 0x9e3779b9) >>> 0,
    });
    this.state = this.inputs.engine.init(this.inputs.bundle, {
      dt: this.dtS,
      figW: 12,
      figH: 180,
      startAz: 30,
      paperMode: true,
      paperScene: this.scene,
      inputScale: 3,
      scale: 3,
      llpcBoost: 1,
    });
    this.sampleIndex = 0;
    this.applyStimulus(finite(params.stimulus_time_s, "stimulus_time_s"));
    return this.sample(params.sample_options);
  }

  applyStimulus(stimulusTimeS) {
    this.stimulus = this.inputs.paper.sampleProtocol(this.protocol, stimulusTimeS);
    this.inputs.engine.stepPaper(this.state, {
      figureAngleDeg: this.stimulus.figure_angle_realized_deg,
      groundAngleDeg: this.stimulus.ground_angle_realized_deg,
      figureVelocityDegS: this.stimulus.figure_velocity_command_deg_s,
      groundVelocityDegS: this.stimulus.ground_velocity_command_deg_s,
    });
  }

  advance(params) {
    exactKeys(params, ["sample_index", "stimulus_time_s", "sample_options"], "advance params");
    requireCondition(Number.isInteger(params.sample_index) && params.sample_index === this.sampleIndex + 1, "paper sample index must advance by one");
    this.sampleIndex = params.sample_index;
    this.applyStimulus(finite(params.stimulus_time_s, "stimulus_time_s"));
    return this.sample(params.sample_options);
  }

  sample(options) {
    exactKeys(options, ["include_full_cell_state", "include_retinal_input"], "sample options");
    const readout = this.inputs.engine.readout(this.state);
    const nod1 = {};
    for (const { cell, index } of this.inputs.nod1) {
      nod1[String(cell.id).replace(/^r/, "")] = this.state.voltage[index];
    }
    const result = {
      sample_index: this.sampleIndex,
      measurement_time_s: this.sampleIndex * this.dtS,
      availability_time_s: this.sampleIndex * this.dtS,
      stimulus_time_s: this.stimulus.time_s,
      stimulus: this.stimulus,
      nod1_voltage_v: nod1,
      pooled_readout: readout,
      full_cell_state_eligible_motor_input: false,
    };
    if (options.include_retinal_input) {
      result.retinal_input_luminance = this.inputs.retinotopic.map(({ cell }) => (
        this.inputs.engine.lumNow(this.state, cell.retinoAzimuth, cell.retinoElevation)
      ));
    }
    if (options.include_full_cell_state) {
      result.full_cell_voltage_v = Array.from(this.state.voltage);
      result.full_cell_activity = Array.from(this.state.act);
    }
    return result;
  }
}

const { sourceRoot, dtS, expectedManifest } = parseArguments(process.argv.slice(2));
const inputs = loadInputs(sourceRoot);
const runtimeManifest = verifyExpectedManifest(inputs, expectedManifest);
const runtime = new Runtime(inputs, dtS);
process.stdout.write(JSON.stringify({
  event: "ready",
  protocol_version: PROTOCOL_VERSION,
  node_version: process.version,
  dt_s: dtS,
  cell_count: 1684,
  retinotopic_t4a_input_count: 1441,
  nod1_root_ids: inputs.nod1.map(({ cell }) => String(cell.id).replace(/^r/, "")),
  assets: inputs.receipts,
  runtime_manifest: runtimeManifest,
  source_root: sourceRoot,
  dataset: {
    circuit: "FlyWire FAFB v783",
    motor_boundary: "four NOD1 voltages only",
    raw_side_labels_are_anatomical: false,
  },
}) + "\n");

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on("line", (line) => {
  let request;
  try {
    request = JSON.parse(line);
    exactKeys(request, ["id", "method", "params"], "request");
    let result;
    if (request.method === "initialize") result = runtime.configure(request.params);
    else if (request.method === "advance") result = runtime.advance(request.params);
    else if (request.method === "close") {
      process.stdout.write(JSON.stringify({ id: request.id, result: { closed: true } }) + "\n");
      process.exit(0);
    } else throw new Error("unknown method");
    process.stdout.write(JSON.stringify({ id: request.id, result }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({
      id: request && request.id,
      error: { name: error.name, message: error.message },
    }) + "\n");
  }
});
