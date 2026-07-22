import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const PaperFGS = require("../lib/paper_stimulus.js");
const protocols = JSON.parse(fs.readFileSync(new URL("../data/paper_protocols.json", import.meta.url), "utf8"));

function classList() {
  const values = new Set();
  return { add: (value) => values.add(value), remove: (value) => values.delete(value), toggle: (value, on) => on ? values.add(value) : values.delete(value) };
}

function makeElement(id) {
  const element = {
    id, value: "", textContent: "", innerHTML: "", disabled: false, dataset: {},
    classList: classList(), children: [],
    appendChild(child) { this.children.push(child); if (!this.value) this.value = child.value || ""; },
    addEventListener() {},
  };
  if (id.endsWith("Canvas")) {
    const context = {};
    for (const name of ["clearRect", "fillRect", "strokeRect", "fillText", "beginPath", "moveTo", "lineTo", "stroke", "closePath", "fill", "save", "restore", "setLineDash"]) context[name] = () => {};
    element.width = 346; element.height = 178; element.getContext = () => context;
  }
  return element;
}

test("authoritative replay still advances and publishes the parallel paper circuit", async () => {
  const ids = [
    "paperProtocol", "paperTextureMode", "paperTrial", "paperUnits", "paperStatus",
    "paperPanel", "paperModeBtn", "legacyModeBtn", "paperRestart", "paperStimulusCanvas",
    "paperPlotCanvas", "paperWingPlotCanvas", "paperTime", "paperInterval", "paperFigure", "paperGround",
    "paperPhase", "paperOutput", "paperMean", "paperHarmonic", "paperValidation",
    "paperSignalProduct", "paperAbsoluteBadge", "paperNoFitBadge", "paperSemBadge",
    "paperWholeFlyBadge", "paperWingBadge", "paperMeter", "paperWingInstant",
    "paperWingBalance", "paperWingPost", "paperWingHarmonic", "paperScores",
    "paperScores2",
  ];
  const elements = new Map(ids.map((id) => [id, makeElement(id)]));
  elements.get("paperProtocol").value = "R83_Fig3a_0_to_90";
  elements.get("paperTextureMode").value = "registered_copy";
  elements.get("paperUnits").value = "dyne_cm";
  elements.get("paperTrial").value = "0";
  elements.get("paperSignalProduct").value = "paper_comparison";
  globalThis.document = {
    getElementById: (id) => elements.get(id),
    createElement: (tag) => makeElement(tag),
  };
  globalThis.PaperFGS = PaperFGS;
  let modelSteps = 0;
  const model = { marker: "same-model-reference" };
  globalThis.FGModel = {
    init: () => model,
    stepPaper: () => { modelSteps += 1; },
    readout: () => ({ nod1L: 0.1, nod1R: 0.2, llpcL: 0.03, llpcR: 0.04 }),
  };
  const frame = {
    raw_trial_reported_yaw_torque_Nm: [1e-8], reported_yaw_torque_Nm: 2e-8,
    reported_yaw_torque_sem_Nm: 1e-9, raw_trial_wing_joint_position_rad: [[0, 0, 0, 0, 0, 0]],
  };
  const replay = {
    schema_version: "paper_fgs_web_replay.v2",
    protocol_id: "R83_Fig3a_0_to_90", scientific_status: "authoritative_native_torque",
    validation: { apparatus_passed: true }, trial_count: 1, replay_rate_hz: 200,
    metrology: { calibration_passed: true, sampling_passed: true, convergence_passed: true, dual_meter_passed: true, paper_extraction_passed: true },
    torque_meter_mode: "comparison",
    frames: Array.from({ length: 500 }, () => frame), analysis: {},
  };
  const replayText = JSON.stringify(replay);
  const replaySha = crypto.createHash("sha256").update(replayText, "utf8").digest("hex");
  const manifest = {
    run_id: "test-authoritative-run", scientific_status: "authoritative_native_torque",
    files: { "web_replay.json": { sha256: replaySha } },
  };
  globalThis.fetch = async (url) => {
    if (url === "data/paper_protocols.json") return { ok: true, json: async () => protocols };
    if (url === "data/paper_replay.json") return { ok: true, text: async () => replayText };
    if (url === "data/paper_replay.json.manifest.json") return { ok: true, json: async () => manifest };
    return { ok: false };
  };
  await import(`../lib/paper_app.js?monitor-test=${Date.now()}`);
  const samples = [];
  await globalThis.PaperUI.start({
    bundle: {}, scene: null,
    onCircuitSample: (sample) => samples.push(sample),
  });
  assert.equal(globalThis.PaperUI.currentMode(), "paper-replay");
  globalThis.PaperUI.tick(0.01);
  assert.equal(modelSteps, 4);
  assert.equal(samples.length, 5); // reset snapshot plus four 400 Hz samples
  assert.equal(samples.at(-1).model, model);
  assert.equal(samples.at(-1).mode, "paper-replay");
  assert.equal(samples.at(-1).authoritativeReplay, true);
  assert.equal(elements.get("paperOutput").textContent.endsWith("dyne·cm"), true);
  assert.equal(elements.get("paperWingInstant").textContent, "unavailable");
  assert.equal(elements.get("paperWingBadge").textContent, "wing torque unavailable");
});
