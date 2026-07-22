import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const PaperFGS = require("../lib/paper_stimulus.js");
const protocols = JSON.parse(fs.readFileSync(new URL("../data/paper_protocols.json", import.meta.url), "utf8"));

function makeElement(id) {
  const values = new Set();
  const element = {
    id, value: "", textContent: "", innerHTML: "", disabled: false, dataset: {}, children: [],
    classList: { add: (value) => values.add(value), remove: (value) => values.delete(value), toggle: (value, on) => on ? values.add(value) : values.delete(value) },
    appendChild(child) { this.children.push(child); if (!this.value) this.value = child.value || ""; },
    addEventListener() {},
  };
  if (id.endsWith("Canvas")) {
    const context = {};
    for (const name of ["clearRect", "fillRect", "strokeRect", "fillText", "beginPath", "moveTo", "lineTo", "stroke", "closePath", "fill", "save", "restore", "setLineDash"]) context[name] = () => {};
    element.width = 346; element.height = 190; element.getContext = () => context;
  }
  return element;
}

test("v3 authoritative replay exposes calibrated bilateral wing mechanics", async () => {
  const ids = [
    "paperProtocol", "paperTextureMode", "paperTrial", "paperUnits", "paperStatus",
    "paperPanel", "paperModeBtn", "legacyModeBtn", "paperRestart", "paperStimulusCanvas",
    "paperPlotCanvas", "paperWingPlotCanvas", "paperTime", "paperInterval", "paperFigure", "paperGround",
    "paperPhase", "paperOutput", "paperMean", "paperHarmonic", "paperValidation",
    "paperSignalProduct", "paperAbsoluteBadge", "paperNoFitBadge", "paperSemBadge",
    "paperWholeFlyBadge", "paperWingBadge", "paperMeter", "paperWingInstant",
    "paperWingBalance", "paperWingPost", "paperWingHarmonic", "paperScores", "paperScores2",
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
  globalThis.FGModel = {
    init: () => ({}), stepPaper: () => {},
    readout: () => ({ nod1L: 0, nod1R: 0, llpcL: 0, llpcR: 0 }),
  };
  const channel = (value) => ({
    raw_trials: [value],
    products: { paper_comparison: { mean: value, sem: Math.abs(value) * 0.1 } },
  });
  const frame = {
    raw_trial_reported_yaw_torque_Nm: [4e-8], reported_yaw_torque_Nm: 4e-8,
    reported_yaw_torque_sem_Nm: 1e-9, raw_trial_wing_joint_position_rad: [[0, 0, 0, 0, 0, 0]],
    torque_products_Nm: { paper_comparison: { mean: 4e-8, sem: 1e-9 } },
    torque_channels_Nm: {
      authoritative_total: channel(4e-8), left_wing: channel(1e-8),
      right_wing: channel(2e-8), wing_sum: channel(3e-8),
      nonwing_residual: channel(1e-8), left_wing_aerodynamic: channel(0.5e-8),
      right_wing_aerodynamic: channel(1.5e-8),
    },
  };
  const wingAnalysis = {
    post_transition_mean_torque_n_m: 1e-8,
    first_harmonic_amplitude_n_m: 0.5e-8,
    first_harmonic_phase_relative_ground_position_deg: 30,
  };
  const replay = {
    schema_version: "paper_fgs_web_replay.v3", protocol_id: "R83_Fig3a_0_to_90",
    scientific_status: "authoritative_native_torque", validation: { apparatus_passed: true },
    trial_count: 1, replay_rate_hz: 200, torque_meter_mode: "comparison",
    metrology: {
      calibration_passed: true, sampling_passed: true, convergence_passed: true,
      dual_meter_passed: true, paper_extraction_passed: true,
      bilateral_wing_calibration_passed: true, per_wing_metrology_passed: true,
      aerodynamic_probe_passed: true,
    },
    channel_authority: { wing_reported_yaw_torque_Nm: "calibrated_simulation_decomposition" },
    frames: Array.from({ length: 500 }, () => frame),
    analysis: {
      first_harmonic_amplitude_n_m: 1e-8,
      first_harmonic_phase_relative_ground_position_deg: 10,
      mechanical_channels: { left_wing: wingAnalysis, right_wing: wingAnalysis },
    },
  };
  const replayText = JSON.stringify(replay);
  const replaySha = crypto.createHash("sha256").update(replayText, "utf8").digest("hex");
  const manifest = {
    run_id: "test-v3-authoritative-run", scientific_status: "authoritative_native_torque",
    files: { "web_replay.json": { sha256: replaySha } },
  };
  globalThis.fetch = async (url) => {
    if (url === "data/paper_protocols.json") return { ok: true, json: async () => protocols };
    if (url === "data/paper_replay.json") return { ok: true, text: async () => replayText };
    if (url === "data/paper_replay.json.manifest.json") return { ok: true, json: async () => manifest };
    return { ok: false };
  };
  await import(`../lib/paper_app.js?v3-monitor-test=${Date.now()}`);
  await globalThis.PaperUI.start({ bundle: {}, scene: null });
  globalThis.PaperUI.tick(0.01);
  assert.match(elements.get("paperWingInstant").textContent, /dyne·cm/);
  assert.match(elements.get("paperWingBalance").textContent, /dyne·cm/);
  assert.equal(elements.get("paperWingBadge").textContent, "calibrated wing-root decomposition");
});
