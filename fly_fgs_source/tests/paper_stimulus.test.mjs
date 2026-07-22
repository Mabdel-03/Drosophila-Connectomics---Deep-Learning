import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const PaperFGS = require("../lib/paper_stimulus.js");
const protocols = JSON.parse(
  fs.readFileSync(new URL("../data/paper_protocols.json", import.meta.url), "utf8"),
);

test("paper coordinate convention points positive azimuth right", () => {
  const direction = PaperFGS.paperRayDirection(30);
  assert.ok(Math.abs(direction[0] - Math.cos(Math.PI / 6)) < 1e-12);
  assert.ok(Math.abs(direction[1] + 0.5) < 1e-12);
  assert.equal(direction[2], 0);
});

test("texture is deterministic, balanced, and wraps in 120 columns", () => {
  const first = PaperFGS.createBinaryTexture({ seed: 123456 });
  const second = PaperFGS.createBinaryTexture({ seed: 123456 });
  assert.equal(first.columns, 120);
  assert.deepEqual(first.values, second.values);
  assert.equal(first.values.reduce((sum, value) => sum + value, 0), first.values.length / 2);
  assert.equal(PaperFGS.textureCell(first, -1, 0), PaperFGS.textureCell(first, 359, 0));
  assert.equal(first.pixel_width_deg, 3);
  assert.equal(first.pixel_height_deg, 3);
});

test("paper luminance levels preserve mean and normalized black-white ratio", () => {
  const config = PaperFGS.DEFAULT_TEXTURE;
  assert.equal((config.black_luminance_cd_m2 + config.white_luminance_cd_m2) / 2, 700);
  assert.ok(Math.abs(
    config.black_luminance_cd_m2 / config.white_luminance_cd_m2 - 0.12359550561797752,
  ) < 1e-15);
  const scene = PaperFGS.createScene();
  const state = { figure_angle_deg: 0, ground_angle_deg: 0 };
  const blackCell = scene.ground_texture.values.findIndex((value) => value === 0);
  const whiteCell = scene.ground_texture.values.findIndex((value) => value === 1);
  for (const [index, expected] of [[blackCell, 154], [whiteCell, 1246]]) {
    const row = Math.floor(index / scene.ground_texture.columns);
    const column = index % scene.ground_texture.columns;
    const azimuth = column * 3 + 1.5;
    const elevation = -90 + row * 3 + 1.5;
    assert.equal(PaperFGS.luminanceAt(scene, azimuth, elevation, state).luminance_cd_m2, expected);
  }
});

test("registered-copy scene removes the synchronous figure boundary", () => {
  const scene = PaperFGS.createScene({ relationship_mode: "registered_copy" });
  const state = { figure_angle_deg: 5, ground_angle_deg: 5 };
  assert.equal(PaperFGS.luminanceAt(scene, 28.999, 0, state).layer, "ground");
  assert.equal(PaperFGS.luminanceAt(scene, 29, 0, state).layer, "figure");
  assert.equal(PaperFGS.luminanceAt(scene, 40.999, 0, state).layer, "figure");
  assert.equal(PaperFGS.luminanceAt(scene, 41, 0, state).layer, "ground");
  for (const azimuth of [23.999, 24, 25.5, 35.999, 36, 37.5]) {
    const actual = PaperFGS.luminanceAt(scene, azimuth + 5, 0, state);
    const expectedBit = PaperFGS.textureCell(scene.ground_texture, azimuth, 0);
    assert.equal(actual.binary_value, expectedBit);
  }
});

test("Figure 3 motion schedules are analytic and signed", () => {
  const fig3a = PaperFGS.resolveProtocol(protocols, "R83_Fig3a_0_to_90");
  assert.equal(fig3a.frequency_hz, 2.5);
  assert.ok(Math.abs(PaperFGS.sampleProtocol(fig3a, 0.1).ground_angle_realized_deg - 5) < 1e-12);
  assert.equal(PaperFGS.sampleProtocol(fig3a, 0.4).relative_phase_realized_deg, 0);
  assert.ok(Math.abs(PaperFGS.sampleProtocol(fig3a, 0.6).relative_phase_realized_deg - 45) < 1e-12);
  assert.equal(PaperFGS.sampleProtocol(fig3a, 0.8).relative_phase_realized_deg, 90);

  const fig3b = PaperFGS.resolveProtocol(protocols, "R83_Fig3b_0_to_270");
  assert.equal(PaperFGS.sampleProtocol(fig3b, 0.8).relative_phase_realized_deg, -90);

  const fig3c = PaperFGS.resolveProtocol(protocols, "R83_Fig3c_0_to_180");
  assert.equal(fig3c.trial_duration_s, 4);
  assert.equal(PaperFGS.sampleProtocol(fig3c, 1.6).relative_phase_realized_deg, 180);

  const omega = 2 * Math.PI * 2.5;
  assert.ok(Math.abs(PaperFGS.sampleProtocol(fig3a, 0).ground_velocity_command_deg_s - 5 * omega) < 1e-12);
  assert.ok(Math.abs(PaperFGS.sampleProtocol(fig3a, 0.1).ground_angle_realized_deg - 5) < 1e-12);
  assert.ok(Math.abs(5 * omega * omega - 1233.7005501361698) < 1e-10);
  assert.ok(Math.abs(PaperFGS.sampleProtocol(fig3c, 0.1).ground_angle_realized_deg - 7.5) < 1e-12);
});

test("torque units round trip exactly at the paper calibration point", () => {
  assert.equal(PaperFGS.torqueNmToDyneCm(1e-7), 1);
  assert.equal(PaperFGS.torqueDyneCmToNm(1), 1e-7);
  assert.equal(PaperFGS.torqueNmToDyneCm(-1e-7), -1);
});

test("portable replay checksum matches SHA-256", () => {
  const payload = "paper replay checksum π";
  const expected = crypto.createHash("sha256").update(payload, "utf8").digest("hex");
  assert.equal(PaperFGS.sha256Utf8(payload), expected);
});
