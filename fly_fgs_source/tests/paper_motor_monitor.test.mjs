import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const Monitor = require("../lib/paper_motor_monitor.js");
const bundle = JSON.parse(fs.readFileSync(new URL("../data/bundle_fg.json", import.meta.url), "utf8"));
const laterality = JSON.parse(fs.readFileSync(new URL("../data/nod1_laterality.v1.json", import.meta.url), "utf8"));

const FG_DN = [
  { t: "DNp26", wL: 149, wN: 289 },
  { t: "DNae002", wL: 530, wN: 0 },
  { t: "DNbe001", wL: 724, wN: 2 },
  { t: "DNa04", wL: 51, wN: 0 },
  { t: "DNg32", wL: 0, wN: 41 },
  { t: "DNge107", wL: 0, wN: 10 },
  { t: "DNp07", wL: 35, wN: 0 },
];

test("individual cable inventory preserves all eight exact root IDs as strings", () => {
  const model = {
    voltage: Float64Array.from({ length: bundle.cells.length }, () => -0.06),
    act: Float64Array.from({ length: bundle.cells.length }, () => 0.25),
  };
  const cells = Monitor.extractCableCells(bundle, model, laterality);
  assert.equal(cells.length, 8);
  assert.deepEqual(cells.map((cell) => cell.root_id), [
    "720575940627502338", "720575940631147776", "720575940628438427",
    "720575940625528556", "720575940627706398", "720575940639209956",
    "720575940623997949", "720575940629456860",
  ]);
  assert.ok(cells.every((cell) => typeof cell.root_id === "string"));
  assert.equal(cells.filter((cell) => cell.motor_boundary).length, 4);
  assert.ok(cells.filter((cell) => cell.type !== "NOD1").every((cell) => cell.anatomical_side === "unresolved"));
});

test("NOD1 raw lanes resolve anatomically and cross once onto DN copies", () => {
  const lanes = Monitor.resolveNod1Lanes({ nod1L: 0.9, nod1R: 0.2 }, laterality);
  assert.deepEqual(lanes.raw_application, { left: 0.9, right: 0.2 });
  assert.deepEqual(lanes.source_anatomical, { left: 0.2, right: 0.9 });
  assert.deepEqual(lanes.contralateral_dn, { left: 0.9, right: 0.2 });
});

test("paper DN preview uses NOD1 counts only and leaves LLPC1-only paths silent", () => {
  const result = Monitor.computePaperDNDrive({ nod1L: 0.9, nod1R: 0 }, FG_DN, laterality);
  const byType = new Map(result.drives.map((drive) => [drive.type, drive]));
  assert.equal(byType.get("DNp26").left, 1.3);
  assert.equal(byType.get("DNp26").right, 0);
  assert.ok(byType.get("DNbe001").left > 0);
  assert.ok(byType.get("DNg32").left > 0);
  assert.ok(byType.get("DNge107").left > 0);
  for (const type of ["DNae002", "DNa04", "DNp07"]) {
    assert.deepEqual(
      { left: byType.get(type).left, right: byType.get(type).right },
      { left: 0, right: 0 },
    );
  }
  assert.equal(result.boundary, "four_exact_nod1_cells_only");
  assert.ok(result.steering_signal_au < 0);
});

test("retinotopic hit testing is invariant to CSS canvas scaling", () => {
  const motion = Monitor.buildRetinotopicIndex(bundle, "motion");
  const layout = {
    x: 8, y: 50, width: 280, height: 80,
    az_min_deg: -145, az_max_deg: 145, el_min_deg: -84, el_max_deg: 86,
  };
  const target = motion[Math.floor(motion.length / 2)];
  const projected = Monitor.projectRetinotopicCell(target, layout);
  const rectangle = { left: 100, top: 40, width: 592, height: 1388 };
  const clientX = rectangle.left + projected.x * rectangle.width / 296;
  const clientY = rectangle.top + projected.y * rectangle.height / 694;
  const point = Monitor.canvasPointFromClient(clientX, clientY, rectangle, 296, 694);
  const hit = Monitor.nearestRetinotopicCell(motion, point, layout, 0.01);
  assert.equal(hit.cell.index, target.index);
  assert.ok(hit.distance_px < 1e-9);
});

