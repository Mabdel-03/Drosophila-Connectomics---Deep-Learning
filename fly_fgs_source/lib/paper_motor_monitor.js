(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PaperMotorMonitor = api;
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  const DEFAULTS = Object.freeze({
    nod1_full_scale: 0.9,
    dn_gain: 2.5,
    dn_weight_max: 300,
    maximum_dn_drive: 1.3,
  });

  function finite(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function clip(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function normalizeRootId(value) {
    const text = String(value == null ? "" : value);
    return /^r\d+$/.test(text) ? text.slice(1) : text;
  }

  function lateralityByRoot(receipt) {
    if (!receipt || receipt.schema_version !== "1.0.0" || !Array.isArray(receipt.cells)) {
      throw new TypeError("a versioned NOD1 laterality receipt is required");
    }
    if (receipt.paper_effector_assignment !== "raw_l_to_physical_right") {
      throw new Error("unexpected paper NOD1 effector assignment");
    }
    const result = new Map();
    for (const cell of receipt.cells) {
      const rootId = normalizeRootId(cell.root_id);
      const rawSide = String(cell.raw_application_side || "").toUpperCase();
      const anatomicalSide = String(cell.resolved_anatomical_side || "").toLowerCase();
      if (!/^\d+$/.test(rootId) || !["L", "R"].includes(rawSide)
          || !["left", "right"].includes(anatomicalSide)) {
        throw new Error("invalid NOD1 laterality cell");
      }
      result.set(rootId, Object.freeze({ rawSide, anatomicalSide }));
    }
    if (result.size !== 4) throw new Error("NOD1 laterality receipt must contain four cells");
    const pairs = Array.from(result.values()).map((item) => `${item.rawSide}:${item.anatomicalSide}`);
    if (pairs.filter((value) => value === "L:right").length !== 2
        || pairs.filter((value) => value === "R:left").length !== 2) {
      throw new Error("NOD1 laterality receipt does not resolve the expected lane swap");
    }
    return result;
  }

  function resolveNod1Lanes(readout, receipt) {
    lateralityByRoot(receipt);
    const rawLeft = Math.max(0, finite(readout && readout.nod1L, 0));
    const rawRight = Math.max(0, finite(readout && readout.nod1R, 0));
    const sourceAnatomical = Object.freeze({
      left: rawRight,
      right: rawLeft,
    });
    return Object.freeze({
      raw_application: Object.freeze({ left: rawLeft, right: rawRight }),
      source_anatomical: sourceAnatomical,
      // The paper bridge uses contralateral source-to-DN lanes. Anatomical-right
      // NOD1 therefore drives the left DN copy and vice versa.
      contralateral_dn: Object.freeze({
        left: sourceAnatomical.right,
        right: sourceAnatomical.left,
      }),
    });
  }

  function computePaperDNDrive(readout, definitions, receipt, options) {
    const config = Object.assign({}, DEFAULTS, options || {});
    for (const key of Object.keys(DEFAULTS)) {
      if (!(finite(config[key], 0) > 0)) throw new RangeError(`${key} must be positive`);
    }
    if (!Array.isArray(definitions)) throw new TypeError("descending-neuron definitions must be an array");
    const lanes = resolveNod1Lanes(readout, receipt);
    const sourceLeft = clip(lanes.contralateral_dn.left / config.nod1_full_scale, 0, 1);
    const sourceRight = clip(lanes.contralateral_dn.right / config.nod1_full_scale, 0, 1);
    const drives = definitions.map((definition) => {
      const synapses = Math.max(0, finite(definition && definition.wN, 0));
      const weight = synapses / config.dn_weight_max;
      return Object.freeze({
        type: String(definition && definition.t || ""),
        nod1_synapses: synapses,
        left: clip(sourceLeft * weight * config.dn_gain, 0, config.maximum_dn_drive),
        right: clip(sourceRight * weight * config.dn_gain, 0, config.maximum_dn_drive),
      });
    });
    const active = drives.filter((drive) => drive.nod1_synapses > 0);
    const signed = active.reduce((sum, drive) => sum + drive.right - drive.left, 0)
      / Math.max(1, active.length);
    return Object.freeze({
      lanes,
      drives: Object.freeze(drives),
      steering_signal_au: signed,
      boundary: "four_exact_nod1_cells_only",
    });
  }

  function extractCableCells(bundle, model, receipt) {
    if (!bundle || !Array.isArray(bundle.cells) || !Array.isArray(bundle.ids)) {
      throw new TypeError("the fly-FGS bundle is required");
    }
    const laterality = receipt ? lateralityByRoot(receipt) : new Map();
    const voltage = model && model.voltage;
    const activity = model && model.act;
    const output = [];
    bundle.cells.forEach((cell, index) => {
      const type = String(cell.type || "").toUpperCase();
      if (!(type.includes("VCH") || type.includes("DCH") || type.includes("NOD1"))) return;
      const rootId = normalizeRootId(bundle.ids[index] || cell.id);
      const resolved = laterality.get(rootId);
      output.push(Object.freeze({
        index,
        root_id: rootId,
        type: type.includes("NOD1") ? "NOD1" : (type.includes("VCH") ? "vCH" : "DCH"),
        raw_application_side: String(cell.side || "?").toUpperCase(),
        anatomical_side: resolved ? resolved.anatomicalSide : "unresolved",
        voltage_v: voltage ? finite(voltage[index], NaN) : NaN,
        voltage_mv: voltage ? finite(voltage[index], NaN) * 1000 : NaN,
        activation: activity ? finite(activity[index], NaN) : NaN,
        motor_boundary: Boolean(resolved && type.includes("NOD1")),
      }));
    });
    return Object.freeze(output);
  }

  function buildRetinotopicIndex(bundle, group) {
    if (!bundle || !Array.isArray(bundle.cells) || !Array.isArray(bundle.ids)) {
      throw new TypeError("the fly-FGS bundle is required");
    }
    const wanted = String(group || "").toLowerCase();
    const output = [];
    bundle.cells.forEach((cell, index) => {
      const type = String(cell.type || "").toUpperCase();
      const matches = wanted === "motion"
        ? (type.includes("T4A") || type.includes("T5A"))
        : wanted === "llpc1" && type.includes("LLPC1");
      if (!matches || cell.retinoAzimuth == null || cell.retinoElevation == null) return;
      output.push(Object.freeze({
        index,
        root_id: normalizeRootId(bundle.ids[index] || cell.id),
        type: type.includes("T4A") ? "T4a" : (type.includes("T5A") ? "T5a" : "LLPC1"),
        raw_application_side: String(cell.side || "?").toUpperCase(),
        azimuth_deg: finite(cell.retinoAzimuth, 0),
        elevation_deg: finite(cell.retinoElevation, 0),
      }));
    });
    return Object.freeze(output);
  }

  function projectRetinotopicCell(cell, layout) {
    const azSpan = finite(layout.az_max_deg, 0) - finite(layout.az_min_deg, 0);
    const elSpan = finite(layout.el_max_deg, 0) - finite(layout.el_min_deg, 0);
    if (!(azSpan > 0) || !(elSpan > 0)) throw new RangeError("invalid retinotopic layout");
    return Object.freeze({
      x: finite(layout.x, 0) + (cell.azimuth_deg - layout.az_min_deg) / azSpan * layout.width,
      y: finite(layout.y, 0) + layout.height
        - (cell.elevation_deg - layout.el_min_deg) / elSpan * layout.height,
    });
  }

  function nearestRetinotopicCell(cells, point, layout, radiusPx) {
    const maximum = Math.max(0, finite(radiusPx, 7));
    let best = null;
    for (const cell of cells || []) {
      const projected = projectRetinotopicCell(cell, layout);
      const distance = Math.hypot(projected.x - point.x, projected.y - point.y);
      if (distance <= maximum && (!best || distance < best.distance_px)) {
        best = Object.freeze({ cell, distance_px: distance, point: projected });
      }
    }
    return best;
  }

  function canvasPointFromClient(clientX, clientY, rectangle, canvasWidth, canvasHeight) {
    if (!(rectangle && rectangle.width > 0 && rectangle.height > 0)) {
      throw new RangeError("canvas client rectangle must have positive dimensions");
    }
    return Object.freeze({
      x: (finite(clientX, 0) - rectangle.left) * canvasWidth / rectangle.width,
      y: (finite(clientY, 0) - rectangle.top) * canvasHeight / rectangle.height,
    });
  }

  return Object.freeze({
    DEFAULTS,
    normalizeRootId,
    lateralityByRoot,
    resolveNod1Lanes,
    computePaperDNDrive,
    extractCableCells,
    buildRetinotopicIndex,
    projectRetinotopicCell,
    nearestRetinotopicCell,
    canvasPointFromClient,
  });
});
