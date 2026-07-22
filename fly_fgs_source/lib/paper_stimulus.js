(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.PaperFGS = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const SCHEMA_VERSION = "1.0.0";
  const GENERATOR_NAME = "xorshift32-balanced-binary-v1";
  const DEG2RAD = Math.PI / 180;
  const DEFAULT_TEXTURE = Object.freeze({
    pixel_width_deg: 3,
    pixel_height_deg: 3,
    horizontal_columns: 120,
    vertical_rows: 60,
    vertical_min_deg: -90,
    vertical_max_deg: 90,
    grid_origin_azimuth_deg: 0,
    grid_origin_elevation_deg: 0,
    black_probability: 0.5,
    relationship_mode: "registered_copy",
    master_seed: 123456,
    ground_seed: 123456,
    figure_seed: null,
    mean_luminance_cd_m2: 700,
    black_luminance_cd_m2: 154,
    white_luminance_cd_m2: 1246,
  });

  function requireFinite(value, name) {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new TypeError(name + " must be finite");
    return number;
  }

  function wrap360(value) {
    const wrapped = requireFinite(value, "angle") % 360;
    return wrapped < 0 ? wrapped + 360 : wrapped;
  }

  function wrap180(value) {
    const wrapped = wrap360(value + 180) - 180;
    return wrapped === 180 ? -180 : wrapped;
  }

  function xorshift32(seed) {
    let state = (Number(seed) >>> 0) || 0x6d2b79f5;
    return function () {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return (state >>> 0) / 4294967296;
    };
  }

  function createBinaryTexture(options) {
    const config = Object.assign({}, DEFAULT_TEXTURE, options || {});
    const columns = Number(config.horizontal_columns);
    const rows = Number(config.vertical_rows);
    if (!Number.isInteger(columns) || columns <= 0 || columns * config.pixel_width_deg !== 360) {
      throw new RangeError("horizontal texture must tile 360 degrees exactly");
    }
    if (!Number.isInteger(rows) || rows <= 0) throw new RangeError("vertical_rows must be positive");
    if (config.black_probability !== 0.5) {
      throw new RangeError("paper texture v1 currently requires black_probability=0.5");
    }
    const count = columns * rows;
    const values = new Uint8Array(count);
    values.fill(1, count / 2);
    const random = xorshift32(config.seed == null ? config.master_seed : config.seed);
    for (let index = count - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      const held = values[index];
      values[index] = values[swap];
      values[swap] = held;
    }
    return Object.freeze({
      schema_version: SCHEMA_VERSION,
      generator_name: GENERATOR_NAME,
      seed: Number(config.seed == null ? config.master_seed : config.seed) >>> 0,
      columns,
      rows,
      pixel_width_deg: config.pixel_width_deg,
      pixel_height_deg: config.pixel_height_deg,
      grid_origin_azimuth_deg: config.grid_origin_azimuth_deg,
      grid_origin_elevation_deg: config.grid_origin_elevation_deg,
      vertical_min_deg: config.vertical_min_deg,
      vertical_max_deg: config.vertical_max_deg,
      black_probability: config.black_probability,
      values,
    });
  }

  function textureCell(texture, azimuthDeg, elevationDeg) {
    const column = Math.floor(
      wrap360(azimuthDeg - texture.grid_origin_azimuth_deg) / texture.pixel_width_deg,
    ) % texture.columns;
    const rowKey = Math.floor(
      (elevationDeg - texture.grid_origin_elevation_deg) / texture.pixel_height_deg,
    );
    const firstRowKey = Math.floor(
      (texture.vertical_min_deg - texture.grid_origin_elevation_deg) / texture.pixel_height_deg,
    );
    const row = rowKey - firstRowKey;
    if (row < 0 || row >= texture.rows) return null;
    return texture.values[row * texture.columns + column];
  }

  function createScene(options) {
    const config = Object.assign({}, DEFAULT_TEXTURE, options || {});
    if (!["registered_copy", "independent_matched_statistics"].includes(config.relationship_mode)) {
      throw new RangeError("unsupported texture relationship_mode");
    }
    const ground = createBinaryTexture(Object.assign({}, config, { seed: config.ground_seed }));
    const figure = config.relationship_mode === "registered_copy"
      ? ground
      : createBinaryTexture(Object.assign({}, config, {
        seed: config.figure_seed == null ? 654321 : config.figure_seed,
      }));
    return Object.freeze({
      schema_version: SCHEMA_VERSION,
      config: Object.freeze(config),
      ground_texture: ground,
      figure_texture: figure,
    });
  }

  function luminanceAt(scene, azimuthDeg, elevationDeg, state) {
    const config = scene.config;
    const figureAngle = requireFinite(state.figure_angle_deg, "figure_angle_deg");
    const groundAngle = requireFinite(state.ground_angle_deg, "ground_angle_deg");
    const figureMean = state.figure_mean_azimuth_deg == null ? 30 : state.figure_mean_azimuth_deg;
    const figureWidth = state.figure_width_deg == null ? 12 : state.figure_width_deg;
    const figureLocal = wrap180(azimuthDeg - (figureMean + figureAngle));
    const inFigure = figureLocal >= -figureWidth / 2 && figureLocal < figureWidth / 2
      && elevationDeg >= config.vertical_min_deg && elevationDeg < config.vertical_max_deg;
    const texture = inFigure ? scene.figure_texture : scene.ground_texture;
    const textureAzimuth = azimuthDeg - (inFigure ? figureAngle : groundAngle);
    const bit = textureCell(texture, textureAzimuth, elevationDeg);
    const luminance = bit === 1
      ? config.white_luminance_cd_m2
      : config.black_luminance_cd_m2;
    return Object.freeze({
      luminance_cd_m2: luminance,
      normalized_luminance: luminance / config.white_luminance_cd_m2,
      binary_value: bit,
      layer: inFigure ? "figure" : "ground",
    });
  }

  function phaseSchedule(protocol, timeS) {
    const time = requireFinite(timeS, "time_s");
    const start = protocol.phase_transition_start_s;
    const duration = protocol.phase_transition_duration_s;
    const delta = protocol.signed_phase_delta_deg;
    if (time < start) return { phase_deg: 0, phase_velocity_deg_s: 0, interval: "synchronous" };
    if (time < start + duration) {
      const fraction = (time - start) / duration;
      return {
        phase_deg: delta * fraction,
        phase_velocity_deg_s: delta / duration,
        interval: "phase_transition",
      };
    }
    return { phase_deg: delta, phase_velocity_deg_s: 0, interval: "relative_motion" };
  }

  function sampleProtocol(protocol, timeS) {
    const time = requireFinite(timeS, "time_s");
    const phase = phaseSchedule(protocol, time);
    const omega = 2 * Math.PI * protocol.frequency_hz;
    const groundTheta = omega * time;
    const figureTheta = groundTheta + phase.phase_deg * DEG2RAD;
    const phaseVelocity = phase.phase_velocity_deg_s * DEG2RAD;
    const groundAngle = protocol.ground_amplitude_deg * Math.sin(groundTheta);
    const figureAngle = protocol.figure_amplitude_deg * Math.sin(figureTheta);
    const groundVelocity = protocol.ground_amplitude_deg * omega * Math.cos(groundTheta);
    const figureVelocity = protocol.figure_amplitude_deg
      * (omega + phaseVelocity) * Math.cos(figureTheta);
    return Object.freeze({
      time_s: time,
      stimulus_interval: phase.interval,
      figure_angle_command_deg: figureAngle,
      figure_angle_realized_deg: figureAngle,
      figure_velocity_command_deg_s: figureVelocity,
      ground_angle_command_deg: groundAngle,
      ground_angle_realized_deg: groundAngle,
      ground_velocity_command_deg_s: groundVelocity,
      relative_displacement_realized_deg: figureAngle - groundAngle,
      relative_phase_command_deg: phase.phase_deg,
      relative_phase_realized_deg: phase.phase_deg,
    });
  }

  function resolveProtocol(documentValue, protocolId) {
    if (!documentValue || documentValue.schema_version !== SCHEMA_VERSION) {
      throw new TypeError("unsupported paper protocol document");
    }
    const selected = documentValue.protocols && documentValue.protocols[protocolId];
    if (!selected) throw new RangeError("unknown protocol_id: " + protocolId);
    return Object.freeze(Object.assign({ protocol_id: protocolId }, documentValue.defaults, selected));
  }

  function paperRayDirection(azimuthDeg) {
    const angle = requireFinite(azimuthDeg, "azimuth_deg") * DEG2RAD;
    return Object.freeze([Math.cos(angle), -Math.sin(angle), 0]);
  }

  function torqueNmToDyneCm(value) {
    return requireFinite(value, "torque_Nm") * 1e7;
  }

  function torqueDyneCmToNm(value) {
    return requireFinite(value, "torque_dyne_cm") * 1e-7;
  }

  function sha256Utf8(textValue) {
    // Portable SHA-256 is needed because the deployed assay is served over
    // plain HTTP, where WebCrypto is not guaranteed to be exposed.
    const text = String(textValue);
    const bytes = typeof TextEncoder === "function"
      ? new TextEncoder().encode(text)
      : Uint8Array.from(unescape(encodeURIComponent(text)), (character) => character.charCodeAt(0));
    const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
    const data = new Uint8Array(paddedLength);
    data.set(bytes);
    data[bytes.length] = 0x80;
    const bitLength = bytes.length * 8;
    const view = new DataView(data.buffer);
    view.setUint32(paddedLength - 8, Math.floor(bitLength / 4294967296), false);
    view.setUint32(paddedLength - 4, bitLength >>> 0, false);
    const constants = new Uint32Array([
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ]);
    const hash = new Uint32Array([
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]);
    const words = new Uint32Array(64);
    const rotateRight = (value, count) => (value >>> count) | (value << (32 - count));
    for (let offset = 0; offset < data.length; offset += 64) {
      for (let index = 0; index < 16; index += 1) words[index] = view.getUint32(offset + index * 4, false);
      for (let index = 16; index < 64; index += 1) {
        const a = words[index - 15];
        const b = words[index - 2];
        const sigma0 = rotateRight(a, 7) ^ rotateRight(a, 18) ^ (a >>> 3);
        const sigma1 = rotateRight(b, 17) ^ rotateRight(b, 19) ^ (b >>> 10);
        words[index] = (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
      }
      let [a, b, c, d, e, f, g, h] = hash;
      for (let index = 0; index < 64; index += 1) {
        const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
        const choice = (e & f) ^ (~e & g);
        const temporary1 = (h + sum1 + choice + constants[index] + words[index]) >>> 0;
        const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
        const majority = (a & b) ^ (a & c) ^ (b & c);
        const temporary2 = (sum0 + majority) >>> 0;
        h = g; g = f; f = e; e = (d + temporary1) >>> 0;
        d = c; c = b; b = a; a = (temporary1 + temporary2) >>> 0;
      }
      hash[0] = (hash[0] + a) >>> 0; hash[1] = (hash[1] + b) >>> 0;
      hash[2] = (hash[2] + c) >>> 0; hash[3] = (hash[3] + d) >>> 0;
      hash[4] = (hash[4] + e) >>> 0; hash[5] = (hash[5] + f) >>> 0;
      hash[6] = (hash[6] + g) >>> 0; hash[7] = (hash[7] + h) >>> 0;
    }
    return Array.from(hash, (word) => word.toString(16).padStart(8, "0")).join("");
  }

  return Object.freeze({
    SCHEMA_VERSION,
    GENERATOR_NAME,
    DEFAULT_TEXTURE,
    wrap360,
    wrap180,
    createBinaryTexture,
    createScene,
    textureCell,
    luminanceAt,
    phaseSchedule,
    sampleProtocol,
    resolveProtocol,
    paperRayDirection,
    torqueNmToDyneCm,
    torqueDyneCmToNm,
    sha256Utf8,
  });
});
