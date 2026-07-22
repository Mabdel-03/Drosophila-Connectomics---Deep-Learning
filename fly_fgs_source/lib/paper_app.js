(function (root) {
  "use strict";

  const DEG2RAD = Math.PI / 180;
  const FIXED_DT_S = 1 / 400;
  const state = {
    active: true,
    started: false,
    bundle: null,
    model: null,
    protocolDocument: null,
    protocol: null,
    sceneDefinition: null,
    timeS: 0,
    accumulatorS: 0,
    output: 0,
    trace: [],
    threeGroup: null,
    groundMesh: null,
    figureMesh: null,
    legacyFigureMesh: null,
    legacyGroundMesh: null,
    setLegacyMode: null,
    replay: null,
    replayManifest: null,
    replayChecksumVerified: false,
    replayAuthoritative: false,
    perWingAvailable: false,
    replayRegistry: new Map(),
    currentReplayFrame: null,
    onCircuitSample: null,
    onCircuitError: null,
  };

  function element(id) { return document.getElementById(id); }

  async function sha256Hex(textValue) {
    if (!globalThis.crypto || !globalThis.crypto.subtle) {
      return PaperFGS.sha256Utf8(textValue);
    }
    const bytes = new TextEncoder().encode(textValue);
    const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  function makeTextureCanvas(texture, alphaMask) {
    const canvas = document.createElement("canvas");
    canvas.width = texture.columns;
    canvas.height = texture.rows;
    const context = canvas.getContext("2d", { alpha: Boolean(alphaMask) });
    const image = context.createImageData(canvas.width, canvas.height);
    for (let row = 0; row < texture.rows; row += 1) {
      for (let column = 0; column < texture.columns; column += 1) {
        const source = row * texture.columns + column;
        const destination = source * 4;
        const bit = texture.values[source];
        const inFigureMask = column >= 8 && column < 12;
        // Three.js alphaMap reads the texture's colour channel, not its alpha
        // channel.  Keep the mask independent of the random-dot value so black
        // figure cells remain opaque.
        const gray = alphaMask ? (inFigureMask ? 255 : 0) : (bit ? 255 : 32);
        image.data[destination] = gray;
        image.data[destination + 1] = gray;
        image.data[destination + 2] = gray;
        image.data[destination + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    return canvas;
  }

  function canvasTexture(canvas) {
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
    return texture;
  }

  function rebuildThreeScene(scene) {
    if (!scene || typeof THREE === "undefined") return;
    if (state.threeGroup) scene.remove(state.threeGroup);
    const group = new THREE.Group();
    const source = state.sceneDefinition.ground_texture;
    const groundMap = canvasTexture(makeTextureCanvas(source, false));
    const figureMap = canvasTexture(makeTextureCanvas(state.sceneDefinition.figure_texture, false));
    const alphaMap = canvasTexture(makeTextureCanvas(source, true));
    const cylinder = (radius, material) => new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, 16, 120, 1, true),
      material,
    );
    // Browser scene units use 333.333 units/m here: 12 units = the tagged
    // 36 mm outer radius and 11.666... units = the tagged 35 mm inner radius.
    state.groundMesh = cylinder(12, new THREE.MeshBasicMaterial({
      map: groundMap,
      side: THREE.BackSide,
      toneMapped: false,
    }));
    state.figureMesh = cylinder(35 / 3, new THREE.MeshBasicMaterial({
      map: figureMap,
      alphaMap,
      transparent: true,
      alphaTest: 0.5,
      depthWrite: false,
      side: THREE.BackSide,
      toneMapped: false,
    }));
    state.groundMesh.userData.paperRadiusM = 0.036;
    state.figureMesh.userData.paperRadiusM = 0.035;
    group.add(state.groundMesh);
    group.add(state.figureMesh);
    group.visible = state.active;
    scene.add(group);
    state.threeGroup = group;
  }

  function resetModel() {
    if (!state.bundle || !state.protocolDocument) return;
    const protocolId = element("paperProtocol").value;
    const registeredReplay = state.replayRegistry.get(protocolId) || null;
    if (registeredReplay) {
      state.replay = registeredReplay.replay;
      state.replayManifest = registeredReplay.manifest;
      state.replayChecksumVerified = registeredReplay.verified;
    } else {
      state.replay = null;
      state.replayManifest = null;
      state.replayChecksumVerified = false;
    }
    state.protocol = PaperFGS.resolveProtocol(state.protocolDocument, protocolId);
    state.sceneDefinition = PaperFGS.createScene({
      relationship_mode: element("paperTextureMode").value,
      master_seed: 123456,
      ground_seed: 123456,
      figure_seed: 654321,
    });
    state.model = FGModel.init(state.bundle, {
      dt: FIXED_DT_S,
      figW: 12,
      figH: 180,
      startAz: 30,
      paperMode: true,
      paperScene: state.sceneDefinition,
      inputScale: 3,
      scale: 3,
      llpcBoost: 1,
    });
    state.timeS = 0;
    state.accumulatorS = 0;
    state.output = 0;
    state.trace = [];
    const replayIsV3 = Boolean(
      state.replay && state.replay.schema_version === "paper_fgs_web_replay.v3"
    );
    const v3MetrologyPassed = !replayIsV3 || Boolean(
      state.replay.metrology
      && state.replay.metrology.bilateral_wing_calibration_passed === true
      && state.replay.metrology.per_wing_metrology_passed === true
      && state.replay.metrology.aerodynamic_probe_passed === true
    );
    state.replayAuthoritative = Boolean(
      state.replay && state.replay.protocol_id === protocolId
      && state.replay.scientific_status === "authoritative_native_torque"
      && state.replay.validation && state.replay.validation.apparatus_passed === true
      && state.replay.metrology && state.replay.metrology.calibration_passed === true
      && state.replay.metrology.sampling_passed === true
      && state.replay.metrology.convergence_passed === true
      && state.replay.metrology.dual_meter_passed === true
      && state.replay.metrology.paper_extraction_passed === true
      && v3MetrologyPassed
      && state.replayChecksumVerified
    );
    state.perWingAvailable = Boolean(
      state.replayAuthoritative && replayIsV3
      && state.replay.channel_authority
      && state.replay.channel_authority.wing_reported_yaw_torque_Nm === "calibrated_simulation_decomposition"
      && Array.isArray(state.replay.frames)
      && state.replay.frames.some((frame) => frame.torque_channels_Nm && frame.torque_channels_Nm.left_wing && frame.torque_channels_Nm.right_wing)
    );
    const trialSelect = element("paperTrial");
    trialSelect.innerHTML = "";
    const trialCount = state.replayAuthoritative ? Number(state.replay.trial_count) : 0;
    if (trialCount > 0) {
      for (let trialId = 0; trialId < trialCount; trialId += 1) {
        const option = document.createElement("option");
        option.value = String(trialId);
        option.textContent = `trial ${trialId + 1}`;
        trialSelect.appendChild(option);
      }
      trialSelect.disabled = false;
    } else {
      const option = document.createElement("option");
      option.value = "0";
      option.textContent = "unavailable";
      trialSelect.appendChild(option);
      trialSelect.disabled = true;
    }
    rebuildThreeScene(state.options.scene);
    updateStatus();
    const initialSample = PaperFGS.sampleProtocol(state.protocol, 0);
    emitCircuitSample(initialSample, FGModel.readout(state.model), true);
    updateDisplay(initialSample);
  }

  function updateStatus() {
    const status = element("paperStatus");
    if (state.replayAuthoritative) {
      status.dataset.mode = "paper-replay";
      status.classList.add("authoritative");
      const runId = state.replayManifest && state.replayManifest.run_id
        ? state.replayManifest.run_id.slice(0, 12) : "unknown";
      const wingStatus = state.perWingAvailable
        ? " Bilateral wing-root calibration and aerodynamic closure also passed."
        : " This v2 replay contains authoritative total torque but no per-wing channels.";
      status.innerHTML = `<strong>AUTHORITATIVE NATIVE LOAD-CELL REPLAY</strong><br/>Absolute whole-fly yaw torque is checksum-verified against immutable artifact ${runId}…; total-meter, sampling, convergence, and apparatus gates passed.${wingStatus}`;
    } else {
      status.dataset.mode = "paper-live-preview";
      status.classList.remove("authoritative");
      status.innerHTML = "<strong>LIVE PREVIEW — NOT PHYSICAL TORQUE</strong><br/>No matching native tether artifact is loaded; the live trace is a dimensionless circuit steering signal.";
    }
    element("paperAbsoluteBadge").classList.toggle("ok", state.replayAuthoritative);
    element("paperAbsoluteBadge").textContent = state.replayAuthoritative
      ? "absolute physical units" : "arbitrary units";
    element("paperNoFitBadge").classList.toggle("ok", state.replayAuthoritative);
    element("paperSemBadge").classList.toggle("ok", state.replayAuthoritative);
    element("paperWholeFlyBadge").classList.toggle("ok", state.replayAuthoritative);
    element("paperWingBadge").classList.toggle("ok", state.perWingAvailable);
    element("paperWingBadge").textContent = state.perWingAvailable
      ? "calibrated wing-root decomposition" : "wing torque unavailable";
  }

  function replayFrame(timeS) {
    if (!state.replayAuthoritative || !Array.isArray(state.replay.frames)) return null;
    const frames = state.replay.frames;
    const rate = Number(state.replay.replay_rate_hz) || 200;
    return frames[Math.min(frames.length - 1, Math.max(0, Math.round(timeS * rate)))];
  }

  function selectedTrialIndex(frame) {
    const selected = Number.parseInt(element("paperTrial").value, 10) || 0;
    const count = frame && Array.isArray(frame.raw_trial_reported_yaw_torque_Nm)
      ? frame.raw_trial_reported_yaw_torque_Nm.length : 0;
    return Math.min(Math.max(0, selected), Math.max(0, count - 1));
  }

  function selectedSignalProduct(frame) {
    const requested = element("paperSignalProduct").value;
    const products = frame && frame.torque_products_Nm;
    if (products && products[requested]) return products[requested];
    return frame ? {
      mean: Number(frame.reported_yaw_torque_Nm) || 0,
      sem: Number(frame.reported_yaw_torque_sem_Nm) || 0,
    } : { mean: 0, sem: 0 };
  }

  function selectedChannelProduct(frame, channel) {
    const requested = element("paperSignalProduct").value;
    const record = frame && frame.torque_channels_Nm && frame.torque_channels_Nm[channel];
    const product = record && record.products && record.products[requested];
    return product || null;
  }

  function selectedChannelRaw(frame, channel, trialIndex) {
    const record = frame && frame.torque_channels_Nm && frame.torque_channels_Nm[channel];
    return record && Array.isArray(record.raw_trials)
      ? Number(record.raw_trials[trialIndex]) : NaN;
  }

  function formatTorque(valueNm, units, digits) {
    if (!Number.isFinite(Number(valueNm))) return "—";
    return units === "Nm"
      ? Number(valueNm).toExponential(digits == null ? 3 : digits) + " N·m"
      : PaperFGS.torqueNmToDyneCm(Number(valueNm)).toFixed(digits == null ? 3 : digits) + " dyne·cm";
  }

  function emitCircuitSample(sample, readout, reset) {
    if (!state.onCircuitSample) return;
    try {
      state.onCircuitSample(Object.freeze({
        model: state.model,
        readout,
        stimulus: sample,
        previewSteeringAu: state.output,
        mode: state.replayAuthoritative ? "paper-replay" : "paper-live-preview",
        authoritativeReplay: state.replayAuthoritative,
        reset: Boolean(reset),
      }));
    } catch (error) {
      console.error("Paper neural monitoring callback failed", error);
      if (state.onCircuitError) state.onCircuitError(error);
      state.onCircuitSample = null;
    }
  }

  function previewStep(sample) {
    FGModel.stepPaper(state.model, {
      figureAngleDeg: sample.figure_angle_realized_deg,
      groundAngleDeg: sample.ground_angle_realized_deg,
      figureVelocityDegS: sample.figure_velocity_command_deg_s,
      groundVelocityDegS: sample.ground_velocity_command_deg_s,
    });
    const readout = FGModel.readout(state.model);
    const right = (readout.nod1R || 0) + (readout.llpcR || 0);
    const left = (readout.nod1L || 0) + (readout.llpcL || 0);
    state.output = right - left;
    emitCircuitSample(sample, readout, false);
    return state.output;
  }

  function updateMeshes(sample) {
    if (!state.groundMesh || !state.figureMesh) return;
    state.groundMesh.rotation.y = -sample.ground_angle_realized_deg * DEG2RAD;
    state.figureMesh.rotation.y = -sample.figure_angle_realized_deg * DEG2RAD;
  }

  function drawStimulus(sample) {
    const canvas = element("paperStimulusCanvas");
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const rows = 24;
    for (let y = 0; y < rows; y += 1) {
      const elevation = 60 - (y + 0.5) * 120 / rows;
      for (let x = 0; x < width; x += 1) {
        const azimuth = -90 + (x + 0.5) * 180 / width;
        const value = PaperFGS.luminanceAt(state.sceneDefinition, azimuth, elevation, {
          figure_angle_deg: sample.figure_angle_realized_deg,
          ground_angle_deg: sample.ground_angle_realized_deg,
          figure_mean_azimuth_deg: 30,
          figure_width_deg: 12,
        }).normalized_luminance;
        const gray = Math.round(value * 255);
        context.fillStyle = `rgb(${gray},${gray},${gray})`;
        context.fillRect(x, y * height / rows, 1, height / rows + 1);
      }
    }
    context.strokeStyle = "#38e2c8";
    context.lineWidth = 1;
    const center = (30 + sample.figure_angle_realized_deg + 90) / 180 * width;
    const half = 6 / 180 * width;
    context.strokeRect(center - half, 0.5, half * 2, height - 1);
    context.fillStyle = "rgba(5,8,12,.72)";
    context.fillRect(5, 5, 184, 17);
    context.fillStyle = "#dfe8f2";
    context.font = "10px ui-monospace,monospace";
    context.fillText("paper visual field −90° … +90°", 10, 17);
  }

  function preparePlot(canvasId) {
    const canvas = element(canvasId);
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#080c12";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "#1e2a3a";
    context.lineWidth = 1;
    return { canvas, context };
  }

  function visibleTrace() {
    return state.trace.slice(Math.max(0, state.trace.length - 600));
  }

  function traceMaximum(points, keys, floor) {
    let maximum = Number(floor) || 1e-15;
    points.forEach((point) => keys.forEach((key) => {
      const value = Number(point[key]);
      if (Number.isFinite(value)) maximum = Math.max(maximum, Math.abs(value));
    }));
    return maximum;
  }

  function drawTraceLine(context, points, key, color, xForTime, yForValue, options) {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = (options && options.width) || 1.4;
    context.setLineDash((options && options.dash) || []);
    context.beginPath();
    let started = false;
    points.forEach((point) => {
      const value = Number(point[key]);
      if (!Number.isFinite(value)) return;
      const x = xForTime(point.time);
      const y = yForValue(value);
      if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
    });
    if (started) context.stroke();
    context.restore();
  }

  function interpolatePaperTorqueNm(timeS) {
    const reference = state.replay && state.replay.paper_reference;
    const times = reference && reference.time_s;
    const values = reference && reference.torque_dyne_cm;
    if (!Array.isArray(times) || !Array.isArray(values) || times.length < 2) return NaN;
    if (timeS <= Number(times[0])) return Number(values[0]) * 1e-7;
    if (timeS >= Number(times[times.length - 1])) return Number(values[values.length - 1]) * 1e-7;
    let low = 0; let high = times.length - 1;
    while (high - low > 1) {
      const middle = Math.floor((low + high) / 2);
      if (Number(times[middle]) <= timeS) low = middle; else high = middle;
    }
    const fraction = (timeS - Number(times[low])) / Math.max(1e-15, Number(times[high]) - Number(times[low]));
    return (Number(values[low]) + fraction * (Number(values[high]) - Number(values[low]))) * 1e-7;
  }

  function drawWingPlot() {
    const { canvas, context } = preparePlot("paperWingPlotCanvas");
    const points = visibleTrace();
    context.font = "9px system-ui";
    if (!state.perWingAvailable || points.length < 2) {
      context.fillStyle = "#65778d";
      context.fillText(state.replayAuthoritative
        ? "v2 total-only replay · per-wing channels unavailable"
        : "physical wing torque unavailable in live browser preview", 9, 18);
      context.fillText("neural/muscle activity remains steering_signal_au", 9, 34);
      return;
    }
    const timeMin = points[0].time;
    const timeMax = points[points.length - 1].time;
    const xForTime = (time) => 5 + (time - timeMin) / Math.max(1e-12, timeMax - timeMin) * (canvas.width - 10);
    const keys = ["leftWing", "rightWing", "wingSum", "nonwingResidual", "leftAero", "rightAero"];
    const maximum = traceMaximum(points, keys, 1e-15);
    const midline = canvas.height * 0.54;
    const yForValue = (value) => midline - value / maximum * canvas.height * 0.38;
    context.strokeStyle = "#263449";
    context.beginPath(); context.moveTo(0, midline); context.lineTo(canvas.width, midline); context.stroke();
    drawTraceLine(context, points, "leftWing", "#5aa9ff", xForTime, yForValue, { width: 1.7 });
    drawTraceLine(context, points, "rightWing", "#ff9f43", xForTime, yForValue, { width: 1.7 });
    drawTraceLine(context, points, "wingSum", "#38e2c8", xForTime, yForValue, { width: 1.4 });
    drawTraceLine(context, points, "nonwingResidual", "#9aa7b5", xForTime, yForValue, { dash: [4, 3] });
    drawTraceLine(context, points, "leftAero", "#5aa9ff", xForTime, yForValue, { dash: [2, 3], width: 1 });
    drawTraceLine(context, points, "rightAero", "#ff9f43", xForTime, yForValue, { dash: [2, 3], width: 1 });
    context.fillStyle = "#5aa9ff"; context.fillText("left", 7, 12);
    context.fillStyle = "#ff9f43"; context.fillText("right", 38, 12);
    context.fillStyle = "#38e2c8"; context.fillText("sum", 76, 12);
    context.fillStyle = "#9aa7b5"; context.fillText("residual", 105, 12);
    context.fillStyle = "#65778d"; context.fillText("dashed color = aerodynamic only", 171, 12);
  }

  function drawPlot() {
    drawWingPlot();
    const { canvas, context } = preparePlot("paperPlotCanvas");
    const points = visibleTrace();
    context.font = "9px system-ui";
    if (points.length < 2) return;
    if (!state.replayAuthoritative) {
      const maximum = traceMaximum(points, ["output"], 1e-9);
      const y = (value) => canvas.height * 0.52 - value / maximum * canvas.height * 0.36;
      const x = (time) => 5 + (time - points[0].time) / Math.max(1e-12, points[points.length - 1].time - points[0].time) * (canvas.width - 10);
      drawTraceLine(context, points, "output", "#ffcf5d", x, y, { width: 1.6 });
      context.fillStyle = "#8fa3ba";
      context.fillText("live steering signal (a.u.) · no physical torque", 8, 13);
      return;
    }
    const timeMin = points[0].time;
    const timeMax = points[points.length - 1].time;
    const xForTime = (time) => 5 + (time - timeMin) / Math.max(1e-12, timeMax - timeMin) * (canvas.width - 10);
    const paperReference = state.replay.paper_reference;
    const paperPoints = paperReference && Array.isArray(paperReference.time_s)
      ? paperReference.time_s.map((time, index) => ({
        time: Number(time),
        value: Number(paperReference.torque_dyne_cm[index]) * 1e-7,
        uncertainty: Number((paperReference.digitization_uncertainty_dyne_cm || [])[index] || 0) * 1e-7,
      })).filter((point) => point.time >= timeMin && point.time <= timeMax) : [];
    const maximum = Math.max(
      traceMaximum(points, ["totalMean", "totalRaw"], 1e-15),
      ...paperPoints.map((point) => Math.abs(point.value) + point.uncertainty),
    );
    const torqueTop = 18;
    const torqueBottom = 126;
    const torqueMid = (torqueTop + torqueBottom) / 2;
    const yForTorque = (value) => torqueMid - value / maximum * (torqueBottom - torqueTop) * 0.46;
    context.strokeStyle = "#263449";
    context.beginPath(); context.moveTo(0, torqueMid); context.lineTo(canvas.width, torqueMid); context.stroke();
    context.fillStyle = "rgba(56,226,200,.14)";
    context.beginPath();
    points.forEach((point, index) => {
      const y = yForTorque(Number(point.totalMean) + Number(point.totalSem || 0));
      if (index === 0) context.moveTo(xForTime(point.time), y); else context.lineTo(xForTime(point.time), y);
    });
    points.slice().reverse().forEach((point) => context.lineTo(
      xForTime(point.time), yForTorque(Number(point.totalMean) - Number(point.totalSem || 0)),
    ));
    context.closePath(); context.fill();
    if (paperPoints.length > 1) {
      context.fillStyle = "rgba(255,255,255,.10)";
      context.beginPath();
      paperPoints.forEach((point, index) => {
        const y = yForTorque(point.value + point.uncertainty);
        if (index === 0) context.moveTo(xForTime(point.time), y); else context.lineTo(xForTime(point.time), y);
      });
      paperPoints.slice().reverse().forEach((point) => context.lineTo(xForTime(point.time), yForTorque(point.value - point.uncertainty)));
      context.closePath(); context.fill();
      const asTrace = paperPoints.map((point) => ({ time: point.time, paper: point.value }));
      drawTraceLine(context, asTrace, "paper", "#ffffff", xForTime, yForTorque, { width: 1.2 });
    }
    drawTraceLine(context, points, "totalRaw", "rgba(56,226,200,.42)", xForTime, yForTorque, { width: 0.8 });
    drawTraceLine(context, points, "totalMean", "#38e2c8", xForTime, yForTorque, { width: 1.8 });
    const residualPoints = points.map((point) => ({
      time: point.time,
      residual: Number(point.totalMean) - interpolatePaperTorqueNm(point.time),
    })).filter((point) => Number.isFinite(point.residual));
    const residualMaximum = traceMaximum(residualPoints, ["residual"], 1e-15);
    const residualMid = 160;
    const yForResidual = (value) => residualMid - value / residualMaximum * 21;
    context.strokeStyle = "#263449";
    context.beginPath(); context.moveTo(0, residualMid); context.lineTo(canvas.width, residualMid); context.stroke();
    drawTraceLine(context, residualPoints, "residual", "#a8b3c1", xForTime, yForResidual, { width: 1.1 });
    context.fillStyle = "#38e2c8"; context.fillText("total mean ± simulation SEM", 7, 11);
    context.fillStyle = "#ffffff"; context.fillText("paper ± digitization", 159, 11);
    context.fillStyle = "#8fa3ba"; context.fillText("simulation − paper residual", 7, 181);
  }

  function updateDisplay(sample) {
    updateMeshes(sample);
    drawStimulus(sample);
    element("paperTime").textContent = sample.time_s.toFixed(3) + " s";
    element("paperInterval").textContent = sample.stimulus_interval;
    element("paperFigure").textContent = `${sample.figure_angle_command_deg.toFixed(2)}° / ${sample.figure_angle_realized_deg.toFixed(2)}°`;
    element("paperGround").textContent = `${sample.ground_angle_command_deg.toFixed(2)}° / ${sample.ground_angle_realized_deg.toFixed(2)}°`;
    element("paperPhase").textContent = sample.relative_phase_realized_deg.toFixed(1) + "°";
    const units = element("paperUnits").value;
    if (state.replayAuthoritative) {
      const frame = replayFrame(sample.time_s);
      state.currentReplayFrame = frame;
      const trialIndex = selectedTrialIndex(frame);
      const torqueNm = frame ? Number(frame.raw_trial_reported_yaw_torque_Nm[trialIndex]) : 0;
      const product = selectedSignalProduct(frame);
      const meanNm = Number(product.mean) || 0;
      const semNm = Number(product.sem) || 0;
      element("paperOutput").textContent = units === "Nm"
        ? torqueNm.toExponential(3) + " N·m"
        : PaperFGS.torqueNmToDyneCm(torqueNm).toFixed(3) + " dyne·cm";
      element("paperMean").textContent = units === "Nm"
        ? `${meanNm.toExponential(3)} ± ${semNm.toExponential(2)} N·m`
        : `${PaperFGS.torqueNmToDyneCm(meanNm).toFixed(3)} ± ${PaperFGS.torqueNmToDyneCm(semNm).toFixed(3)} dyne·cm`;
      const productName = element("paperSignalProduct").value;
      const comparisons = state.replay.analysis && state.replay.analysis.paper_figure3_comparisons;
      const productAnalysis = comparisons && comparisons[productName];
      const harmonicNm = productAnalysis
        ? Number(productAnalysis.first_harmonic.simulation_amplitude_dyne_cm) * 1e-7
        : Number((state.replay.analysis || {}).first_harmonic_amplitude_n_m) || 0;
      const phaseDeg = productAnalysis
        ? Number(productAnalysis.first_harmonic.simulation_phase_relative_ground_position_deg)
        : Number((state.replay.analysis || {}).first_harmonic_phase_relative_ground_position_deg) || 0;
      element("paperHarmonic").textContent = units === "Nm"
        ? `${harmonicNm.toExponential(2)} N·m @ ${phaseDeg.toFixed(1)}°`
        : `${PaperFGS.torqueNmToDyneCm(harmonicNm).toFixed(3)} dyne·cm @ ${phaseDeg.toFixed(1)}°`;
      element("paperValidation").textContent = "apparatus passed · behavior reported, not forced";
      const checksum = state.replayManifest && state.replayManifest.run_id
        ? state.replayManifest.run_id.slice(0, 8) : "unknown";
      element("paperMeter").textContent = `${state.replay.torque_meter_mode} / ${productName} / ${checksum}`;
      if (state.perWingAvailable && frame) {
        const leftRaw = selectedChannelRaw(frame, "left_wing", trialIndex);
        const rightRaw = selectedChannelRaw(frame, "right_wing", trialIndex);
        const sumRaw = selectedChannelRaw(frame, "wing_sum", trialIndex);
        const residualRaw = selectedChannelRaw(frame, "nonwing_residual", trialIndex);
        element("paperWingInstant").textContent = `${formatTorque(leftRaw, units, 3)} / ${formatTorque(rightRaw, units, 3)}`;
        element("paperWingBalance").textContent = `${formatTorque(sumRaw, units, 3)} / ${formatTorque(residualRaw, units, 3)}`;
        const channels = (state.replay.analysis || {}).mechanical_channels || {};
        const leftBaseAnalysis = channels.left_wing || {};
        const rightBaseAnalysis = channels.right_wing || {};
        const leftAnalysis = (leftBaseAnalysis.signal_products || {})[productName] || leftBaseAnalysis;
        const rightAnalysis = (rightBaseAnalysis.signal_products || {})[productName] || rightBaseAnalysis;
        element("paperWingPost").textContent = `${formatTorque(leftAnalysis.post_transition_mean_torque_n_m, units, 3)} / ${formatTorque(rightAnalysis.post_transition_mean_torque_n_m, units, 3)}`;
        const leftAmplitude = Number(leftAnalysis.first_harmonic_amplitude_n_m);
        const rightAmplitude = Number(rightAnalysis.first_harmonic_amplitude_n_m);
        const leftPhase = Number(leftAnalysis.first_harmonic_phase_relative_ground_position_deg);
        const rightPhase = Number(rightAnalysis.first_harmonic_phase_relative_ground_position_deg);
        element("paperWingHarmonic").textContent = `${formatTorque(leftAmplitude, units, 2)} @ ${Number.isFinite(leftPhase) ? leftPhase.toFixed(1) : "—"}° / ${formatTorque(rightAmplitude, units, 2)} @ ${Number.isFinite(rightPhase) ? rightPhase.toFixed(1) : "—"}°`;
      } else {
        element("paperWingInstant").textContent = "unavailable";
        element("paperWingBalance").textContent = "unavailable";
        element("paperWingPost").textContent = "unavailable";
        element("paperWingHarmonic").textContent = "unavailable";
      }
      const scores = productAnalysis && productAnalysis.scores;
      element("paperScores").textContent = scores
        ? `${Number(scores.rmse_dyne_cm).toFixed(3)} dyne·cm / ${Number(scores.normalized_rmse_percent_of_paper_range).toFixed(1)}%`
        : "unavailable";
      element("paperScores2").textContent = scores
        ? `${Number(scores.mae_dyne_cm).toFixed(3)} / ${Number(scores.mean_bias_dyne_cm).toFixed(3)} / ${Number(scores.waveform_correlation).toFixed(3)}`
        : "unavailable";
    } else {
      state.currentReplayFrame = null;
      element("paperOutput").textContent = state.output.toFixed(4) + " a.u.";
      element("paperMean").textContent = "not physical";
      element("paperHarmonic").textContent = "not available";
      element("paperValidation").textContent = "live preview only";
      element("paperMeter").textContent = "browser / steering_signal_au";
      element("paperWingInstant").textContent = "unavailable";
      element("paperWingBalance").textContent = "unavailable";
      element("paperWingPost").textContent = "unavailable";
      element("paperWingHarmonic").textContent = "unavailable";
      element("paperScores").textContent = "unavailable";
      element("paperScores2").textContent = "unavailable";
    }
    drawPlot();
  }

  function tick(deltaS) {
    if (!state.active || !state.model || !state.protocol) return;
    state.accumulatorS += Math.min(Number(deltaS) || 0, 0.05);
    let steps = 0;
    while (state.accumulatorS + 1e-12 >= FIXED_DT_S && steps < 24) {
      state.timeS += FIXED_DT_S;
      if (state.timeS > state.protocol.trial_duration_s) { state.timeS = 0; state.trace = []; }
      const sample = PaperFGS.sampleProtocol(state.protocol, state.timeS);
      const previewOutput = previewStep(sample);
      const frame = replayFrame(state.timeS);
      // The browser circuit always advances for monitoring. A verified replay
      // remains the sole source of the plotted physical torque channel.
      const output = frame ? Number(selectedSignalProduct(frame).mean) : previewOutput;
      if (Math.round(state.timeS / FIXED_DT_S) % 2 === 0) {
        const trialIndex = selectedTrialIndex(frame);
        const totalProduct = frame ? selectedSignalProduct(frame) : null;
        const channelValue = (name, property) => {
          const product = selectedChannelProduct(frame, name);
          return product ? Number(product[property]) : NaN;
        };
        state.trace.push({
          figure: sample.figure_angle_realized_deg,
          ground: sample.ground_angle_realized_deg,
          output,
          totalMean: totalProduct ? Number(totalProduct.mean) : NaN,
          totalSem: totalProduct ? Number(totalProduct.sem) : NaN,
          totalRaw: frame && Array.isArray(frame.raw_trial_reported_yaw_torque_Nm)
            ? Number(frame.raw_trial_reported_yaw_torque_Nm[trialIndex]) : NaN,
          leftWing: channelValue("left_wing", "mean"),
          rightWing: channelValue("right_wing", "mean"),
          wingSum: channelValue("wing_sum", "mean"),
          nonwingResidual: channelValue("nonwing_residual", "mean"),
          leftAero: channelValue("left_wing_aerodynamic", "mean"),
          rightAero: channelValue("right_wing_aerodynamic", "mean"),
          time: state.timeS,
        });
        if (state.trace.length > 1200) state.trace.shift();
      }
      state.accumulatorS = Math.max(0, state.accumulatorS - FIXED_DT_S);
      steps += 1;
      updateDisplay(sample);
    }
  }

  function applyMeasuredWingFrame(wingL, wingR) {
    if (!state.replayAuthoritative || !state.currentReplayFrame || !wingL || !wingR) return;
    const trialIndex = selectedTrialIndex(state.currentReplayFrame);
    const traces = state.currentReplayFrame.raw_trial_wing_joint_position_rad;
    if (!Array.isArray(traces) || !Array.isArray(traces[trialIndex]) || traces[trialIndex].length !== 6) return;
    const q = traces[trialIndex].map(Number);
    // Replay the measured FlyBody yaw/roll/pitch triplets.  These assignments
    // are a display-frame transform only; physical values remain the native
    // joint arrays in the immutable artifact.
    wingL.phi.rotation.y = -q[0];
    wingL.theta.rotation.x = q[1];
    wingL.alpha.rotation.z = -q[2];
    wingR.phi.rotation.y = q[3];
    wingR.theta.rotation.x = q[4];
    wingR.alpha.rotation.z = q[5];
  }

  function setPaperActive(active) {
    state.active = Boolean(active);
    element("paperPanel").classList.toggle("show", state.active);
    element("paperModeBtn").classList.toggle("on", state.active);
    element("legacyModeBtn").classList.toggle("on", !state.active);
    if (state.threeGroup) state.threeGroup.visible = state.active;
    if (state.legacyFigureMesh) state.legacyFigureMesh.visible = !state.active;
    if (state.legacyGroundMesh) state.legacyGroundMesh.visible = false;
    if (state.setLegacyMode) state.setLegacyMode(!state.active);
    element("paperStatus").dataset.mode = state.active
      ? (state.replayAuthoritative ? "paper-replay" : "paper-live-preview")
      : "legacy-tracking";
  }

  async function start(options) {
    if (state.started) return;
    state.started = true;
    state.options = options;
    state.bundle = options.bundle;
    state.legacyFigureMesh = options.figureMesh || null;
    state.legacyGroundMesh = options.groundMesh || null;
    state.setLegacyMode = options.setLegacyMode || null;
    state.onCircuitSample = typeof options.onCircuitSample === "function"
      ? options.onCircuitSample : null;
    state.onCircuitError = typeof options.onCircuitError === "function"
      ? options.onCircuitError : null;
    const protocolResponse = await fetch("data/paper_protocols.json", { cache: "no-store" });
    if (!protocolResponse.ok) throw new Error("paper protocol definitions failed to load");
    state.protocolDocument = await protocolResponse.json();
    const select = element("paperProtocol");
    for (const [protocolId, protocol] of Object.entries(state.protocolDocument.protocols)) {
      const option = document.createElement("option");
      option.value = protocolId;
      option.textContent = protocol.label;
      select.appendChild(option);
    }
    async function loadReplayPair(replayUrl, manifestUrl) {
      const [replayResponse, manifestResponse] = await Promise.all([
        fetch(replayUrl, { cache: "no-store" }),
        fetch(manifestUrl, { cache: "no-store" }),
      ]);
      if (replayResponse.ok && manifestResponse.ok) {
        const replayText = await replayResponse.text();
        const replay = JSON.parse(replayText);
        const manifest = await manifestResponse.json();
        const expected = manifest.files && manifest.files["web_replay.json"]
          && manifest.files["web_replay.json"].sha256;
        const observed = await sha256Hex(replayText);
        const verified = Boolean(
          expected && observed && expected === observed
          && manifest.scientific_status === "authoritative_native_torque",
        );
        return { replay, manifest, verified };
      }
      return null;
    }
    try {
      const indexResponse = await fetch("data/paper_replay_index.json", { cache: "no-store" });
      if (indexResponse.ok) {
        const index = await indexResponse.json();
        const entries = await Promise.all(Object.entries(index.replays || {}).map(
          async ([protocolId, entry]) => [protocolId, await loadReplayPair(entry.replay, entry.manifest)],
        ));
        entries.forEach(([protocolId, entry]) => { if (entry) state.replayRegistry.set(protocolId, entry); });
      } else {
        const legacy = await loadReplayPair(
          "data/paper_replay.json", "data/paper_replay.json.manifest.json",
        );
        if (legacy) state.replayRegistry.set(legacy.replay.protocol_id, legacy);
      }
    } catch (error) {
      console.info("No authoritative paper replay installed", error);
    }
    select.addEventListener("change", resetModel);
    element("paperTextureMode").addEventListener("change", resetModel);
    element("paperRestart").addEventListener("click", resetModel);
    element("paperUnits").addEventListener("change", () => updateDisplay(PaperFGS.sampleProtocol(state.protocol, state.timeS)));
    element("paperTrial").addEventListener("change", () => {
      state.trace = [];
      updateDisplay(PaperFGS.sampleProtocol(state.protocol, state.timeS));
    });
    element("paperSignalProduct").addEventListener("change", () => {
      state.trace = [];
      updateDisplay(PaperFGS.sampleProtocol(state.protocol, state.timeS));
    });
    element("paperModeBtn").addEventListener("click", () => setPaperActive(true));
    element("legacyModeBtn").addEventListener("click", () => setPaperActive(false));
    resetModel();
    setPaperActive(true);
  }

  root.PaperUI = Object.freeze({
    start,
    tick,
    setPaperActive,
    applyMeasuredWingFrame,
    currentMode: () => (state.active
      ? (state.replayAuthoritative ? "paper-replay" : "paper-live-preview")
      : "legacy-tracking"),
    isPaperActive: () => state.active,
  });
})(typeof window !== "undefined" ? window : globalThis);
