/* fgmodel.js — the REAL nod1_sim conductance model, ported to plain JS and
 * refactored from a fixed-steps batch (loop.ts runCore) into an incremental
 * stepper so the fly's live heading can drive it frame-by-frame.
 *   - hinesPrepare/hinesSolve: verbatim from src/sim/hines.ts (tree-elimination cable solver)
 *   - the conductance iteration: verbatim from src/sim/loop.ts runCore body
 *     (signed external drive, e_t4_null null-direction reversal, release model,
 *      alpha-conductance synapses, point + cable cells in one solve)
 *   - the drive: src/sim/drive.ts opticFlowDriveBatch, re-expressed in a RETINAL
 *     frame with a scene-history ring so a moving observer (the turning fly)
 *     produces the correct self-motion / wide-field signal.
 * Bundle (cables, synapse events, retinotopic cells) is emitted by the backend's
 * own _cable_model/_build_synapse_events (emit_bundle.py), downsampled for speed. */
(function (root) {
  const clip = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
  const sigmoid = (x) => 1 / (1 + Math.exp(-clip(x, -40, 40)));
  function wrap180(v) { let m = (v + 180) % 360; if (m < 0) m += 360; return m - 180; }
  function angDist(a, b) { return Math.abs(wrap180(a - b)); }
  function hash01(x, y, seed) {
    let n = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(seed | 0, 1442695041)) >>> 0;
    n = Math.imul((n ^ (n >>> 13)) >>> 0, 1274126177) >>> 0;
    n = (n ^ (n >>> 16)) >>> 0;
    return n / 4294967295;
  }
  const LPW = [0, 1, 2, 3, 4].map((k) => Math.exp(-k * 0.5));

  // ---- Hines tree-elimination cable solver (verbatim port of hines.ts) ----
  function hinesPrepare(parents, coupling, diag, cap, dt) {
    const n = parents.length;
    const d0 = new Float64Array(n);
    for (let i = 0; i < n; i++) d0[i] = diag[i] + cap[i] / dt;
    const pending = new Int32Array(n);
    for (let i = 0; i < n; i++) if (parents[i] >= 0) pending[parents[i]]++;
    const generation = new Int32Array(n);
    const queue = [];
    for (let i = 0; i < n; i++) if (pending[i] === 0) queue.push(i);
    let head = 0;
    while (head < queue.length) {
      const i = queue[head++]; const p = parents[i];
      if (p >= 0) { if (generation[i] + 1 > generation[p]) generation[p] = generation[i] + 1; if (--pending[p] === 0) queue.push(p); }
    }
    let maxGen = 0; for (let i = 0; i < n; i++) if (generation[i] > maxGen) maxGen = generation[i];
    const byGen = Array.from({ length: maxGen + 1 }, () => []);
    for (let i = 0; i < n; i++) byGen[generation[i]].push(i);
    const forward = [];
    for (let g = 0; g <= maxGen; g++) {
      const nr = byGen[g].filter((i) => parents[i] >= 0);
      if (nr.length) forward.push({ nodes: Int32Array.from(nr), parents: Int32Array.from(nr.map((i) => parents[i])), coupling: Float64Array.from(nr.map((i) => coupling[i])), contrib: new Float64Array(nr.length * 2) });
    }
    const backward = [];
    for (let g = maxGen; g >= 0; g--) {
      const nodes = byGen[g];
      if (nodes.length) backward.push({ nodes: Int32Array.from(nodes), parents: Int32Array.from(nodes.map((i) => (parents[i] < 0 ? 0 : parents[i]))), coupling: Float64Array.from(nodes.map((i) => (parents[i] < 0 ? 0 : coupling[i]))) });
    }
    return { n, d0, forward, backward, diagRhs: new Float64Array(n * 2) };
  }
  function hinesSolve(hp, compG, rhs) {
    const n = hp.n, dr = hp.diagRhs;
    for (let i = 0; i < n; i++) { dr[2 * i] = hp.d0[i] + compG[i]; dr[2 * i + 1] = rhs[i]; }
    for (const gen of hp.forward) {
      const nodes = gen.nodes, par = gen.parents, coup = gen.coupling, ct = gen.contrib, m = nodes.length;
      for (let k = 0; k < m; k++) { const node = nodes[k]; const ratio = coup[k] / dr[2 * node]; ct[2 * k] = coup[k] * ratio; ct[2 * k + 1] = ratio * dr[2 * node + 1]; }
      for (let k = 0; k < m; k++) { const p = par[k]; dr[2 * p] -= ct[2 * k]; dr[2 * p + 1] -= ct[2 * k + 1]; }
    }
    const voltage = new Float64Array(n);
    for (const gen of hp.backward) {
      const nodes = gen.nodes, par = gen.parents, coup = gen.coupling, m = nodes.length;
      for (let k = 0; k < m; k++) { const node = nodes[k]; voltage[node] = (dr[2 * node + 1] - coup[k] * voltage[par[k]]) / dr[2 * node]; }
    }
    return voltage;
  }

  const RINGN = 48;

  function init(bundle, opts) {
    opts = opts || {};
    const dt = opts.dt || bundle.dt || 0.005;
    const mp = bundle.modelParams || {};
    const cells = bundle.cells, n = cells.length;
    const cableIndices = bundle.cableIndices.slice();
    const isCable = new Set(cableIndices);
    const e_leak = (mp.ELeakMV ?? -60) / 1000, e_exc = (mp.EExcMV ?? -21) / 1000, e_inh = (mp.EInhMV ?? -70) / 1000;
    const e_t4_null = (mp.t4NullReversalMV ?? mp.EInhMV ?? -70) / 1000;
    const external_g_unit = (mp.externalConductanceNS ?? 0.6) * 1e-9;
    const point_leak = (mp.pointLeakNS ?? 1.0) * 1e-9;
    const point_cap = Math.max(1e-15, point_leak * ((mp.pointTauMS ?? 17) / 1000));
    const activity_rest = sigmoid((e_leak + 0.045) / 0.004);

    const hines = {}, cap = {}, leak = {}, outI = {}, size = {}, cable_v = {}, comp_g = {}, comp_rhs = {}, crhs = {};
    for (const i of cableIndices) {
      const c = bundle.cables[i] || bundle.cables[String(i)];
      hines[i] = hinesPrepare(c.parents, c.coupling, c.diag, c.cap, dt);
      cap[i] = Float64Array.from(c.cap); leak[i] = Float64Array.from(c.leak); outI[i] = c.output_i; size[i] = c.size;
      cable_v[i] = new Float64Array(c.size).fill(e_leak);
      comp_g[i] = new Float64Array(c.size); comp_rhs[i] = new Float64Array(c.size); crhs[i] = new Float64Array(c.size);
    }
    // Drop LLPC1<->LLPC1 recurrent synapses (excitatory lateral recurrence within the output sheet):
    // each LLPC1 cell now receives only its feed-forward inputs, with no self-reinforcing recurrence.
    const llpcSet = new Set();
    for (let i = 0; i < n; i++) if (String(cells[i].type || "").toUpperCase().indexOf("LLPC1") >= 0) llpcSet.add(i);
    const E = bundle.events, nEvRaw = E.pre.length, keep = [];
    for (let k = 0; k < nEvRaw; k++) if (!(llpcSet.has(E.pre[k]) && llpcSet.has(E.post[k]))) keep.push(k);
    const nEv = keep.length;
    const pickI = (a) => { const o = new Int32Array(nEv); for (let j = 0; j < nEv; j++) o[j] = a[keep[j]]; return o; };
    const pickF = (a) => { const o = new Float64Array(nEv); for (let j = 0; j < nEv; j++) o[j] = a[keep[j]]; return o; };
    const ev = { pre: pickI(E.pre), post: pickI(E.post), prec: pickI(E.prec), postc: pickI(E.postc), sign: pickF(E.sign), gunit: pickF(E.gunit) };
    const ev_erev = new Float64Array(nEv);
    for (let k = 0; k < nEv; k++) ev_erev[k] = ev.sign[k] < 0 ? e_inh : e_exc;
    const point_post = [];
    for (let k = 0; k < nEv; k++) if (ev.postc[k] < 0) point_post.push([k, ev.post[k]]);
    const cablePre = {}, cablePost = {};
    for (const i of cableIndices) { cablePre[i] = []; cablePost[i] = []; }
    for (let k = 0; k < nEv; k++) {
      if (isCable.has(ev.pre[k]) && ev.prec[k] >= 0) cablePre[ev.pre[k]].push(k);
      if (isCable.has(ev.post[k]) && ev.postc[k] >= 0) cablePost[ev.post[k]].push(k);
    }
    const readout = {};
    for (const i of cableIndices) {
      const comps = new Set([outI[i]]);
      for (let k = 0; k < nEv; k++) { if (ev.pre[k] === i && ev.prec[k] >= 0) comps.add(ev.prec[k]); if (ev.post[k] === i && ev.postc[k] >= 0) comps.add(ev.postc[k]); }
      readout[i] = Int32Array.from([...comps].sort((a, b) => a - b));
    }
    const rel_model = String(mp.releaseModel ?? "rectilinear").toLowerCase();
    const rel_vthresh = (mp.releaseThresholdMV ?? -53) / 1000, rel_gain = (mp.releaseGainPerMV ?? 0.06) * 1000;
    const rel_vhalf = (mp.synapseVHalfMV ?? -45) / 1000, rel_slope = Math.max(1e-6, (mp.synapseSlopeMV ?? 4) / 1000);
    const rel_baseline = sigmoid((e_leak - rel_vhalf) / rel_slope);
    const syn_tau = (mp.synapseTauMS ?? 5) / 1000, syn_alpha = syn_tau <= 0 ? 1 : 1 - Math.exp(-dt / syn_tau);

    // motion (T4/T5) retinotopic array — from drive.ts precomputeMotionArrays
    const M = { idx: [], az: [], el: [], is_t4: [], pdx: [] };
    for (let i = 0; i < n; i++) {
      const c = cells[i], t = String(c.type || "").toUpperCase();
      const motion = t.startsWith("T4") || t.startsWith("T5");
      if (!c.retinotopic || !motion) continue;
      M.idx.push(i); M.az.push(+(c.retinoAzimuth || 0)); M.el.push(+(c.retinoElevation || 0));
      M.is_t4.push(t.startsWith("T4"));
      const d = String(c.preferredDirection || "right"); M.pdx.push(d === "left" ? -1 : d === "right" ? 1 : 0);
    }
    const nullMode = String(mp.t4NullMotionMode || "hyperpolarizing").toLowerCase();
    const D = {
      edgeGain: Math.max(0, mp.opticFlowEdgeGain ?? 8), responseGain: Math.max(0, mp.opticFlowGain ?? 2),
      v50: Math.max(1e-6, mp.opticFlowVelocity50DPS ?? 60), nullLeak: clip(mp.opticFlowNullLeak ?? 0.03, 0, 1),
      nullHyper: (nullMode === "hyperpolarizing" || nullMode === "hyperpolarize" || nullMode === "inhibitory" || nullMode === "signed"),
      nullHyperGain: clip(mp.t4NullHyperpolarizationGain ?? 1, 0, 4),
      edgeWindow: Math.max(dt, (mp.opticFlowEdgeWindowMS ?? 20) / 1000),
    };
    const pointIdx = []; for (let i = 0; i < n; i++) if (!isCable.has(i)) pointIdx.push(i);
    const llpcIdx = [], llpcIdxL = [], llpcIdxR = [];
    for (let i = 0; i < n; i++) if (String(cells[i].type || "").toUpperCase().indexOf("LLPC1") >= 0) {
      llpcIdx.push(i); (String(cells[i].side || "").toUpperCase() === "L" ? llpcIdxL : llpcIdxR).push(i);
    }
    // boost the (real, existing) T4→LLPC1 synapse conductance — compensates for the ~5× T4
    // subsampling so LLPC1 depolarises from its own inputs and can drive its descending targets
    if (opts.llpcBoost && opts.llpcBoost > 1) {
      const s = new Set(llpcIdx);
      for (let k = 0; k < nEv; k++) if (s.has(ev.post[k])) ev.gunit[k] *= opts.llpcBoost;
    }
    // short-term synaptic depression on the RECURRENT (output↔output) synapses: this dense
    // excitatory recurrence otherwise latches (bistable) — with the release model it self-sustains.
    // Depression depletes a sustained synapse so the latch collapses, while transient (figure-driven)
    // responses survive → NOD1/LLPC1 rise AND fall with the stimulus instead of pinning ON.
    const outSetD = new Set([...cableIndices, ...llpcIdx]);
    const ev_isRec = new Uint8Array(nEv);
    for (let k = 0; k < nEv; k++) if (outSetD.has(ev.pre[k]) && outSetD.has(ev.post[k])) ev_isRec[k] = 1;
    const depU = opts.depU ?? 0;                                   // utilisation per release (0 = disabled)
    const dep_rec = depU > 0 ? dt / ((opts.depTauMS ?? 350) / 1000) : 0;  // recovery rate

    const st = {
      bundle, dt, n, cells, cableIndices, isCable,
      e_leak, e_exc, e_inh, e_t4_null, external_g_unit, point_leak, point_cap_dt: point_cap / dt, activity_rest,
      hines, cap, leak, outI, size, cable_v, comp_g, comp_rhs, crhs,
      ev, nEv, ev_erev, point_post, cablePre, cablePost, readout,
      rel_model, rel_vthresh, rel_gain, rel_vhalf, rel_slope, rel_baseline, syn_alpha,
      M, D, pointIdx, llpcIdx, llpcIdxL, llpcIdxR,
      voltage: new Float64Array(n).fill(e_leak), ev_s: new Float64Array(nEv), act: new Float64Array(n),
      ev_dep: new Float64Array(nEv).fill(1), ev_isRec, depOn: depU > 0, dep_U: depU, dep_rec,
      external: new Float64Array(n), point_g: new Float64Array(n), point_rhs: new Float64Array(n),
      ext_lp: new Float64Array(n), input_alpha: 1 - Math.exp(-dt / ((opts.inputLowpassMS ?? 50) / 1000)),   // temporal low-pass on the retinal drive (photoreceptor/lamina integration) — de-jitters the input
      vpre: new Float64Array(nEv), rel: new Float64Array(nEv), g_ev: new Float64Array(nEv), grev: new Float64Array(nEv),
      dF: 0, dG: 0, groundWorldDisp: 0, figAz: 0, figVelRet: 0, groundVelRet: 0,
      inputScale: opts.inputScale ?? 1,
      scene: { figW: opts.figW ?? 16, figH: opts.figH ?? 120, figOn: opts.figOn !== false, scale: opts.scale ?? 10, startAz: opts.startAz ?? 0, seedFig: opts.seedFig ?? 23, seedGnd: opts.seedGnd ?? 17, contrast: opts.contrast ?? 1, groundOn: opts.groundOn !== false,
        gratingOn: opts.gratingOn ?? false, gratingPeriod: opts.gratingPeriod ?? 30, gratingHi: opts.gratingHi ?? 0.75, gratingLo: opts.gratingLo ?? 0 },   // rotating optomotor drum
      ring: { dF: new Float64Array(RINGN), dG: new Float64Array(RINGN), head: 0, len: 0, N: RINGN },
      lpStride: Math.max(1, Math.round(((mp.inputTauMS ?? 20) / 1000 * 0.5) / dt)),
      edgeBack: Math.max(1, Math.round(D.edgeWindow / dt)),
    };
    // cell type/side lookup for readout pooling
    st.cableInfo = cableIndices.map((i) => ({ i, type: String(cells[i].type || "").toUpperCase(), side: String(cells[i].side || "").toUpperCase() }));
    st.wfIdx = st.cableInfo.filter((c) => c.type.indexOf("VCH") >= 0 || c.type.indexOf("DCH") >= 0).map((c) => c.i);   // vCH + DCH wide-field cell indices — for optional ablation
    st.ablateWf = false;
    return st;
  }

  function pushSnap(st) { const r = st.ring; r.dF[r.head] = st.dF; r.dG[r.head] = st.dG; r.head = (r.head + 1) % r.N; if (r.len < r.N) r.len++; }
  function snapBack(st, back) { const r = st.ring; if (r.len === 0) return -1; if (back > r.len - 1) back = r.len - 1; return (r.head - 1 - back + r.N * 2) % r.N; }
  function textureVal(worldAz, scale, seed) { scale = clip(scale, 1, 10); const phase = (hash01(seed, 11, 503) - 0.5) * scale; const idx = Math.floor((worldAz + phase) / scale); return ((idx + seed) & 1) ? 1 : 0; }
  // retinal luminance at azimuth `az` for a past scene snapshot: a WHITE bar (the object/figure,
  // displaced by dF) over the background — either black, or a rotating square-wave grating drum.
  function lumSnap(st, az, el, snapIdx) {
    const r = st.ring, S = st.scene;
    const figAz = wrap180(S.startAz + r.dF[snapIdx]);
    const inside = S.figOn !== false && angDist(az, figAz) <= S.figW / 2 && Math.abs(el) <= S.figH / 2;
    if (inside) return 1;                              // the figure (white bar) is opaque, on top (skipped when hidden)
    if (S.gratingOn) {                                 // rotating optomotor drum: square-wave grating
      const gAz = az - r.dG[snapIdx];                  // stripes rotate with the ground displacement (dG)
      return (Math.floor(gAz / S.gratingPeriod) & 1) ? S.gratingHi : S.gratingLo;
    }
    return 0;                                          // black background
  }
  function filtLum(st, az, el, back) {
    let tot = 0, wsum = 0;
    for (let k = 0; k < 5; k++) { const idx = snapBack(st, back + k * st.lpStride); if (idx < 0) break; const w = LPW[k]; tot += w * lumSnap(st, az, el, idx); wsum += w; }
    return wsum > 0 ? tot / wsum : 0;
  }
  function computeDrive(st) {
    const M = st.M, D = st.D, S = st.scene, ext = st.external, m = M.idx.length;
    for (let j = 0; j < m; j++) {
      const az = M.az[j], el = M.el[j];
      const now = filtLum(st, az, el, 0), before = filtLum(st, az, el, st.edgeBack);
      let evv = M.is_t4[j] ? (now - before) : (before - now);
      evv = Math.min(1, Math.max(0, evv) * D.edgeGain);
      const inside = S.figOn !== false && angDist(az, st.figAz) <= S.figW / 2 && Math.abs(el) <= S.figH / 2;
      const vx = inside ? st.figVelRet : st.groundVelRet;   // retinal velocity of the surface at this az
      const projected = vx * M.pdx[j];
      const anti = -projected, antiFactor = anti / (D.v50 + Math.abs(anti));
      let mf;
      if (projected >= 0) mf = projected / (D.v50 + Math.abs(projected));
      else mf = D.nullHyper ? -D.nullHyperGain * antiFactor : D.nullLeak * antiFactor;
      let drive = D.responseGain * evv * mf * st.inputScale;
      drive = D.nullHyper ? clip(drive, -2, 2) : clip(drive, 0, 2);
      const idx = M.idx[j];
      st.ext_lp[idx] += st.input_alpha * (drive - st.ext_lp[idx]);   // temporal low-pass — smooths the frame-to-frame jitter before it reaches the cells
      ext[idx] = st.ext_lp[idx];
    }
  }

  // one dt of the closed loop. ctrl: {heading, headingVel, figWorldAz, figVelWorld, groundVelWorld}
  function step(st, ctrl) {
    const dt = st.dt, hv = ctrl.headingVel || 0, head = ctrl.heading || 0;
    st.groundWorldDisp += (ctrl.groundVelWorld || 0) * dt;
    st.dF = (ctrl.figWorldAz || 0) - head;              // figure retinal displacement
    st.dG = st.groundWorldDisp - head;                  // ground retinal displacement (self-motion)
    st.figAz = wrap180(st.scene.startAz + st.dF);
    st.figVelRet = (ctrl.figVelWorld || 0) - hv;
    st.groundVelRet = (ctrl.groundVelWorld || 0) - hv;
    pushSnap(st);
    st.external.fill(0);
    computeDrive(st);

    const { n, ev, nEv, cableIndices, isCable, e_leak, e_exc, e_inh, e_t4_null, external_g_unit, point_leak } = st;
    const point_g = st.point_g, point_rhs = st.point_rhs; point_g.fill(0); point_rhs.fill(0);
    for (const i of cableIndices) { st.comp_g[i].fill(0); st.comp_rhs[i].fill(0); }
    for (let i = 0; i < n; i++) {
      const drive = st.external[i]; if (!drive) continue;
      let g, rev;
      if (drive > 0) { g = drive * external_g_unit; rev = e_exc; }
      else { g = -drive * external_g_unit; const ct = String(st.cells[i].type || "").toUpperCase(); rev = (ct.startsWith("T4") || ct.startsWith("T5")) ? e_t4_null : e_inh; }
      if (isCable.has(i)) { const s = st.outI[i]; st.comp_g[i][s] += g; st.comp_rhs[i][s] += g * rev; }
      else { point_g[i] += g; point_rhs[i] += g * rev; }
    }
    if (nEv) {
      const voltage = st.voltage, vpre = st.vpre;
      for (let k = 0; k < nEv; k++) vpre[k] = voltage[ev.pre[k]];
      for (const ci of cableIndices) { const g = st.cablePre[ci], cv = st.cable_v[ci]; for (let mm = 0; mm < g.length; mm++) { const kk = g[mm]; vpre[kk] = cv[ev.prec[kk]]; } }
      const rel = st.rel;
      if (st.rel_model === "sigmoid") { for (let kk = 0; kk < nEv; kk++) { const sg = sigmoid((vpre[kk] - st.rel_vhalf) / st.rel_slope); rel[kk] = clip((sg - st.rel_baseline) / Math.max(1e-9, 1 - st.rel_baseline), 0, 1); } }
      else { for (let kk = 0; kk < nEv; kk++) rel[kk] = clip(st.rel_gain * (vpre[kk] - st.rel_vthresh), 0, 1); }
      const ev_s = st.ev_s, g_ev = st.g_ev, grev = st.grev, ev_erev = st.ev_erev;
      if (st.depOn) {
        const ev_dep = st.ev_dep, isRec = st.ev_isRec, dU = st.dep_U, dR = st.dep_rec;
        for (let kk = 0; kk < nEv; kk++) {
          ev_s[kk] += st.syn_alpha * (rel[kk] - ev_s[kk]);
          if (isRec[kk]) { let d = ev_dep[kk] + (1 - ev_dep[kk]) * dR - ev_dep[kk] * ev_s[kk] * dU; ev_dep[kk] = d < 0.02 ? 0.02 : d; g_ev[kk] = ev.gunit[kk] * ev_s[kk] * ev_dep[kk]; }
          else g_ev[kk] = ev.gunit[kk] * ev_s[kk];
          grev[kk] = g_ev[kk] * ev_erev[kk];
        }
      } else {
        for (let kk = 0; kk < nEv; kk++) { ev_s[kk] += st.syn_alpha * (rel[kk] - ev_s[kk]); g_ev[kk] = ev.gunit[kk] * ev_s[kk]; grev[kk] = g_ev[kk] * ev_erev[kk]; }
      }
      for (let mm = 0; mm < st.point_post.length; mm++) { const kk = st.point_post[mm][0], dst = st.point_post[mm][1]; point_g[dst] += g_ev[kk]; point_rhs[dst] += grev[kk]; }
      for (const ci of cableIndices) { const g = st.cablePost[ci], cg = st.comp_g[ci], cr = st.comp_rhs[ci]; for (let mm = 0; mm < g.length; mm++) { const kk = g[mm], pc = ev.postc[kk]; cg[pc] += g_ev[kk]; cr[pc] += grev[kk]; } }
    }
    for (const i of cableIndices) {
      const v_old = st.cable_v[i], sz = st.size[i], capi = st.cap[i], leaki = st.leak[i], cr = st.comp_rhs[i], rhs = st.crhs[i];
      for (let c = 0; c < sz; c++) rhs[c] = (capi[c] / dt) * v_old[c] + leaki[c] * e_leak + cr[c];
      const v_new = hinesSolve(st.hines[i], st.comp_g[i], rhs);
      st.cable_v[i] = v_new;
      const ridx = st.readout[i]; let sum = 0; for (let mm = 0; mm < ridx.length; mm++) sum += v_new[ridx[mm]]; st.voltage[i] = sum / ridx.length;
    }
    if (st.ablateWf) { const el0 = st.e_leak; for (let m = 0; m < st.wfIdx.length; m++) { const i = st.wfIdx[m]; st.cable_v[i].fill(el0); st.voltage[i] = el0; } }  // vCH+DCH ablation: clamp to rest → release nothing downstream, traces go flat
    const cap_dt = st.point_cap_dt;
    for (let mm = 0; mm < st.pointIdx.length; mm++) { const i = st.pointIdx[mm]; const left = cap_dt + point_leak + point_g[i]; const rhs = cap_dt * st.voltage[i] + point_leak * e_leak + point_rhs[i]; st.voltage[i] = rhs / Math.max(left, 1e-18); }
    const act = st.act, arest = st.activity_rest;
    for (let i = 0; i < n; i++) { const asig = sigmoid((st.voltage[i] + 0.045) / 0.004); act[i] = clip((asig - arest) / Math.max(1e-9, 1 - arest), 0, 1); }
  }

  // pool cable activity by type/side + hemifield T4 (wide-field) — for steering
  function readout(st) {
    const o = { nod1L: 0, nod1R: 0, vchL: 0, vchR: 0, dchL: 0, dchR: 0, t4L: 0, t4R: 0 };
    const mv = { nod1L: 0, nod1R: 0, vchL: 0, vchR: 0, dchL: 0, dchR: 0 };  // mean membrane potential (mV) per group
    const cnt = { nod1L: 0, nod1R: 0, vchL: 0, vchR: 0, dchL: 0, dchR: 0 };
    for (const ci of st.cableInfo) {
      const a = st.act[ci.i], v = st.voltage[ci.i] * 1000; const L = ci.side === "L";
      let key = null;
      if (ci.type.indexOf("NOD1") >= 0) key = L ? "nod1L" : "nod1R";
      else if (ci.type.indexOf("VCH") >= 0) key = L ? "vchL" : "vchR";
      else if (ci.type.indexOf("DCH") >= 0) key = L ? "dchL" : "dchR";
      if (key) { o[key] += a; mv[key] += v; cnt[key]++; }
    }
    for (const k in cnt) if (cnt[k]) { o[k] /= cnt[k]; mv[k] /= cnt[k]; }
    let tl = 0, tr = 0, nl = 0, nr = 0, dl = 0, dr = 0;
    const M = st.M;
    for (let j = 0; j < M.idx.length; j++) {
      const a = st.act[M.idx[j]]; if (M.az[j] < 0) { tl += a; nl++; } else { tr += a; nr++; }
      const d = st.external[M.idx[j]]; if (d > 0) { if (M.az[j] < 0) dl += d; else dr += d; }  // pooled optic-flow drive per hemifield (directional / lateralised steering signal)
    }
    o.t4L = nl ? tl / nl : 0; o.t4R = nr ? tr / nr : 0;
    o.driveL = dl; o.driveR = dr;
    let la = 0; for (let m = 0; m < st.llpcIdx.length; m++) la += st.act[st.llpcIdx[m]];
    o.llpcAct = st.llpcIdx.length ? la / st.llpcIdx.length : 0;
    let ll = 0; for (let m = 0; m < st.llpcIdxL.length; m++) ll += st.act[st.llpcIdxL[m]];
    let lr = 0; for (let m = 0; m < st.llpcIdxR.length; m++) lr += st.act[st.llpcIdxR[m]];
    o.llpcL = st.llpcIdxL.length ? ll / st.llpcIdxL.length : 0;
    o.llpcR = st.llpcIdxR.length ? lr / st.llpcIdxR.length : 0;
    o.mv = mv;
    return o;
  }

  // current retinal luminance at azimuth `az` (for display) — uses the latest scene snapshot
  function lumNow(st, az, el) { const idx = snapBack(st, 0); return idx < 0 ? 0.5 : lumSnap(st, az, el || 0, idx); }

  const API = { init, step, readout, lumNow, hinesPrepare, hinesSolve, _internal: { lumSnap, filtLum, computeDrive } };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.FGModel = API;
})(typeof window !== "undefined" ? window : globalThis);
