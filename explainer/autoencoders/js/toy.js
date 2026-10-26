// 1.2 — a 2-D → 1 → 2-D autoencoder trained live (linear or tanh MLP), next to PCA.

import { h, fmt, colors, rgbCss, sequential, fitCanvas, widthOf, rng, segToggle, ctl, visibility, showTip, hideTip, clamp } from "./util.js";
import { Dense, Net, mseGrad } from "./nn.js";
import { legendRow } from "./charts.js";
import { T } from "./i18n.js";

const N = 300;
const MAX_STEPS = { linear: 3000, mlp: 6000 };

function makeData(preset, seed = 7) {
  const r = rng(seed);
  const X = new Float32Array(N * 2);
  const t = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    let x;
    let y;
    if (preset === "cloud") {
      const a = r.normal();
      const b = r.normal();
      x = 0.72 * a;
      y = 0.4 * a + 0.2 * b;
      t[i] = a;
    } else if (preset === "arc") {
      const th = -2.3 + 4.6 * r.uniform();
      x = Math.cos(th) + 0.04 * r.normal();
      y = Math.sin(th) + 0.04 * r.normal();
      t[i] = th;
    } else {
      const u = -1.6 + 3.2 * r.uniform();
      x = u;
      y = 0.55 * Math.sin(2.2 * u) + 0.04 * r.normal();
      t[i] = u;
    }
    X[2 * i] = x;
    X[2 * i + 1] = y;
  }
  let mx = 0;
  let my = 0;
  for (let i = 0; i < N; i++) (mx += X[2 * i]), (my += X[2 * i + 1]);
  mx /= N;
  my /= N;
  for (let i = 0; i < N; i++) (X[2 * i] -= mx), (X[2 * i + 1] -= my);
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of t) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
  const tn = t.map((v) => (v - lo) / (hi - lo));
  // best 1-D linear reconstruction (PCA): smallest eigenvalue of the covariance, averaged over the two coordinates
  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (let i = 0; i < N; i++) {
    const a = X[2 * i];
    const b = X[2 * i + 1];
    sxx += a * a;
    syy += b * b;
    sxy += a * b;
  }
  (sxx /= N), (syy /= N), (sxy /= N);
  const tr = sxx + syy;
  const lmin = (tr - Math.sqrt(Math.max(0, tr * tr - 4 * (sxx * syy - sxy * sxy)))) / 2;
  return { X, t: tn, pcaMse: lmin / 2 };
}

function makeModel(kind, seed) {
  if (kind === "linear") {
    return { enc: new Net([new Dense(2, 1, "linear", seed)]), dec: new Net([new Dense(1, 2, "linear", seed + 1)]), lr: 0.02 };
  }
  return {
    enc: new Net([new Dense(2, 24, "tanh", seed), new Dense(24, 24, "tanh", seed + 1), new Dense(24, 1, "linear", seed + 2)]),
    dec: new Net([new Dense(1, 24, "tanh", seed + 3), new Dense(24, 24, "tanh", seed + 4), new Dense(24, 2, "linear", seed + 5)]),
    lr: 0.005,
  };
}

export function initToy(root) {
  const st = { preset: "arc", kind: "mlp", seed: 1, steps: 0, running: false, loss: NaN, hover: -1, zProbe: null };
  let data = makeData(st.preset);
  let model = makeModel(st.kind, st.seed);
  let Z = new Float32Array(N);
  let R = new Float32Array(N * 2);
  let started = false;

  const plot = h("canvas", { class: "plot crosshair", "aria-label": T.toyPlotLabel });
  const strip = h("canvas", { class: "plot crosshair", "aria-label": T.toyStripLabel });
  const runBtn = h("button", { class: "btn primary", type: "button" }, T.train);
  const resetBtn = h("button", { class: "btn", type: "button" }, T.reset);
  const stats = h("div", { class: "stats" });
  const isVisible = visibility(root);

  runBtn.addEventListener("click", () => (st.running ? stop() : start()));
  resetBtn.addEventListener("click", () => reset());

  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.data, segToggle([{ value: "cloud", label: T.cloud }, { value: "arc", label: T.arc }, { value: "wave", label: T.wave }], st.preset, (v) => {
        st.preset = v;
        data = makeData(v);
        reset();
      })),
      ctl(T.model, segToggle([{ value: "linear", label: T.linearAE }, { value: "mlp", label: T.nonlinearAE }], st.kind, (v) => {
        st.kind = v;
        reset();
      })),
      runBtn, resetBtn),
    legendRow([["s1", T.decoderCurve], ["s2", T.reconstructions], ["muted", T.toyPoints, "dotkey"]]),
    plot,
    h("div", { class: "sub" }, T.toyStripTitle),
    strip,
    stats,
  );

  function forwardAll() {
    Z = model.enc.forward(data.X, N);
    R = model.dec.forward(Z, N);
    st.loss = mseGrad(R, data.X).loss;
  }

  function trainStep() {
    const z = model.enc.forward(data.X, N);
    const r = model.dec.forward(z, N);
    const { loss, grad } = mseGrad(r, data.X);
    model.enc.zeroGrad();
    model.dec.zeroGrad();
    model.enc.backward(model.dec.backward(grad));
    model.enc.step(model.lr);
    model.dec.step(model.lr);
    st.steps++;
    return loss;
  }

  function loop() {
    if (!st.running) return;
    if (isVisible()) {
      for (let i = 0; i < 12 && st.steps < MAX_STEPS[st.kind]; i++) trainStep();
      forwardAll();
      draw();
      if (st.steps >= MAX_STEPS[st.kind]) return stop();
    }
    requestAnimationFrame(loop);
  }
  function start() {
    if (st.steps >= MAX_STEPS[st.kind]) reset(false);
    st.running = true;
    runBtn.textContent = T.pause;
    requestAnimationFrame(loop);
  }
  function stop() {
    st.running = false;
    runBtn.textContent = st.steps >= MAX_STEPS[st.kind] ? T.trainAgain : T.train;
    draw();
  }
  function reset(redraw = true) {
    st.seed++;
    model = makeModel(st.kind, st.seed);
    st.steps = 0;
    st.hover = -1;
    forwardAll();
    if (redraw) draw();
    if (st.running) runBtn.textContent = T.pause;
    else runBtn.textContent = T.train;
  }

  // equal-aspect mapping from data space to the canvas
  let geo = null;
  function layout() {
    const W = widthOf(root);
    const H = Math.round(clamp(W * 0.55, 260, 380));
    const ctx = fitCanvas(plot, W, H);
    plot.style.width = `${W}px`;
    const span = 1.75;
    const scale = Math.min(W, H) / (2 * span) * 0.98;
    geo = { W, H, ctx, scale, cx: W / 2, cy: H / 2 };
    const sctx = fitCanvas(strip, W, 44);
    strip.style.width = `${W}px`;
    geo.sctx = sctx;
  }
  const px = (x) => geo.cx + x * geo.scale;
  const py = (y) => geo.cy - y * geo.scale;

  function zRange() {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of Z) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
    if (!(hi > lo)) (lo -= 1), (hi += 1);
    return [lo, hi];
  }

  function draw() {
    if (!geo) layout();
    const c = colors();
    const { ctx, W, H } = geo;
    ctx.clearRect(0, 0, W, H);
    // axes
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, geo.cy);
    ctx.lineTo(W, geo.cy);
    ctx.moveTo(geo.cx, 0);
    ctx.lineTo(geo.cx, H);
    ctx.stroke();
    // residuals
    ctx.strokeStyle = c.lineStrong;
    ctx.beginPath();
    for (let i = 0; i < N; i += 2) {
      ctx.moveTo(px(data.X[2 * i]), py(data.X[2 * i + 1]));
      ctx.lineTo(px(R[2 * i]), py(R[2 * i + 1]));
    }
    ctx.stroke();
    // decoder curve
    const [zlo, zhi] = zRange();
    const pad = 0.25 * (zhi - zlo);
    const n = 160;
    const zs = Float32Array.from({ length: n }, (_, i) => zlo - pad + ((zhi - zlo + 2 * pad) * i) / (n - 1));
    const curve = model.dec.forward(zs, n);
    ctx.strokeStyle = c.s1;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) ctx[i ? "lineTo" : "moveTo"](px(curve[2 * i]), py(curve[2 * i + 1]));
    ctx.stroke();
    // data points (sequential color = position along the shape)
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = rgbCss(sequential(0.2 + 0.8 * data.t[i]));
      ctx.beginPath();
      ctx.arc(px(data.X[2 * i]), py(data.X[2 * i + 1]), 3.2, 0, 2 * Math.PI);
      ctx.fill();
    }
    // reconstructions
    ctx.fillStyle = c.s2;
    for (let i = 0; i < N; i++) ctx.fillRect(px(R[2 * i]) - 1.5, py(R[2 * i + 1]) - 1.5, 3, 3);
    // hover
    if (st.hover >= 0) {
      const i = st.hover;
      ctx.strokeStyle = c.inkCss;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px(data.X[2 * i]), py(data.X[2 * i + 1]));
      ctx.lineTo(px(R[2 * i]), py(R[2 * i + 1]));
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px(data.X[2 * i]), py(data.X[2 * i + 1]), 6, 0, 2 * Math.PI);
      ctx.stroke();
    }
    if (st.zProbe != null) {
      const p = model.dec.forward(Float32Array.of(st.zProbe), 1);
      ctx.fillStyle = c.inkCss;
      ctx.beginPath();
      ctx.arc(px(p[0]), py(p[1]), 6, 0, 2 * Math.PI);
      ctx.fill();
    }
    drawStrip(zlo - pad, zhi + pad);
    renderStats();
  }

  function drawStrip(lo, hi) {
    const c = colors();
    const ctx = geo.sctx;
    const W = geo.W;
    ctx.clearRect(0, 0, W, 44);
    const zx = (z) => 12 + ((z - lo) / (hi - lo)) * (W - 24);
    ctx.strokeStyle = c.lineStrong;
    ctx.beginPath();
    ctx.moveTo(12, 22);
    ctx.lineTo(W - 12, 22);
    ctx.stroke();
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = rgbCss(sequential(0.2 + 0.8 * data.t[i]));
      ctx.fillRect(zx(Z[i]) - 1, 10, 2.5, 24);
    }
    if (st.hover >= 0) {
      ctx.fillStyle = c.inkCss;
      ctx.fillRect(zx(Z[st.hover]) - 1.5, 4, 3, 36);
    }
    if (st.zProbe != null) {
      ctx.fillStyle = c.inkCss;
      ctx.beginPath();
      ctx.arc(zx(st.zProbe), 22, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
    geo.strip = { lo, hi, zx };
  }

  function renderStats() {
    stats.replaceChildren(
      h("span", null, h("b", null, st.steps.toLocaleString()), T.steps),
      h("span", null, h("b", null, fmt(st.loss, 4)), T.aeMse),
      h("span", null, h("b", null, fmt(data.pcaMse, 4)), T.pcaMse),
      h("span", { class: "stat-note" }, st.kind === "linear" ? T.toyNoteLinear : T.toyNoteMlp),
    );
  }

  plot.addEventListener("pointermove", (ev) => {
    const r = plot.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    const my = ev.clientY - r.top;
    let best = -1;
    let bd = 14 * 14;
    for (let i = 0; i < N; i++) {
      const d = (px(data.X[2 * i]) - mx) ** 2 + (py(data.X[2 * i + 1]) - my) ** 2;
      if (d < bd) (bd = d), (best = i);
    }
    if (best !== st.hover) {
      st.hover = best;
      if (!st.running) draw();
    }
    if (best >= 0) {
      showTip(`${T.point} <span class="mono">(${fmt(data.X[2 * best], 2)}, ${fmt(data.X[2 * best + 1], 2)})</span><br>${T.code} z = <span class="mono">${fmt(Z[best], 3)}</span><br>${T.rebuilt} <span class="mono">(${fmt(R[2 * best], 2)}, ${fmt(R[2 * best + 1], 2)})</span>`, ev);
    } else hideTip();
  });
  plot.addEventListener("pointerleave", () => {
    st.hover = -1;
    hideTip();
    if (!st.running) draw();
  });
  strip.addEventListener("pointermove", (ev) => {
    if (!geo?.strip) return;
    const r = strip.getBoundingClientRect();
    const { lo, hi } = geo.strip;
    st.zProbe = lo + (clamp(ev.clientX - r.left, 12, geo.W - 12) - 12) / (geo.W - 24) * (hi - lo);
    showTip(`${T.decodeZ} z = <span class="mono">${fmt(st.zProbe, 2)}</span>`, ev);
    if (!st.running) draw();
  });
  strip.addEventListener("pointerleave", () => {
    st.zProbe = null;
    hideTip();
    if (!st.running) draw();
  });

  forwardAll();
  layout();
  draw();
  // start training the first time the demo scrolls into view
  new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting) && !started) {
      started = true;
      obs.disconnect();
      start();
    }
  }, { threshold: 0.35 }).observe(root);

  return {
    redraw() {
      layout();
      draw();
    },
  };
}
