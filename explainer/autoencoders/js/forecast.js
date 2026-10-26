// Parts 4.3, 5 and 6 — NLinear's weights on ETTh1, the forecasting autoencoder (W = D·E), the order test,
// a live training lab on synthetic data, and the conditional VAE's fan chart.

import { h, fmt, pct, clamp, f32, rng, segToggle, ctl, showTip, hideTip, robustScale, divLegend, widthOf, visibility, LANG, LANG_LOCALE } from "./util.js";
import { Dense, Net, mseGrad, ridge } from "./nn.js";
import { lineChart, heatmap, spark, legendRow, range } from "./charts.js";
import { T } from "./i18n.js";

const KS = [1, 2, 4, 8, 16, 32];

// ───────────── data & models ─────────────
export function prepareForecast(j) {
  const { L, T: H } = j;
  const n = j.series.n;
  const S = f32(j.series.data);
  const series = Array.from({ length: j.vars.length }, (_, v) => S.subarray(v * n, (v + 1) * n));
  const ae = {};
  for (const [k, m] of Object.entries(j.ae)) ae[k] = { k: +k, E: f32(m.E), bE: f32(m.bE), D: f32(m.D), bD: f32(m.bD), test: m.test, shuffled: m.shuffled, halfExchanged: m.halfExchanged };
  const c = j.cvae;
  const cvae = {
    E: f32(c.E), bE: f32(c.bE), D: f32(c.D), bD: f32(c.bD), H: f32(c.H), bH: f32(c.bH), P: f32(c.P), bP: f32(c.bP), U: f32(c.U),
    sigma: Float32Array.from(f32(c.logSigma), Math.exp), k: c.E.shape[0], hidden: c.H.shape[0], kz: c.U.shape[1], test: c.test, baseline: c.baseline,
  };
  return {
    L, T: H, vars: j.vars, series, n, nWindows: n - L - H + 1, start: new Date(j.series.start.replace(" ", "T") + "Z"),
    nlinear: { W: f32(j.nlinear.W), b: f32(j.nlinear.b), test: j.nlinear.test, shuffled: j.nlinear.shuffled, halfExchanged: j.nlinear.halfExchanged },
    repeat: j.repeat, ae, cvae,
  };
}

const window_ = (F, v, i) => F.series[v].subarray(i, i + F.L);
const target = (F, v, i) => F.series[v].subarray(i + F.L, i + F.L + F.T);

export function nlinear(F, x) {
  const { W, b } = F.nlinear;
  const L = x.length;
  const last = x[L - 1];
  const y = new Float32Array(F.T);
  for (let t = 0; t < F.T; t++) {
    let acc = b[t] + last;
    const o = t * L;
    for (let i = 0; i < L; i++) acc += W[o + i] * (x[i] - last);
    y[t] = acc;
  }
  return y;
}

export function aeForecast(F, k, x) {
  const m = F.ae[k];
  const L = x.length;
  const last = x[L - 1];
  const z = new Float32Array(k);
  for (let j = 0; j < k; j++) {
    let acc = m.bE[j];
    for (let i = 0; i < L; i++) acc += m.E[j * L + i] * (x[i] - last);
    z[j] = acc;
  }
  const y = new Float32Array(F.T);
  for (let t = 0; t < F.T; t++) {
    let acc = m.bD[t] + last;
    for (let j = 0; j < k; j++) acc += m.D[t * k + j] * z[j];
    y[t] = acc;
  }
  return { y, z, last };
}

export function cvaeParts(F, x) {
  const c = F.cvae;
  const L = x.length;
  const last = x[L - 1];
  const xn = Float32Array.from(x, (v) => v - last);
  const code = new Float32Array(c.k);
  for (let j = 0; j < c.k; j++) {
    let acc = c.bE[j];
    for (let i = 0; i < L; i++) acc += c.E[j * L + i] * xn[i];
    code[j] = acc;
  }
  const base = new Float32Array(F.T);
  for (let t = 0; t < F.T; t++) {
    let acc = c.bD[t] + last;
    for (let j = 0; j < c.k; j++) acc += c.D[t * c.k + j] * code[j];
    base[t] = acc;
  }
  const hid = new Float32Array(c.hidden);
  for (let u = 0; u < c.hidden; u++) {
    let acc = c.bH[u];
    for (let i = 0; i < L; i++) acc += c.H[u * L + i] * xn[i];
    hid[u] = Math.tanh(acc);
  }
  const mu = new Float32Array(c.kz);
  const logvar = new Float32Array(c.kz);
  for (let o = 0; o < 2 * c.kz; o++) {
    let acc = c.bP[o];
    for (let u = 0; u < c.hidden; u++) acc += c.P[o * c.hidden + u] * hid[u];
    if (o < c.kz) mu[o] = acc;
    else logvar[o - c.kz] = acc;
  }
  return { base, mu, logvar };
}

// mean of the prior, decoded: the CVAE's point forecast
export function cvaeMean(F, x) {
  const { base, mu } = cvaeParts(F, x);
  const c = F.cvae;
  return base.map((v, t) => {
    let acc = v;
    for (let j = 0; j < c.kz; j++) acc += c.U[t * c.kz + j] * mu[j];
    return acc;
  });
}

const mse = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
};

function dateLabel(F, i) {
  const d = new Date(F.start.getTime() + (i + F.L) * 3600e3);
  return new Intl.DateTimeFormat(LANG_LOCALE, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }).format(d);
}

function varSelect(F, current, onPick) {
  const sel = h("select", { "aria-label": T.variable });
  F.vars.forEach((v, i) => sel.append(h("option", { value: i, selected: i === current ? true : null }, `${v} · ${T.varNames[v] || ""}`)));
  sel.addEventListener("change", () => onPick(+sel.value));
  return sel;
}

function windowSlider(F, current, onPick) {
  const input = h("input", { type: "range", min: 0, max: F.nWindows - 1, step: 1, value: current, "aria-label": T.forecastTime });
  const out = h("output");
  const set = (i) => (out.textContent = dateLabel(F, i));
  input.addEventListener("input", () => {
    set(+input.value);
    onPick(+input.value);
  });
  set(current);
  return h("div", { class: "ctl wide" }, h("span", { class: "lbl" }, T.forecastTime), input, out);
}

// ───────────── sections ─────────────
export function initForecast(F, roots) {
  const shared = { k: 8, v: 6, i: 1000, listeners: [] };
  shared.set = (patch) => {
    Object.assign(shared, patch);
    shared.listeners.forEach((f) => f());
  };
  const parts = [
    linearSection(F, roots.linear),
    lowRankSection(F, roots.lowrank, shared),
    windowSection(F, roots.window, shared),
    orderSection(F, roots.order, shared),
    labSection(roots.lab),
    fanSection(F, roots.fan),
  ];
  const nl = fmt(F.nlinear.test.mse, 3);
  document.getElementById("nl-mse")?.replaceChildren(LANG === "ru" ? nl.replace(".", ",") : nl);
  return { redraw: () => parts.forEach((p) => p?.redraw()) };
}

function heatBlock(M, rows, cols, scale, { yTop, yBottom, xLeft, xRight, onHover, onClick, marker, height = 230 }) {
  const canvas = heatmap(M, rows, cols, scale, { height, onHover, onClick });
  const box = h("div", { style: { position: "relative" } }, canvas);
  if (marker != null) {
    box.append(h("div", { style: { position: "absolute", left: "-8px", width: "5px", height: `${Math.max(3, height / rows)}px`, top: `${((marker + 0.5) / rows) * height - Math.max(3, height / rows) / 2}px`, background: "var(--accent)", borderRadius: "2px" } }));
  }
  return h("div", { class: "heat-wrap" },
    h("div", { class: "heat-y", style: { height: `${height}px` } }, h("span", null, yTop), h("span", null, yBottom)),
    box,
    h("div", { class: "heat-x" }, h("span", null, xLeft), h("span", null, xRight)));
}

// 4.3
function linearSection(F, root) {
  const st = { row: 23 };
  const { L } = F;
  const { W } = F.nlinear;
  const scale = robustScale(W, 0.995);
  const heatBox = h("div");
  const profBox = h("div");
  function render() {
    heatBox.replaceChildren(
      h("div", { class: "viz-head", style: { display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px" } },
        h("div", { class: "panel-title" }, T.nlinearW), divLegend(scale, 3)),
      heatBlock(W, F.T, L, scale, {
        yTop: T.hoursAhead(1), yBottom: T.hoursAhead(F.T), xLeft: T.hoursAgo(L), xRight: T.hoursAgo(1), marker: st.row,
        onHover: (r, c, ev) => showTip(T.weightTip(r + 1, L - c, W[r * L + c]), ev),
        onClick: (r) => ((st.row = r), render()),
      }));
    const xs = range(-L, 0);
    const ys = Array.from(W.subarray(st.row * L, st.row * L + L));
    const h0 = st.row + 1;
    const marks = xs.filter((x) => (h0 - x) % 24 === 0).map((x) => ({ x, cls: "now" }));
    profBox.replaceChildren(
      h("div", { class: "panel-title" }, T.rowProfile(h0)),
      lineChart({
        width: widthOf(root), height: 170, series: [{ xs, ys, cls: "s1", label: T.weight }], marks, xLabel: T.pastHourAxis,
        tipTitle: (x) => T.hoursAgo(-x), tipFmt: (v) => fmt(v, 4), yFmt: (v) => fmt(v, 2),
      }),
      h("p", { class: "sub" }, T.rowNote(h0)));
  }
  root.replaceChildren(heatBox, profBox);
  render();
  return { redraw: render };
}

// 5.1
function lowRankSection(F, root, shared) {
  const st = { k: shared.k };
  const { L } = F;
  const scale = robustScale(F.nlinear.W, 0.995);
  const heatBox = h("div");
  const chartBox = h("div");
  const statBox = h("div");
  function product(k) {
    const m = F.ae[k];
    const M = new Float32Array(F.T * L);
    for (let t = 0; t < F.T; t++) for (let j = 0; j < k; j++) {
      const d = m.D[t * k + j];
      for (let i = 0; i < L; i++) M[t * L + i] += d * m.E[j * L + i];
    }
    return M;
  }
  function render() {
    const full = st.k === "full";
    const M = full ? F.nlinear.W : product(st.k);
    heatBox.replaceChildren(
      h("div", { style: { display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px" } },
        h("div", { class: "panel-title", html: full ? T.nlinearW : T.productDE(st.k) }), divLegend(scale, 3)),
      heatBlock(M, F.T, L, scale, { yTop: T.hoursAhead(1), yBottom: T.hoursAhead(F.T), xLeft: T.hoursAgo(L), xRight: T.hoursAgo(1), height: 200,
        onHover: (r, c, ev) => showTip(T.weightTip(r + 1, L - c, M[r * L + c]), ev) }));
    const xs = KS.map((k) => Math.log2(k));
    const ys = KS.map((k) => F.ae[k].test.mse);
    chartBox.replaceChildren(
      h("div", { class: "panel-title" }, T.mseVsK),
      lineChart({
        width: Math.min(widthOf(root), 520), height: 190, y0: 0.35, y1: 0.66, xTicks: xs, xFmt: (v) => String(2 ** v),
        series: [{ xs, ys, cls: "s1", label: T.aeTestMse }], hlines: [{ y: F.nlinear.test.mse, label: `NLinear ${fmt(F.nlinear.test.mse, 3)}` }],
        marks: full ? [] : [{ x: Math.log2(st.k), cls: "hair" }], xLabel: T.bottleneckK, tipTitle: (x) => `k = ${2 ** x}`,
      }));
    const params = full ? L * F.T + F.T : st.k * (L + F.T) + st.k + F.T;
    const err = full ? F.nlinear.test.mse : F.ae[st.k].test.mse;
    statBox.replaceChildren(h("div", { class: "stats" },
      h("span", null, h("b", null, params.toLocaleString(LANG_LOCALE)), T.parameters),
      h("span", null, h("b", null, fmt(err, 3)), T.testMse),
      h("span", { class: "stat-note" }, T.repeatNote(F.repeat.test.mse))));
  }
  root.replaceChildren(
    h("div", { class: "controls" }, ctl("k", segToggle([...KS.map((k) => ({ value: k, label: String(k) })), { value: "full", label: T.fullNLinear }], st.k, (v) => {
      st.k = v;
      if (v !== "full") shared.set({ k: v });
      render();
    }))),
    heatBox, h("div", { class: "viz-row", style: { marginTop: "10px" } }, h("div", { class: "grow" }, chartBox), h("div", { class: "side" }, statBox)));
  render();
  return { redraw: render };
}

// 5.2
function windowSection(F, root, shared) {
  const st = { showAll: false, hl: -1 };
  const chartBox = h("div");
  const minis = h("div", { class: "minis" });
  const stats = h("div", { class: "stats" });
  const kToggle = h("div");
  function renderK() {
    kToggle.replaceChildren(ctl("k", segToggle(KS.map((k) => ({ value: k, label: String(k) })), shared.k, (v) => shared.set({ k: v }))));
  }
  function render() {
    const { v, i, k } = shared;
    const x = window_(F, v, i);
    const y = target(F, v, i);
    const { y: ya, z, last } = aeForecast(F, k, x);
    const yn = nlinear(F, x);
    const yr = new Float32Array(F.T).fill(last);
    const span = st.showAll ? F.L : 168;
    const xsPast = range(-span, 0);
    const xsF = range(0, F.T);
    const m = F.ae[k];
    const pieces = Array.from({ length: k }, (_, j) => ({ j, c: Float32Array.from({ length: F.T }, (_, t) => z[j] * m.D[t * k + j]) }))
      .map((p) => ({ ...p, norm: Math.sqrt(p.c.reduce((s, q) => s + q * q, 0)) }))
      .sort((a, b) => b.norm - a.norm);
    const series = [
      { xs: xsPast, ys: Array.from(x.subarray(F.L - span)), cls: "past", label: T.past, thin: true },
      { xs: xsF, ys: Array.from(y), cls: "ink", label: T.actual },
      { xs: xsF, ys: Array.from(yr), cls: "s3", label: "Repeat", thin: true },
      { xs: xsF, ys: Array.from(yn), cls: "s2", label: "NLinear" },
      { xs: xsF, ys: Array.from(ya), cls: "s1", label: T.aeK(k) },
    ];
    if (st.hl >= 0) {
      const p = pieces.find((q) => q.j === st.hl);
      if (p) series.push({ xs: xsF, ys: Array.from(p.c, (q) => q + last), cls: "s1", thin: true, opacity: 0.55, label: T.piece(p.j + 1), noTip: true });
    }
    chartBox.replaceChildren(
      legendRow([["muted", T.past], ["ink", T.actual], ["s1", T.aeK(k)], ["s2", "NLinear"], ["s3", "Repeat"]]),
      lineChart({ width: widthOf(root), height: 240, series, marks: [{ x: 0 }], xLabel: T.hoursFromNow, tipTitle: (t) => T.hourRel(t) }));
    stats.replaceChildren(
      h("span", null, h("b", null, fmt(mse(ya, y), 3)), T.windowMse(T.aeK(k))),
      h("span", null, h("b", null, fmt(mse(yn, y), 3)), T.windowMse("NLinear")),
      h("span", null, h("b", null, fmt(mse(yr, y), 3)), T.windowMse("Repeat")),
      h("span", { class: "stat-note" }, T.lastValue(fmt(last, 2))));
    const scale = Math.max(1e-6, ...pieces.map((p) => Math.max(...Array.from(p.c, Math.abs))));
    minis.replaceChildren(...pieces.slice(0, 8).map((p) => {
      const card = h("div", { class: `mini${st.hl === p.j ? " on" : ""}` },
        h("div", { class: "ttl" }, h("span", null, h("b", null, `z${sub(p.j + 1)}`), ` = ${fmt(z[p.j], 2)}`), h("span", null, `d${sub(p.j + 1)}`)),
        spark(p.c, { scale, cls: "s1" }));
      card.addEventListener("pointerenter", () => ((st.hl = p.j), render()));
      card.addEventListener("pointerleave", () => ((st.hl = -1), render()));
      return card;
    }));
  }
  const allToggle = h("input", { type: "checkbox" });
  allToggle.addEventListener("change", () => ((st.showAll = allToggle.checked), render()));
  root.replaceChildren(
    h("div", { class: "controls" }, ctl(T.variable, varSelect(F, shared.v, (v) => shared.set({ v }))), kToggle,
      h("label", { class: "toggle" }, allToggle, T.showAllInput)),
    h("div", { class: "controls" }, windowSlider(F, shared.i, (i) => shared.set({ i }))),
    chartBox, stats,
    h("div", { class: "panel-title", style: { marginTop: "14px" }, html: T.piecesTitle }), minis);
  shared.listeners.push(() => {
    renderK();
    render();
  });
  renderK();
  render();
  return { redraw: render };
}

const sub = (n) => String(n).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);

// 5.3
function orderSection(F, root, shared) {
  const st = { mode: "ori", seed: 1 };
  const chartBox = h("div");
  const tableBox = h("div");
  function scramble(x) {
    if (st.mode === "ori") return x;
    if (st.mode === "half") {
      const hh = x.length / 2;
      return Float32Array.from({ length: x.length }, (_, i) => x[(i + hh) % x.length]);
    }
    const r = rng(st.seed);
    const idx = range(0, x.length);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(r.uniform() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return Float32Array.from(idx, (i) => x[i]);
  }
  function render() {
    const { v, i } = shared;
    const x = scramble(window_(F, v, i));
    const y = target(F, v, i);
    const ya = aeForecast(F, 16, x).y;
    const yn = nlinear(F, x);
    const xsPast = range(-F.L, 0);
    const xsF = range(0, F.T);
    chartBox.replaceChildren(
      legendRow([["muted", st.mode === "ori" ? T.past : T.pastScrambled], ["ink", T.actual], ["s1", T.aeK(16)], ["s2", "NLinear"]]),
      lineChart({
        width: widthOf(root), height: 220, marks: [{ x: 0 }], xLabel: T.hoursFromNow, tipTitle: (t) => T.hourRel(t),
        series: [{ xs: xsPast, ys: Array.from(x), cls: "past", thin: true, label: T.input }, { xs: xsF, ys: Array.from(y), cls: "ink", label: T.actual },
          { xs: xsF, ys: Array.from(yn), cls: "s2", label: "NLinear" }, { xs: xsF, ys: Array.from(ya), cls: "s1", label: T.aeK(16) }],
      }),
      h("div", { class: "stats" },
        h("span", null, h("b", null, fmt(mse(ya, y), 3)), T.windowMse(T.aeK(16))),
        h("span", null, h("b", null, fmt(mse(yn, y), 3)), T.windowMse("NLinear"))));
  }
  const pctUp = (a, b) => `+${Math.round((b / a - 1) * 100)}%`;
  const rows = [
    [T.aeK(16) + T.ours, F.ae[16].test.mse, F.ae[16].shuffled.mse, F.ae[16].halfExchanged.mse, true],
    ["NLinear" + T.ours, F.nlinear.test.mse, F.nlinear.shuffled.mse, F.nlinear.halfExchanged.mse],
    ["Linear" + T.paperTag, 0.395, 0.824, 0.431],
    ["FEDformer" + T.paperTag, 0.376, 0.753, 0.405],
    ["Autoformer" + T.paperTag, 0.455, 0.838, 0.458],
    ["Informer" + T.paperTag, 0.974, 0.971, 0.971],
  ];
  tableBox.replaceChildren(h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
    h("thead", null, h("tr", null, h("th", null, T.testSetT96), h("th", null, T.ordOriginal), h("th", null, T.shuffled), h("th", null, T.halvesSwapped))),
    h("tbody", null, rows.map(([name, o, s, hx, hl]) => h("tr", { class: hl ? "hl" : null },
      h("td", null, name), h("td", null, fmt(o, 3)), h("td", null, `${fmt(s, 3)} (${pctUp(o, s)})`), h("td", null, `${fmt(hx, 3)} (${pctUp(o, hx)})`)))))),
  h("p", { class: "sub" }, T.orderNote));
  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.inputOrder, segToggle([{ value: "ori", label: T.ordOriginal }, { value: "shuf", label: T.shuffled }, { value: "half", label: T.halvesSwapped }], st.mode, (v) => ((st.mode = v), render()))),
      h("button", { class: "btn", type: "button", onclick: () => ((st.mode === "shuf" ? st.seed++ : 0), render()) }, T.reshuffle)),
    h("p", { class: "sub", style: { marginTop: 0 } }, T.sameWindow),
    chartBox, tableBox);
  shared.listeners.push(render);
  render();
  return { redraw: render };
}

// 5.4 — live training on a synthetic series
function labSection(root) {
  const Lw = 168;
  const Tw = 48;
  const N = 2400;
  const SPLIT = 1800;
  const st = { daily: 1, shape: "sine", weekly: 0, trend: 0, noise: 0.1, shifts: false, k: 2, running: false, steps: 0, hist: [], seed: 1 };
  let data = null;
  let net = null;
  const chartBox = h("div");
  const basisBox = h("div", { class: "minis" });
  const fcBox = h("div");
  const stats = h("div", { class: "stats" });
  const runBtn = h("button", { class: "btn primary", type: "button" }, T.train);
  const isVisible = visibility(root);

  function makeSeries() {
    const r = rng(17);
    const y = new Float32Array(N);
    let level = 0;
    for (let t = 0; t < N; t++) {
      if (st.shifts && t > 0 && t % 300 === 0) level += (r.uniform() - 0.5) * 2.4;
      const ph = (2 * Math.PI * t) / 24;
      const daily = st.shape === "sine" ? Math.sin(ph) : 0.75 * Math.sin(ph) + 0.6 * Math.sin(2 * ph + 1.1) + 0.35 * Math.sin(3 * ph + 0.4);
      y[t] = st.daily * daily + st.weekly * Math.sin((2 * Math.PI * t) / 168) + st.trend * (t / N) * 3 + level + st.noise * r.normal();
    }
    return y;
  }
  function makeWindows(y, from, to) {
    const n = to - from - Lw - Tw + 1;
    const X = new Float32Array(n * Lw);
    const Y = new Float32Array(n * Tw);
    for (let w = 0; w < n; w++) {
      const last = y[from + w + Lw - 1];
      for (let i = 0; i < Lw; i++) X[w * Lw + i] = y[from + w + i] - last;
      for (let t = 0; t < Tw; t++) Y[w * Tw + t] = y[from + w + Lw + t] - last;
    }
    return { X, Y, n };
  }
  function prepare() {
    const y = makeSeries();
    const tr = makeWindows(y, 0, SPLIT);
    const te = makeWindows(y, SPLIT - Lw, N);
    // full-rank reference: ridge least squares with a bias column
    const Xb = new Float32Array(tr.n * (Lw + 1));
    for (let w = 0; w < tr.n; w++) {
      Xb.set(tr.X.subarray(w * Lw, w * Lw + Lw), w * (Lw + 1));
      Xb[w * (Lw + 1) + Lw] = 1;
    }
    const B = ridge(Xb, tr.Y, tr.n, Lw + 1, Tw, 1e-4);
    const lsPred = (X, w) => Float32Array.from({ length: Tw }, (_, t) => {
      let acc = B[Lw * Tw + t];
      for (let i = 0; i < Lw; i++) acc += X[w * Lw + i] * B[i * Tw + t];
      return acc;
    });
    let ls = 0;
    let rep = 0;
    for (let w = 0; w < te.n; w++) {
      const p = lsPred(te.X, w);
      for (let t = 0; t < Tw; t++) {
        ls += (p[t] - te.Y[w * Tw + t]) ** 2;
        rep += te.Y[w * Tw + t] ** 2;
      }
    }
    data = { y, tr, te, lsMse: ls / (te.n * Tw), repeatMse: rep / (te.n * Tw), lsPred };
  }
  function reset() {
    st.seed++;
    net = new Net([new Dense(Lw, st.k, "linear", st.seed * 7), new Dense(st.k, Tw, "linear", st.seed * 7 + 1)]);
    st.steps = 0;
    st.hist = [];
    evaluate();
    render();
  }
  function evaluate() {
    const lossOf = (set) => {
      const P = net.forward(set.X, set.n);
      return mseGrad(P, set.Y).loss;
    };
    st.hist.push([st.steps, lossOf(data.tr), lossOf(data.te)]);
  }
  const r = rng(99);
  function trainStep() {
    const bs = 128;
    const X = new Float32Array(bs * Lw);
    const Y = new Float32Array(bs * Tw);
    for (let b = 0; b < bs; b++) {
      const w = Math.floor(r.uniform() * data.tr.n);
      X.set(data.tr.X.subarray(w * Lw, w * Lw + Lw), b * Lw);
      Y.set(data.tr.Y.subarray(w * Tw, w * Tw + Tw), b * Tw);
    }
    const P = net.forward(X, bs);
    const { grad } = mseGrad(P, Y);
    net.zeroGrad();
    net.backward(grad);
    net.step(0.01);
    st.steps++;
  }
  function loop() {
    if (!st.running) return;
    if (isVisible()) {
      for (let i = 0; i < 6; i++) trainStep();
      if (st.steps % 24 === 0) evaluate();
      render();
      if (st.steps >= 2400) return stop();
    }
    requestAnimationFrame(loop);
  }
  function start() {
    if (st.steps >= 2400) reset();
    st.running = true;
    runBtn.textContent = T.pause;
    requestAnimationFrame(loop);
  }
  function stop() {
    st.running = false;
    runBtn.textContent = T.train;
    evaluate();
    render();
  }
  function render() {
    const W = widthOf(root);
    const last = st.hist[st.hist.length - 1];
    const xs = st.hist.map((p) => p[0]);
    const yMax = Math.max(data.lsMse * 3, Math.min(data.repeatMse * 1.05, Math.max(...st.hist.map((p) => p[2])) * 1.05));
    chartBox.replaceChildren(
      legendRow([["s1", T.trainMse], ["s2", T.testMseShort]]),
      lineChart({
        width: Math.min(W, 640), height: 180, x0: 0, x1: Math.max(240, st.steps), y0: 0, y1: yMax,
        series: [{ xs, ys: st.hist.map((p) => p[1]), cls: "s1", label: T.trainMse }, { xs, ys: st.hist.map((p) => p[2]), cls: "s2", label: T.testMseShort }],
        hlines: [{ y: data.lsMse, label: `${T.fullLinear} ${fmt(data.lsMse, 3)}` }, ...(data.repeatMse < yMax ? [{ y: data.repeatMse, label: `Repeat ${fmt(data.repeatMse, 3)}` }] : [])],
        xLabel: T.trainingSteps, tipTitle: (x) => `${T.step} ${x}`, yFmt: (v) => fmt(v, 2),
      }));
    stats.replaceChildren(
      h("span", null, h("b", null, st.steps.toLocaleString(LANG_LOCALE)), T.steps),
      h("span", null, h("b", null, fmt(last[2], 3)), T.testMse),
      h("span", null, h("b", null, fmt(data.lsMse, 3)), T.fullLinear),
      h("span", { class: "stat-note" }, T.labParams(st.k * (Lw + Tw) + st.k + Tw, Lw * Tw + Tw)));
    // decoder basis: columns of D (Tw × k)
    const D = net.layers[1].W;
    basisBox.replaceChildren(...range(0, st.k).map((j) => h("div", { class: "mini" },
      h("div", { class: "ttl" }, h("b", null, `d${sub(j + 1)}`), h("span", null, T.next48)),
      spark(Float32Array.from({ length: Tw }, (_, t) => D[t * st.k + j]), { cls: "s2" }))));
    // one test forecast
    const w = data.te.n - 1 - 100;
    const base = data.y[SPLIT - Lw + w + Lw - 1];
    const P = net.forward(data.te.X.subarray(w * Lw, w * Lw + Lw), 1);
    const ls = data.lsPred(data.te.X, w);
    const xsP = range(-96, 0);
    const xsF = range(0, Tw);
    fcBox.replaceChildren(
      legendRow([["muted", T.past], ["ink", T.actual], ["s1", T.aeK(st.k)], ["s3", T.fullLinear]]),
      lineChart({
        width: Math.min(W, 640), height: 170, marks: [{ x: 0 }], xLabel: T.hoursFromNow, tipTitle: (t) => T.hourRel(t),
        series: [
          { xs: xsP, ys: xsP.map((t) => data.te.X[w * Lw + Lw + t] + base), cls: "past", thin: true, label: T.past },
          { xs: xsF, ys: xsF.map((t) => data.te.Y[w * Tw + t] + base), cls: "ink", label: T.actual },
          { xs: xsF, ys: Array.from(ls, (v) => v + base), cls: "s3", label: T.fullLinear, thin: true },
          { xs: xsF, ys: Array.from(P, (v) => v + base), cls: "s1", label: T.aeK(st.k) },
        ],
      }));
  }
  const slider = (label, key, min, max, step) => {
    const input = h("input", { type: "range", min, max, step, value: st[key], "aria-label": label });
    const out = h("output", null, fmt(st[key], 2));
    input.addEventListener("input", () => {
      st[key] = +input.value;
      out.textContent = fmt(st[key], 2);
      prepare();
      reset();
    });
    return h("div", { class: "ctl wide" }, h("span", { class: "lbl" }, label), input, out);
  };
  runBtn.addEventListener("click", () => (st.running ? stop() : start()));
  const shiftToggle = h("input", { type: "checkbox" });
  shiftToggle.addEventListener("change", () => {
    st.shifts = shiftToggle.checked;
    prepare();
    reset();
  });
  root.replaceChildren(
    h("div", { class: "controls" }, slider(T.dailyCycle, "daily", 0, 1.5, 0.05),
      ctl(T.dailyShape, segToggle([{ value: "sine", label: T.sine }, { value: "peaks", label: T.peaks }], st.shape, (v) => ((st.shape = v), prepare(), reset())))),
    h("div", { class: "controls" }, slider(T.weeklyCycle, "weekly", 0, 1, 0.05), slider(T.trend, "trend", -1, 1, 0.05)),
    h("div", { class: "controls" }, slider(T.noise, "noise", 0, 0.6, 0.02), h("label", { class: "toggle" }, shiftToggle, T.levelShifts)),
    h("div", { class: "controls" }, ctl("k", segToggle([1, 2, 3, 4, 6, 8].map((k) => ({ value: k, label: String(k) })), st.k, (v) => ((st.k = v), reset()))), runBtn,
      h("button", { class: "btn", type: "button", onclick: () => reset() }, T.reset)),
    stats,
    h("div", { class: "viz-row", style: { marginTop: "10px" } }, h("div", { class: "grow" }, chartBox), h("div", { class: "grow" }, fcBox)),
    h("div", { class: "panel-title", style: { marginTop: "12px" }, html: T.decoderBasis }), basisBox);
  prepare();
  reset();
  // start training the first time the lab scrolls into view
  new IntersectionObserver((entries, obs) => {
    if (entries.some((e) => e.isIntersecting)) {
      obs.disconnect();
      if (!st.running && st.steps === 0) start();
    }
  }, { threshold: 0.3 }).observe(root);
  return { redraw: render };
}

// 6.2 — the conditional VAE's fan chart
function fanSection(F, root) {
  const st = { v: 0, i: 2200, n: 300, seed: 1 };
  const c = F.cvae;
  const chartBox = h("div");
  const stats = h("div", { class: "stats" });
  function render() {
    const x = window_(F, st.v, st.i);
    const y = target(F, st.v, st.i);
    const { base, mu, logvar } = cvaeParts(F, x);
    const r = rng(st.seed);
    const S = st.n;
    const samples = new Float32Array(S * F.T);
    const paths = [];
    for (let s = 0; s < S; s++) {
      const z = Float32Array.from(mu, (m, j) => m + Math.exp(0.5 * logvar[j]) * r.normal());
      const path = new Float32Array(F.T);
      for (let t = 0; t < F.T; t++) {
        let acc = base[t];
        for (let j = 0; j < c.kz; j++) acc += c.U[t * c.kz + j] * z[j];
        path[t] = acc;
        samples[s * F.T + t] = acc + c.sigma[t] * r.normal();
      }
      if (s < 3) paths.push(path);
    }
    const q = (p) => Float32Array.from({ length: F.T }, (_, t) => {
      const col = Float32Array.from({ length: S }, (_, s) => samples[s * F.T + t]).sort();
      const pos = p * (S - 1);
      const lo = Math.floor(pos);
      return col[lo] + (col[Math.min(S - 1, lo + 1)] - col[lo]) * (pos - lo);
    });
    const [q10, q25, q50, q75, q90] = [0.1, 0.25, 0.5, 0.75, 0.9].map(q);
    const xsP = range(-168, 0);
    const xsF = range(0, F.T);
    let inside = 0;
    let width = 0;
    for (let t = 0; t < F.T; t++) {
      if (y[t] >= q10[t] && y[t] <= q90[t]) inside++;
      width += q90[t] - q10[t];
    }
    chartBox.replaceChildren(
      legendRow([["muted", T.past], ["ink", T.actual], ["s1", T.median], ["s1", T.band80, "band"], ["s3", T.trajectories]]),
      lineChart({
        width: widthOf(root), height: 260, marks: [{ x: 0 }], xLabel: T.hoursFromNow, tipTitle: (t) => T.hourRel(t),
        bands: [{ xs: xsF, lo: q10, hi: q90, color: "var(--s1)", opacity: 0.16 }, { xs: xsF, lo: q25, hi: q75, color: "var(--s1)", opacity: 0.28 }],
        series: [
          { xs: xsP, ys: Array.from(x.subarray(F.L - 168)), cls: "past", thin: true, label: T.past },
          ...paths.map((p, k) => ({ xs: xsF, ys: Array.from(p), cls: "s3", thin: true, label: `${T.trajectory} ${k + 1}` })),
          { xs: xsF, ys: Array.from(q50), cls: "s1", label: T.median },
          { xs: xsF, ys: Array.from(y), cls: "ink", label: T.actual },
        ],
      }));
    stats.replaceChildren(
      h("span", null, h("b", null, `${inside} / ${F.T}`), T.insideBand),
      h("span", null, h("b", null, fmt(width / F.T, 2)), T.bandWidth),
      h("span", { class: "stat-note" }, T.fanNote));
  }
  const tb = c.test;
  const bl = c.baseline;
  const table = h("div", { class: "scroll-x" }, h("table", { class: "tbl" },
    h("thead", null, h("tr", null, h("th", null, T.testSetAll), h("th", null, T.pointMse), h("th", null, T.coverage80), h("th", null, T.width80), h("th", null, "CRPS"))),
    h("tbody", null,
      h("tr", { class: "ref" }, h("td", null, T.residualBaseline), h("td", null, fmt(bl.mse, 3)), h("td", null, pct(bl.coverage80, 1)), h("td", null, fmt(bl.width80, 2)), h("td", null, fmt(bl.crps, 3))),
      h("tr", { class: "hl" }, h("td", null, T.cvaeName), h("td", null, fmt(tb.mse, 3)), h("td", null, pct(tb.coverage80, 1)), h("td", null, fmt(tb.width80, 2)), h("td", null, fmt(tb.crps, 3))))));
  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.variable, varSelect(F, st.v, (v) => ((st.v = v), render()))),
      ctl(T.samples, segToggle([{ value: 100, label: "100" }, { value: 300, label: "300" }, { value: 1000, label: "1000" }], st.n, (v) => ((st.n = v), render()))),
      h("button", { class: "btn primary", type: "button", onclick: () => (st.seed++, render()) }, T.resample)),
    h("div", { class: "controls" }, windowSlider(F, st.i, (i) => ((st.i = i), render()))),
    chartBox, stats, h("div", { style: { marginTop: "12px" } }, table));
  render();
  return { redraw: render };
}

