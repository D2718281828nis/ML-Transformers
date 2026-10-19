// Interlude "How the embedding matrix is created and trained" (E.1–E.5).
// E.1 compares a freshly initialized row with GPT-2's trained row; E.3–E.5 train the
// tiny tied-embedding model from mini-lm.js live in the browser.

import { MAX_VOCAB, prepare, initModel, trainStep, currentLoss, exampleGrads, gaussian } from "./mini-lm.js";
import { cosine } from "./model.js";
import { T } from "./i18n.js";
import { h, s, fmt, fill, lineChart, heatRows, matrixGrid, cellSize, divergingRGB, robustScale, legend, tokChip, tokTip, showTip, hideTip } from "./viz.js";

const STEPS_PER_RUN = 600;
const STEPS_PER_FRAME = 3;
const TRAIL_EVERY = 4;

const st = { preset: 0, text: T.ePresets[0].text, seed: 1, pair: 0, reroll: 0, runTo: 0 };
const ui = {};
let data = null;
let model = null;
let trails = [];
let view = 0.05; // half-width of the plotted square
let raf = 0;
let lastReal = null;

const $ = (id) => document.getElementById(id);
const tc = (i) => `var(--tok${i % 10})`;

// ───────────── E.1: a fresh random row vs. the trained one ─────────────
export function renderEmbInit(args) {
  lastReal = args;
  const box = $("viz-emb-init");
  if (!box) return;
  const { res, tokens, focus, placeholder } = args;
  if (!res) return placeholder(box);
  const n = tokens.length;
  const t = tokens[focus];
  // Same token → same random row, as in a real freshly created matrix.
  const fresh = (id) => {
    const g = gaussian(id * 7919 + st.reroll * 104729 + 1);
    return Float32Array.from({ length: 768 }, () => 0.02 * g());
  };
  const freshRows = tokens.map((tk) => fresh(tk.id));
  const trained = res.tokEmb;
  const sc = robustScale([trained[focus]]);
  const std = (v) => {
    const m = v.reduce((a, x) => a + x, 0) / v.length;
    return Math.sqrt(v.reduce((a, x) => a + (x - m) ** 2, 0) / v.length);
  };
  const strip = (label, vec, tip) =>
    h("div", { style: { marginBottom: "10px" } },
      h("div", { class: "sub", style: { margin: "0 0 3px" } }, label),
      heatRows([vec], { scale: sc, rowHeight: 22, axis: false, describe: (_r, d, v) => tip(tokTip(t), d, fmt(v, 4)) }));
  const size = Math.min(cellSize(n, 260), 34);
  const cosGrid = (rows) =>
    matrixGrid({
      n, size,
      rowLabels: tokens.map((tk) => tokChip(tk)),
      colLabels: tokens.map((tk) => tokChip(tk)),
      cell: (i, j) => {
        const c = cosine(rows[i], rows[j]);
        return { bg: divergingRGB(c, 1), text: fmt(c, 2), tip: `cos(${tokTip(tokens[i])}, ${tokTip(tokens[j])}) = <b>${fmt(c, 3)}</b>` };
      },
    });
  fill(box,
    h("div", { class: "viz-head" },
      h("div", { class: "panel-title", style: { margin: 0 } }, T.selectedToken, tokChip(t)),
      h("div", { class: "viz-head-right" },
        h("button", { class: "btn", type: "button", onclick: () => (st.reroll++, renderEmbInit(lastReal)) }, T.eReroll),
        legend(sc))),
    strip(T.eFresh, freshRows[focus], T.eFreshTip),
    strip(T.eTrained, trained[focus], T.eTrainedTip),
    h("div", { class: "stats", style: { marginTop: "4px" } }, h("span", null, T.eStds(fmt(std(freshRows[focus]), 3), fmt(std(trained[focus]), 3)))),
    h("div", { class: "viz-row", style: { marginTop: "16px" } },
      h("div", null, h("div", { class: "panel-title" }, T.eCosInit), cosGrid(freshRows)),
      h("div", null, h("div", { class: "panel-title" }, T.eCosTrained), cosGrid(trained))),
    h("p", { class: "sub" }, T.eCosNote),
  );
}

// ───────────── E.3–E.5: the tiny model ─────────────
export function initEmbeddingTraining() {
  setText(st.text, st.preset);
}

export function renderEmbeddingTraining() {
  renderMini();
  renderExample();
  renderNeighbours();
}

function setText(text, preset = -1) {
  stop();
  st.text = text;
  st.preset = preset;
  st.pair = 0;
  data = prepare(text);
  restart();
}

function restart() {
  stop();
  model = initModel(data.vocab.length, st.seed);
  trails = data.vocab.map((_, w) => [[model.params.E[w * 2], model.params.E[w * 2 + 1]]]);
  st.runTo = 0;
  fitView(true);
  renderEmbeddingTraining();
}

function fitView(snap) {
  let m = 0;
  for (const v of model.params.E) m = Math.max(m, Math.abs(v));
  const target = Math.max(0.05, m * 1.18);
  view = snap ? target : view + (target - view) * 0.25;
}

function step() {
  trainStep(model, data.pairs);
  if (model.t % TRAIL_EVERY === 0) data.vocab.forEach((_, w) => trails[w].push([model.params.E[w * 2], model.params.E[w * 2 + 1]]));
}

function stop() {
  cancelAnimationFrame(raf);
  raf = 0;
}

function play() {
  if (raf) return stop(), renderMini();
  if (data.pairs.length === 0) return;
  st.runTo = model.t + STEPS_PER_RUN;
  const frame = () => {
    for (let k = 0; k < STEPS_PER_FRAME && model.t < st.runTo; k++) step();
    fitView(false);
    renderEmbeddingTraining();
    raf = model.t < st.runTo ? requestAnimationFrame(frame) : 0;
    if (!raf) {
      fitView(true);
      renderEmbeddingTraining();
    }
  };
  raf = requestAnimationFrame(frame);
}

// 2-D scatter of the rows of E, with optional arrows and the vector h.
function scatter(E, { arrows = [], hvec = null, trail = true } = {}) {
  const W = 360;
  const pad = 30;
  const X = (x) => W / 2 + (x / view) * (W / 2 - pad);
  const Y = (y) => W / 2 - (y / view) * (W / 2 - pad);
  const svg = s("svg", { class: "scatter", viewBox: `0 0 ${W} ${W}`, role: "img", "aria-label": T.ePlotAria });
  svg.append(
    s("line", { class: "axis", x1: pad / 2, x2: W - pad / 2, y1: W / 2, y2: W / 2 }),
    s("line", { class: "axis", y1: pad / 2, y2: W - pad / 2, x1: W / 2, x2: W / 2 }),
    s("text", { class: "ax", x: W - pad / 2, y: W / 2 - 5, "text-anchor": "end" }, `${T.dim(0)} · ${fmt(view, view < 1 ? 2 : 1)}`),
    s("text", { class: "ax", x: W / 2 + 5, y: pad / 2 + 8 }, `${T.dim(1)} · ${fmt(view, view < 1 ? 2 : 1)}`),
  );
  const V = data.vocab.length;
  if (trail) {
    for (let w = 0; w < V; w++) {
      if (trails[w].length < 2) continue;
      svg.append(s("polyline", { class: "trail", points: trails[w].map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(" "), style: `stroke: ${tc(w)}` }));
    }
  }
  if (hvec) {
    // h can be far outside the view; keep its direction and shorten it to fit.
    const k = Math.min(1, (0.92 * view) / Math.max(Math.abs(hvec[0]), Math.abs(hvec[1]), 1e-12));
    const hx = X(hvec[0] * k);
    const hy = Y(hvec[1] * k);
    svg.append(
      s("line", { class: "hvec", x1: W / 2, y1: W / 2, x2: hx, y2: hy }),
      s("circle", { class: "hdot", cx: hx, cy: hy, r: 4 }),
      s("text", { class: "hlabel", x: hx + 6, y: hy - 6 }, "h"),
    );
  }
  for (const a of arrows) {
    const x1 = X(a.from[0]);
    const y1 = Y(a.from[1]);
    const x2 = x1 + a.dx;
    const y2 = y1 - a.dy;
    const ang = Math.atan2(y2 - y1, x2 - x1);
    const hd = 7;
    svg.append(
      s("line", { class: `arrow ${a.kind}`, x1, y1, x2, y2 }),
      s("polygon", {
        class: `arrowhead ${a.kind}`,
        points: `${x2},${y2} ${x2 - hd * Math.cos(ang - 0.45)},${y2 - hd * Math.sin(ang - 0.45)} ${x2 - hd * Math.cos(ang + 0.45)},${y2 - hd * Math.sin(ang + 0.45)}`,
      }),
    );
  }
  // points and labels; labels of points that sit on top of each other are stacked
  const placed = [];
  for (let w = 0; w < V; w++) {
    const px = X(E[w * 2]);
    const py = Y(E[w * 2 + 1]);
    const label = data.vocab[w];
    const lw = 7 * label.length + 4;
    let dy = 0;
    for (const off of [0, 13, -13, 26, -26, 39, -39]) {
      const box = [px + 7, py - 6 + off, lw, 12];
      if (!placed.some((b) => box[0] < b[0] + b[2] && b[0] < box[0] + box[2] && box[1] < b[1] + b[3] && b[1] < box[1] + box[3])) {
        dy = off;
        placed.push(box);
        break;
      }
    }
    const g = s("g", { class: "pt" });
    g.append(
      s("circle", { cx: px, cy: py, r: 5.5, style: `fill: ${tc(w)}` }),
      s("text", { x: px + 8, y: py + 4 + dy }, label),
    );
    g.addEventListener("pointermove", (ev) => showTip(`<b>${label}</b><br>E[${w}] = (${fmt(E[w * 2], 3)}, ${fmt(E[w * 2 + 1], 3)})`, ev));
    g.addEventListener("pointerleave", hideTip);
    svg.append(g);
  }
  return svg;
}

function renderMini() {
  const box = $("viz-emb-train");
  if (!box || !model) return;
  let dyn = box.querySelector(":scope > .dyn");
  if (!dyn) {
    ui.presets = T.ePresets.map((p, i) => h("button", { type: "button", onclick: () => ((ui.area.value = p.text), setText(p.text, i)) }, p.label));
    ui.area = h("textarea", { class: "corpus", rows: 2, spellcheck: "false", maxlength: 1200, "aria-label": T.eCorpusLabel });
    ui.area.value = st.text;
    let tmr = 0;
    ui.area.addEventListener("input", () => {
      clearTimeout(tmr);
      tmr = setTimeout(() => setText(ui.area.value, -1), 300);
    });
    ui.play = h("button", { class: "btn primary", type: "button", onclick: play });
    ui.step = h("button", { class: "btn", type: "button", onclick: () => (stop(), step(), fitView(true), renderEmbeddingTraining()) }, T.eStep);
    ui.restart = h("button", { class: "btn", type: "button", onclick: () => (st.seed++, restart()) }, T.eNewStart);
    ui.counter = h("span", { class: "mono step-count" });
    dyn = h("div", { class: "dyn" });
    fill(box,
      h("div", { class: "examples presets" }, h("span", { class: "panel-title", style: { margin: 0 } }, T.presetsLabel), ui.presets),
      h("label", { class: "panel-title", style: { display: "block" } }, T.eCorpusLabel),
      ui.area,
      h("div", { class: "bpe-controls", style: { marginTop: "8px" } }, ui.play, ui.step, ui.restart, ui.counter),
      dyn);
  }
  ui.presets.forEach((b, i) => b.setAttribute("aria-pressed", String(i === st.preset)));
  ui.play.textContent = raf ? T.ePause : model.t ? T.eTrainMore : T.eTrain;
  ui.counter.textContent = T.eStepN(model.t);
  const disabled = data.pairs.length === 0;
  ui.play.disabled = ui.step.disabled = disabled;
  if (disabled) return fill(dyn, h("div", { class: "placeholder" }, T.eNotEnough));

  const V = data.vocab.length;
  const E = model.params.E;
  const now = currentLoss(model, data.pairs);
  const losses = [...model.losses, now];
  fill(dyn,
    data.truncated ? h("div", { class: "callout warn" }, T.eTruncated(MAX_VOCAB)) : null,
    h("div", { class: "emb-train-grid" },
      h("div", null, h("div", { class: "panel-title" }, T.ePlotTitle), scatter(E), h("p", { class: "sub" }, T.eZoomNote)),
      h("div", null,
        h("div", { class: "loss-line" }, T.eLossLine(fmt(now, 3), fmt(Math.log(V), 3), fmt(data.floor, 3))),
        lineChart([{ label: T.eLossChart, values: losses }], { height: 130, positive: true, digits: 2, xLabel: (i) => String(i), tipLabel: T.eAfterSteps }),
        h("div", { class: "panel-title", style: { marginTop: "12px" } }, T.eMatrixTitle(V)),
        h("div", { class: "scroll-x" },
          h("table", { class: "sm-table emb-table" },
            h("thead", null, h("tr", null, h("th", null, T.eWord), h("th", null, T.dim(0)), h("th", null, T.dim(1)))),
            h("tbody", null, data.vocab.map((word, w) =>
              h("tr", null,
                h("td", null, h("span", { class: "tok static", style: `--tc: ${tc(w)}` }, word)),
                h("td", null, fmt(E[w * 2], 3)),
                h("td", null, fmt(E[w * 2 + 1], 3))))))))),
  );
}

// ───────────── E.4: one example up close ─────────────
function uniquePairs() {
  const seen = new Map();
  for (const [i, j] of data.pairs) {
    const key = `${i},${j}`;
    seen.set(key, { i, j, n: (seen.get(key)?.n || 0) + 1 });
  }
  return [...seen.values()];
}

function renderExample() {
  const box = $("viz-emb-step");
  if (!box || !model) return;
  if (!data.pairs.length) return fill(box, h("div", { class: "placeholder" }, T.eNotEnough));
  const pairs = uniquePairs();
  st.pair = Math.min(st.pair, pairs.length - 1);
  const { i, j } = pairs[st.pair];
  const V = data.vocab.length;
  const P = model.params;
  const g = exampleGrads(P, V, i, j);
  const word = (w) => h("span", { class: "tok static", style: `--tc: ${tc(w)}` }, data.vocab[w]);

  // Arrows point against the gradient (the way a step moves the row), scaled for visibility.
  const raw = [];
  for (let w = 0; w < V; w++) {
    if (w !== j && g.p[w] < 0.03) continue;
    raw.push({ from: [P.E[w * 2], P.E[w * 2 + 1]], v: [-g.gOut[w * 2], -g.gOut[w * 2 + 1]], kind: w === j ? "pull" : "push" });
  }
  raw.push({ from: [P.E[i * 2], P.E[i * 2 + 1]], v: [-g.gIn[0], -g.gIn[1]], kind: "in" });
  const maxN = Math.max(...raw.map((a) => Math.hypot(...a.v)), 1e-12);
  const arrows = raw
    .filter((a) => Math.hypot(...a.v) > 0.04 * maxN)
    .map((a) => ({ ...a, dx: (a.v[0] / maxN) * 70, dy: (a.v[1] / maxN) * 70 }));

  const order = [...g.p.keys()].sort((a, b) => g.p[b] - g.p[a]);
  fill(box,
    h("div", { class: "panel-title" }, T.ePickPair),
    h("div", { class: "chunks", style: { marginBottom: "12px" } },
      pairs.map((pr, k) =>
        h("button", { class: "chunk", type: "button", "aria-pressed": String(k === st.pair), onclick: () => ((st.pair = k), renderExample()) },
          h("span", { class: "chunk-text" }, `${data.vocab[pr.i]} → ${data.vocab[pr.j]}`),
          pr.n > 1 ? h("span", { class: "chunk-kind" }, `×${pr.n}`) : null))),
    h("ol", { class: "steps-list emb-steps" },
      h("li", null, T.eSteps[0], " ", word(i), `: e = (${fmt(P.E[i * 2], 3)}, ${fmt(P.E[i * 2 + 1], 3)})`),
      h("li", null, T.eSteps[1], `: h = (${fmt(g.h[0], 3)}, ${fmt(g.h[1], 3)})`),
      h("li", null, T.eSteps[2])),
    h("div", { class: "emb-train-grid" },
      h("div", null,
        h("div", { class: "panel-title" }, T.eProbsTitle),
        h("table", { class: "sm-table prob-table" },
          h("tbody", null, order.map((w) =>
            h("tr", { class: w === j ? "win" : null },
              h("td", null, word(w)),
              h("td", { class: "barcell" }, h("div", { class: "wbar", style: { width: `${g.p[w] * 100}%` } })),
              h("td", null, `${(g.p[w] * 100).toFixed(1)}%`))))),
        h("div", { class: "bpe-msg", style: { marginTop: "8px" } }, T.eLossEx(fmt(g.p[j], 3), fmt(g.loss, 3)))),
      h("div", null,
        h("div", { class: "panel-title" }, T.eArrowsTitle),
        scatter(P.E, { arrows, hvec: g.h, trail: false }),
        h("ul", { class: "arrow-legend" },
          h("li", null, h("i", { class: "pull" }), T.eArrowPull),
          h("li", null, h("i", { class: "push" }), T.eArrowPush),
          h("li", null, h("i", { class: "in" }), T.eArrowIn)),
        h("p", { class: "sub" }, T.eArrowsNote))),
  );
}

// ───────────── E.5: who ended up next to whom ─────────────
function renderNeighbours() {
  const box = $("viz-emb-near");
  if (!box || !model) return;
  if (!data.pairs.length) return fill(box);
  const E = model.params.E;
  const V = data.vocab.length;
  const list = [];
  for (let a = 0; a < V; a++) for (let b = a + 1; b < V; b++) list.push({ a, b, d: Math.hypot(E[a * 2] - E[b * 2], E[a * 2 + 1] - E[b * 2 + 1]) });
  list.sort((x, y) => x.d - y.d);
  const word = (w) => h("span", { class: "tok static", style: `--tc: ${tc(w)}` }, data.vocab[w]);
  fill(box,
    h("div", { class: "panel-title" }, T.eNeighbours, " ", model.t ? null : h("span", { class: "muted" }, T.eUntrained)),
    h("table", { class: "sm-table near-table" },
      h("thead", null, h("tr", null, h("th", null, ""), h("th", null, ""), h("th", null, T.eDistance))),
      h("tbody", null, list.slice(0, 6).map(({ a, b, d }) => h("tr", null, h("td", null, word(a)), h("td", null, word(b)), h("td", null, fmt(d, 3)))))),
    h("p", { class: "sub" }, T.eNeighboursNote),
  );
}
