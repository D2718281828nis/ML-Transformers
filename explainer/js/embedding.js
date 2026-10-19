// Interlude "How the embedding matrix is created and trained" (E.1–E.5).
// E.1 compares a freshly initialized row with GPT-2's trained row; E.3–E.5 train the
// tiny tied-embedding model from mini-lm.js live in the browser.

import { MAX_VOCAB, prepare, initModel, trainStep, currentLoss, exampleGrads, gaussian } from "./mini-lm.js";
import { cosine } from "./model.js";
import { T } from "./i18n.js";
import { h, s, fmt, fill, lineChart, heatRows, matrixGrid, cellSize, divergingRGB, robustScale, legend, tokChip, tokTip, showTip, hideTip } from "./viz.js";

const STEPS_PER_RUN = 600;
const SPEEDS = [1, 3, 12]; // training steps per animation frame: slow, normal, fast
const SNAPSHOT_EVERY = 4; // history for the rewind slider and the trails
const MAX_SNAPSHOTS = 2000;

const st = {
  preset: 0, text: T.ePresets[0].text, seed: 1, pair: 0, reroll: 0, runTo: 0,
  speed: 1, // index into SPEEDS
  viewIdx: null, // null = live; otherwise the snapshot shown by the rewind slider
  hover: null, // word highlighted in plot + table
  dragged: null, // { w, before, after } after a row was moved by hand
};
const ui = {};
let data = null;
let model = null;
let hist = []; // snapshots { t, E, loss }
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
  hist = [];
  st.runTo = 0;
  st.viewIdx = null;
  st.dragged = null;
  snapshot();
  fitView(true);
  renderEmbeddingTraining();
}

function snapshot() {
  const last = hist[hist.length - 1];
  const snap = { t: model.t, E: Float64Array.from(model.params.E), loss: currentLoss(model, data.pairs) };
  if (last && last.t === snap.t && !st.dragged) hist[hist.length - 1] = snap;
  else hist.push(snap);
  if (hist.length > MAX_SNAPSHOTS) hist.splice(1, 1);
}

// The E being shown: live, or a snapshot picked with the rewind slider.
const shownE = () => (st.viewIdx == null ? model.params.E : hist[st.viewIdx].E);

function fitView(snap) {
  let m = 0;
  for (const v of shownE()) m = Math.max(m, Math.abs(v));
  const target = Math.max(0.05, m * 1.18);
  view = snap ? target : view + (target - view) * 0.25;
}

function step() {
  trainStep(model, data.pairs);
  if (model.t % SNAPSHOT_EVERY === 0) snapshot();
}

function goLive() {
  if (st.viewIdx == null) return;
  st.viewIdx = null;
  fitView(true);
}

function stop() {
  cancelAnimationFrame(raf);
  raf = 0;
}

function play() {
  if (raf) return stop(), renderMini();
  if (data.pairs.length === 0) return;
  goLive();
  st.dragged = null;
  st.runTo = model.t + STEPS_PER_RUN;
  const frame = () => {
    for (let k = 0; k < SPEEDS[st.speed] && model.t < st.runTo; k++) step();
    fitView(false);
    renderEmbeddingTraining();
    raf = model.t < st.runTo ? requestAnimationFrame(frame) : 0;
    if (!raf) {
      snapshot();
      fitView(true);
      renderEmbeddingTraining();
    }
  };
  raf = requestAnimationFrame(frame);
}

function oneStep() {
  stop();
  goLive();
  st.dragged = null;
  step();
  snapshot();
  fitView(true);
  renderEmbeddingTraining();
}

// Highlight one word's dot, path and table row everywhere in the trainer.
function setHover(w) {
  st.hover = w;
  applyHover();
}
function applyHover() {
  const box = $("viz-emb-train");
  if (!box) return;
  box.classList.toggle("hovering", st.hover != null);
  for (const el of box.querySelectorAll(".hl")) el.classList.remove("hl");
  if (st.hover != null) for (const el of box.querySelectorAll(`[data-w="${st.hover}"]`)) el.classList.add("hl");
}

// 2-D scatter of the rows of E, with optional arrows and the vector h.
function scatter(E, { arrows = [], hvec = null, trail = false, draggable = false, onDrag = null, onDrop = null } = {}) {
  const W = 360;
  const pad = 30;
  const X = (x) => W / 2 + (x / view) * (W / 2 - pad);
  const Y = (y) => W / 2 - (y / view) * (W / 2 - pad);
  const svg = s("svg", { class: `scatter${draggable ? " draggable" : ""}`, viewBox: `0 0 ${W} ${W}`, role: "img", "aria-label": T.ePlotAria });
  svg.append(
    s("line", { class: "axis", x1: pad / 2, x2: W - pad / 2, y1: W / 2, y2: W / 2 }),
    s("line", { class: "axis", y1: pad / 2, y2: W - pad / 2, x1: W / 2, x2: W / 2 }),
    s("text", { class: "ax", x: W - pad / 2, y: W / 2 - 5, "text-anchor": "end" }, `${T.dim(0)} · ${fmt(view, view < 1 ? 2 : 1)}`),
    s("text", { class: "ax", x: W / 2 + 5, y: pad / 2 + 8 }, `${T.dim(1)} · ${fmt(view, view < 1 ? 2 : 1)}`),
  );
  const V = data.vocab.length;
  if (trail) {
    const upto = st.viewIdx == null ? hist.length : st.viewIdx + 1;
    if (upto > 1) {
      for (let w = 0; w < V; w++) {
        const pts = [];
        for (let k = 0; k < upto; k++) pts.push(`${X(hist[k].E[w * 2]).toFixed(1)},${Y(hist[k].E[w * 2 + 1]).toFixed(1)}`);
        svg.append(s("polyline", { class: "trail", "data-w": w, points: pts.join(" "), style: `stroke: ${tc(w)}` }));
      }
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
    const g = s("g", { class: "pt", "data-w": w });
    const dot = s("circle", { cx: px, cy: py, r: 5.5, style: `fill: ${tc(w)}` });
    const txt = s("text", { x: px + 8, y: py + 4 + dy }, label);
    g.append(dot, txt);
    g.addEventListener("pointermove", (ev) => svg.dataset.dragging || showTip(`<b>${label}</b><br>E[${w}] = (${fmt(E[w * 2], 3)}, ${fmt(E[w * 2 + 1], 3)})`, ev));
    g.addEventListener("pointerenter", () => setHover(w));
    g.addEventListener("pointerleave", () => (hideTip(), svg.dataset.dragging || setHover(null)));
    if (draggable) {
      // Drag a dot = change that row of E by hand.
      g.addEventListener("pointerdown", (ev) => {
        ev.preventDefault();
        hideTip();
        svg.setPointerCapture(ev.pointerId);
        svg.dataset.dragging = "1";
        const move = (e) => {
          const pt = svg.createSVGPoint();
          pt.x = e.clientX;
          pt.y = e.clientY;
          const p = pt.matrixTransform(svg.getScreenCTM().inverse());
          const x = Math.max(-view, Math.min(view, ((p.x - W / 2) / (W / 2 - pad)) * view));
          const y = Math.max(-view, Math.min(view, (-(p.y - W / 2) / (W / 2 - pad)) * view));
          E[w * 2] = x;
          E[w * 2 + 1] = y;
          dot.setAttribute("cx", X(x));
          dot.setAttribute("cy", Y(y));
          txt.setAttribute("x", X(x) + 8);
          txt.setAttribute("y", Y(y) + 4);
          onDrag?.(w);
        };
        const up = () => {
          svg.removeEventListener("pointermove", move);
          svg.removeEventListener("pointerup", up);
          svg.removeEventListener("pointercancel", up);
          delete svg.dataset.dragging;
          onDrop?.(w);
        };
        svg.addEventListener("pointermove", move);
        svg.addEventListener("pointerup", up);
        svg.addEventListener("pointercancel", up);
      });
    }
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
    ui.step = h("button", { class: "btn", type: "button", onclick: oneStep }, T.eStep);
    ui.restart = h("button", { class: "btn", type: "button", onclick: () => (st.seed++, restart()) }, T.eNewStart);
    ui.counter = h("span", { class: "mono step-count" });
    ui.speeds = T.eSpeeds.map((label, k) => h("button", { type: "button", onclick: () => ((st.speed = k), renderMini()) }, label));
    ui.scrub = h("input", { type: "range", min: 0, max: 0, step: 1, value: 0, class: "scrub", "aria-label": T.eRewindAria });
    ui.scrub.addEventListener("input", () => {
      stop();
      const v = Number(ui.scrub.value);
      st.viewIdx = v >= hist.length - 1 ? null : v;
      fitView(true);
      renderMini();
    });
    ui.scrubLabel = h("span", { class: "mono scrub-label" });
    dyn = h("div", { class: "dyn" });
    fill(box,
      h("div", { class: "examples presets" }, h("span", { class: "panel-title", style: { margin: 0 } }, T.presetsLabel), ui.presets),
      h("label", { class: "panel-title", style: { display: "block" } }, T.eCorpusLabel),
      ui.area,
      h("div", { class: "bpe-controls", style: { marginTop: "8px" } },
        ui.play, ui.step, ui.restart, ui.counter,
        h("span", { class: "speed" }, h("span", { class: "panel-title", style: { margin: 0 } }, T.eSpeed), h("span", { class: "seg-toggle" }, ui.speeds))),
      h("div", { class: "scrub-row" }, h("span", { class: "panel-title", style: { margin: 0 } }, T.eRewind), ui.scrub, ui.scrubLabel),
      h("p", { class: "sub hint" }, T.eHint),
      dyn);
  }
  ui.presets.forEach((b, i) => b.setAttribute("aria-pressed", String(i === st.preset)));
  ui.speeds.forEach((b, k) => b.setAttribute("aria-pressed", String(k === st.speed)));
  ui.play.textContent = raf ? T.ePause : model.t ? T.eTrainMore : T.eTrain;
  ui.counter.textContent = T.eStepN(model.t);
  const disabled = data.pairs.length === 0;
  ui.play.disabled = ui.step.disabled = ui.scrub.disabled = disabled;
  ui.scrub.max = Math.max(0, hist.length - 1);
  ui.scrub.value = st.viewIdx ?? hist.length - 1;
  ui.scrubLabel.textContent = st.viewIdx == null ? T.eLive(model.t) : T.eRewound(hist[st.viewIdx].t);
  if (disabled) return fill(dyn, h("div", { class: "placeholder" }, T.eNotEnough));

  const V = data.vocab.length;
  const live = st.viewIdx == null;
  const E = shownE();
  const now = live ? currentLoss(model, data.pairs) : hist[st.viewIdx].loss;
  const losses = [...model.losses, currentLoss(model, data.pairs)];
  const lossLine = h("div", { class: "loss-line" }, T.eLossLine(fmt(now, 3), fmt(Math.log(V), 3), fmt(data.floor, 3)));
  const cells = [];
  const cellStyle = (v) => {
    const c = divergingRGB(v, view);
    return `background: rgb(${c[0] | 0} ${c[1] | 0} ${c[2] | 0}); color: ${0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 140 ? "#fff" : "inherit"}`;
  };
  // While dragging, update the loss and the dragged row in place (no full re-render).
  let before = 0;
  const onDrag = (w) => {
    lossLine.textContent = T.eLossLine(fmt(currentLoss(model, data.pairs), 3), fmt(Math.log(V), 3), fmt(data.floor, 3));
    cells[w].forEach((td, d) => {
      td.textContent = fmt(E[w * 2 + d], 3);
      td.setAttribute("style", cellStyle(E[w * 2 + d]));
    });
  };
  const plot = scatter(E, {
    trail: true,
    draggable: live && !raf,
    onDrag,
    onDrop: (w) => {
      st.dragged = { w, before, after: currentLoss(model, data.pairs) };
      snapshot();
      renderEmbeddingTraining();
    },
  });
  plot.addEventListener("pointerdown", () => (before = currentLoss(model, data.pairs)), true);
  fill(dyn,
    data.truncated ? h("div", { class: "callout warn" }, T.eTruncated(MAX_VOCAB)) : null,
    h("div", { class: "emb-train-grid" },
      h("div", null, h("div", { class: "panel-title" }, T.ePlotTitle), plot, h("p", { class: "sub" }, T.eZoomNote)),
      h("div", null,
        lossLine,
        st.dragged && live
          ? h("div", { class: "callout warn", style: { margin: "4px 0 8px" } },
              T.eDragged(data.vocab[st.dragged.w], fmt(st.dragged.before, 3), fmt(st.dragged.after, 3)))
          : null,
        lineChart([{ label: T.eLossChart, values: losses }], {
          height: 130, positive: true, digits: 2, xLabel: (i) => String(i), tipLabel: T.eAfterSteps,
          highlight: live ? losses.length - 1 : Math.min(hist[st.viewIdx].t, losses.length - 1),
        }),
        h("div", { class: "panel-title", style: { marginTop: "12px" } }, T.eMatrixTitle(V), " ", live ? null : h("span", { class: "muted" }, T.eRewound(hist[st.viewIdx].t))),
        h("div", { class: "scroll-x" },
          h("table", { class: "sm-table emb-table" },
            h("thead", null, h("tr", null, h("th", null, T.eWord), h("th", null, T.dim(0)), h("th", null, T.dim(1)))),
            h("tbody", null, data.vocab.map((word, w) => {
              const tds = [0, 1].map((d) => h("td", { class: "val", style: cellStyle(E[w * 2 + d]) }, fmt(E[w * 2 + d], 3)));
              cells[w] = tds;
              const tr = h("tr", { "data-w": w }, h("td", null, h("span", { class: "tok static", style: `--tc: ${tc(w)}` }, word)), tds);
              tr.addEventListener("pointerenter", () => setHover(w));
              tr.addEventListener("pointerleave", () => setHover(null));
              return tr;
            })))))),
  );
  applyHover();
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
        scatter(P.E, { arrows, hvec: g.h }),
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
