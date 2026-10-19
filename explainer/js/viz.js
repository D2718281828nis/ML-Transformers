// DOM + drawing helpers shared by the sections in app.js.

import { T } from "./i18n.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  setProps(el, props);
  append(el, children);
  return el;
}

export function s(tag, props, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  setProps(el, props);
  append(el, children);
  return el;
}

function setProps(el, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Number formatting with a real minus sign and no "-0.000".
export function fmt(v, digits = 3) {
  if (v === -Infinity) return "−∞";
  if (v === Infinity) return "∞";
  if (v == null || Number.isNaN(v)) return "–";
  if (Math.abs(v) < 0.5 * 10 ** -digits) v = 0;
  const str = v.toFixed(digits);
  return str.startsWith("-") ? "−" + str.slice(1) : str;
}

export function pct(p, digits = 1) {
  return `${(p * 100).toFixed(digits)}%`;
}

export const int = (n) => T.int(n);

// ───────────── colors ─────────────
let palette = null;
export function resetPalette() {
  palette = null;
}
function colors() {
  if (palette) return palette;
  const cs = getComputedStyle(document.documentElement);
  const rgb = (name) => cs.getPropertyValue(name).trim().split(/[\s,]+/).map(Number);
  palette = { neg: rgb("--c-neg"), pos: rgb("--c-pos"), zero: rgb("--c-zero"), hi: rgb("--c-seq"), lo: rgb("--c-seq-lo") };
  return palette;
}
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export function divergingRGB(v, scale) {
  if (!Number.isFinite(v)) return colors().zero;
  const t = Math.max(-1, Math.min(1, v / scale));
  const c = colors();
  return mix(c.zero, t < 0 ? c.neg : c.pos, Math.abs(t));
}
export function sequentialRGB(p) {
  const c = colors();
  return mix(c.lo, c.hi, Math.max(0, Math.min(1, p)));
}
export const rgbCss = (c) => `rgb(${c[0] | 0} ${c[1] | 0} ${c[2] | 0})`;
export function isDark(c) {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 140;
}

// Color scale that ignores the top 1% of magnitudes, so one outlier dimension
// doesn't wash out everything else (those cells just saturate).
export function robustScale(arrays, q = 0.99) {
  const all = [];
  for (const a of arrays) for (const v of a) if (Number.isFinite(v)) all.push(Math.abs(v));
  if (!all.length) return 1;
  all.sort((a, b) => a - b);
  return all[Math.min(all.length - 1, Math.floor(q * all.length))] || 1e-6;
}

export function legend(scale, { digits = 2, label = "" } = {}) {
  const c = colors();
  const bar = h("span", {
    class: "bar",
    style: { background: `linear-gradient(to right, ${rgbCss(c.neg)}, ${rgbCss(c.zero)}, ${rgbCss(c.pos)})` },
  });
  return h("span", { class: "legend" }, label ? `${label} ` : null, fmt(-scale, digits), bar, `+${fmt(scale, digits)}`);
}

export function seqLegend(label = T.attnWeight) {
  const c = colors();
  const bar = h("span", { class: "bar", style: { background: `linear-gradient(to right, ${rgbCss(c.lo)}, ${rgbCss(c.hi)})` } });
  return h("span", { class: "legend" }, `${label} 0`, bar, "1");
}

// ───────────── tooltip ─────────────
let tipEl = null;
export function showTip(html, ev) {
  tipEl ||= document.getElementById("tooltip");
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const pad = 14;
  const r = tipEl.getBoundingClientRect();
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = Math.max(8, ev.clientX - r.width - pad);
  if (y + r.height > innerHeight - 8) y = Math.max(8, ev.clientY - r.height - pad);
  tipEl.style.left = `${x}px`;
  tipEl.style.top = `${y}px`;
}
export function hideTip() {
  tipEl ||= document.getElementById("tooltip");
  tipEl.hidden = true;
}

// ───────────── tokens ─────────────
export function tokText(text) {
  const frag = document.createDocumentFragment();
  for (const part of text.split(/([ \n\t\r])/)) {
    if (!part) continue;
    if (part === " ") frag.append(h("span", { class: "ws" }, "␣"));
    else if (part === "\n" || part === "\r") frag.append(h("span", { class: "ws" }, "↵"));
    else if (part === "\t") frag.append(h("span", { class: "ws" }, "⇥"));
    else frag.append(part);
  }
  return frag;
}

export const tokPlain = (text) => text.replace(/ /g, "␣").replace(/\n|\r/g, "↵").replace(/\t/g, "⇥");
export const tokTip = (tok) => `<span class="tt-tok">${escapeHtml(tokPlain(tok.text))}</span>`;

export function tokChip(tok, { showId = false, extra = null } = {}) {
  return h(
    "span",
    {
      class: `tok${tok.partial ? " partial" : ""}`,
      "data-tok": tok.index,
      style: `--tc: var(--tok${tok.index % 10})`,
      title: tok.partial ? T.partialTitle : null,
    },
    h("span", null, tokText(tok.text)),
    showId ? h("span", { class: "tid" }, tok.id) : null,
    extra,
  );
}

// ───────────── heat strips ─────────────
// rows: Float32Array[]; labels: Node[]; one canvas per row (1 px per dimension,
// scaled up with pixelated rendering). Hover reports (row, dim, value).
export function heatRows(rows, {
  labels = [],
  scale = robustScale(rows),
  rowHeight = 14,
  segment = null, // [lo, hi) dims to highlight, others are dimmed
  boundaries = 0, // draw a thin line every N dims
  describe = null, // (r, d, v) => html
  onPick = null, // (r, d) on click
  axis = true,
} = {}) {
  const len = rows[0]?.length || 0;
  const wrap = h("div", { class: "heat" });
  const overlay = h("div", { class: "heat-overlay", style: { gridRow: `1 / span ${rows.length}` } });
  const hair = h("div", { class: "hair", hidden: true });
  const canvases = rows.map((row, r) => {
    const cv = h("canvas", { class: "heat-row", width: len, height: 1, style: { height: `${rowHeight}px`, gridColumn: 2, gridRow: r + 1 } });
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(len, 1);
    for (let d = 0; d < len; d++) {
      const c = divergingRGB(row[d], scale);
      img.data[d * 4] = c[0];
      img.data[d * 4 + 1] = c[1];
      img.data[d * 4 + 2] = c[2];
      img.data[d * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const dimAt = (ev) => Math.max(0, Math.min(len - 1, Math.floor((ev.offsetX / cv.clientWidth) * len)));
    cv.addEventListener("pointermove", (ev) => {
      const d = dimAt(ev);
      hair.hidden = false;
      hair.style.left = `${((d + 0.5) / len) * 100}%`;
      if (describe) showTip(describe(r, d, row[d]), ev);
    });
    cv.addEventListener("pointerleave", () => {
      hair.hidden = true;
      hideTip();
    });
    if (onPick) cv.addEventListener("click", (ev) => onPick(r, dimAt(ev)));
    return cv;
  });
  rows.forEach((_, r) => {
    wrap.append(h("div", { class: "heat-label", style: { gridColumn: 1, gridRow: r + 1 } }, labels[r] || null));
  });
  wrap.append(...canvases);
  if (boundaries) {
    for (let b = boundaries; b < len; b += boundaries) overlay.append(h("div", { class: "bound", style: { left: `${(b / len) * 100}%` } }));
  }
  if (segment) {
    const [lo, hi] = segment;
    overlay.append(
      h("div", { class: "dim", style: { left: "0", width: `${(lo / len) * 100}%` } }),
      h("div", { class: "dim", style: { left: `${(hi / len) * 100}%`, right: "0" } }),
      h("div", { class: "seg", style: { left: `${(lo / len) * 100}%`, width: `${((hi - lo) / len) * 100}%` } }),
    );
  }
  overlay.append(hair);
  wrap.append(overlay);
  if (axis && rows.length) {
    wrap.append(h("div", { class: "heat-axis", style: { gridRow: rows.length + 1 } }, h("span", null, T.dim(0)), h("span", null, T.dim(len - 1))));
  }
  return wrap;
}

// Single image for a whole matrix (rows × cols), e.g. the position table.
// `bracket: [r0, r1)` marks a band of rows with a bar on the left edge.
export function heatImage(getRow, nRows, nCols, { scale, rowPx = 2, describe = null, bracket = null } = {}) {
  const cv = h("canvas", { class: "heat-row", width: nCols, height: nRows, style: { height: `${nRows * rowPx}px` } });
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(nCols, nRows);
  for (let r = 0; r < nRows; r++) {
    const row = getRow(r);
    for (let d = 0; d < nCols; d++) {
      const c = divergingRGB(row[d], scale);
      const o = (r * nCols + d) * 4;
      img.data[o] = c[0];
      img.data[o + 1] = c[1];
      img.data[o + 2] = c[2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  const wrap = h("div", { style: { position: "relative" } }, cv);
  if (bracket) {
    const [r0, r1] = bracket;
    wrap.append(h("div", { class: "row-bracket", style: { top: `${(r0 / nRows) * 100}%`, height: `${((r1 - r0) / nRows) * 100}%` } }));
  }
  if (describe) {
    cv.addEventListener("pointermove", (ev) => {
      const d = Math.max(0, Math.min(nCols - 1, Math.floor((ev.offsetX / cv.clientWidth) * nCols)));
      const r = Math.max(0, Math.min(nRows - 1, Math.floor((ev.offsetY / cv.clientHeight) * nRows)));
      showTip(describe(r, d, getRow(r)[d]), ev);
    });
    cv.addEventListener("pointerleave", hideTip);
  }
  return wrap;
}

// ───────────── n × m matrix of cells ─────────────
export function cellSize(n, maxWidth) {
  const avail = Math.min(maxWidth, Math.max(240, (document.documentElement.clientWidth || 800) - 150));
  return Math.max(9, Math.min(46, Math.floor(avail / Math.max(1, n)) - 2));
}

export function matrixGrid({ n, m = n, size, rowLabels, colLabels, corner = "", cell }) {
  // cell(i, j) -> { bg: rgb[], text?, cls?, tip?: html, onClick?, onEnter? }
  const grid = h("div", { class: "mgrid", style: { gridTemplateColumns: `max-content repeat(${m}, ${size}px)`, gridAutoRows: `${size}px` } });
  grid.append(h("div", { class: "corner", style: { gridRow: 1, height: "auto" } }, corner));
  for (let j = 0; j < m; j++) grid.append(h("div", { class: "clab" }, colLabels[j]));
  grid.style.gridTemplateRows = "auto";
  const showText = size >= 30;
  for (let i = 0; i < n; i++) {
    grid.append(h("div", { class: "rlab" }, rowLabels[i]));
    for (let j = 0; j < m; j++) {
      const c = cell(i, j);
      const el = h("div", {
        class: `cell ${c.cls || ""}${c.bg && isDark(c.bg) ? " light" : ""}`,
        style: { background: c.bg ? rgbCss(c.bg) : null },
      }, showText ? c.text ?? "" : "");
      if (c.tip) {
        el.addEventListener("pointermove", (ev) => showTip(c.tip, ev));
        el.addEventListener("pointerleave", hideTip);
      }
      if (c.onEnter) el.addEventListener("pointerenter", c.onEnter);
      if (c.onClick) el.addEventListener("click", c.onClick);
      grid.append(el);
    }
  }
  return h("div", { class: "scroll-x" }, grid);
}

// ───────────── bar chart for a short vector (64 dims) ─────────────
export function bars(values, { scale = robustScale([values], 1), height = 54, colorOf = null, highlight = -1, describe = null } = {}) {
  const n = values.length;
  const bw = 6;
  const W = n * bw;
  const mid = height / 2;
  const svg = s("svg", { class: "bars-svg", viewBox: `0 0 ${W} ${height}`, preserveAspectRatio: "none" });
  svg.append(s("line", { class: "zero", x1: 0, x2: W, y1: mid, y2: mid, "vector-effect": "non-scaling-stroke" }));
  for (let d = 0; d < n; d++) {
    const v = values[d];
    const len = Math.min(1, Math.abs(v) / scale) * (mid - 1);
    const y = v >= 0 ? mid - len : mid;
    const c = colorOf ? colorOf(d, v) : divergingRGB(v, scale);
    svg.append(s("rect", { x: d * bw + 0.5, y, width: bw - 1, height: Math.max(0.5, len), fill: rgbCss(c) }));
  }
  if (highlight >= 0) svg.append(s("rect", { class: "hl", x: highlight * bw, y: 0, width: bw, height, "vector-effect": "non-scaling-stroke" }));
  if (describe) {
    svg.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const d = Math.max(0, Math.min(n - 1, Math.floor(((ev.clientX - r.left) / r.width) * n)));
      showTip(describe(d, values[d]), ev);
    });
    svg.addEventListener("pointerleave", hideTip);
  }
  return svg;
}

// ───────────── line chart: a few series over an index (positions, rounds, …) ─────────────
const SERIES_COLORS = ["#6366f1", "#14b8a6", "#f59e0b", "#ec4899"];

export function lineChart(series, {
  height = 160,
  mark = 0, // shade indices 0..mark-1
  markLabel = T.yourSentence,
  positive = false, // y axis 0..max instead of −max..max
  xLabel = T.pos, // (i) => axis label at both ends
  tipLabel = T.position, // (i) => first tooltip line
  digits = 3,
  points = false,
  highlight = -1, // index to ring on every series
  ticks = false, // label every index (for unevenly spaced x values)
} = {}) {
  const N = series[0].values.length;
  const W = 600;
  const H = height;
  const [padL, padR, padT, padB] = [44, 10, 8, 18];
  let max = 0;
  for (const ser of series) for (const v of ser.values) max = Math.max(max, Math.abs(v));
  max ||= 1;
  const x = (i) => padL + (N > 1 ? i / (N - 1) : 0.5) * (W - padL - padR);
  const y = (v) => padT + (positive ? 1 - v / max : 1 - (v + max) / (2 * max)) * (H - padT - padB);
  const yfmt = (v) => (digits ? fmt(v, 2) : int(Math.round(v)));
  const svg = s("svg", { class: "line-svg", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": T.chartAria(series.map((x) => x.label).join(", ")) });
  if (mark) svg.append(s("rect", { class: "mark", x: x(0) - (mark === 1 ? 2 : 0), y: padT, width: Math.max(4, x(mark - 1) - x(0)), height: H - padT - padB }));
  svg.append(
    s("line", { class: "zero", x1: padL, x2: W - padR, y1: y(0), y2: y(0) }),
    s("text", { class: "ax", x: padL - 5, y: y(max) + 7, "text-anchor": "end" }, yfmt(max)),
    s("text", { class: "ax", x: padL - 5, y: y(0) + 3, "text-anchor": "end" }, "0"),
    positive ? null : s("text", { class: "ax", x: padL - 5, y: y(-max), "text-anchor": "end" }, yfmt(-max)),
    ...(ticks
      ? Array.from({ length: N }, (_, i) => s("text", { class: "ax", x: x(i), y: H - 3, "text-anchor": i === 0 ? "start" : i === N - 1 ? "end" : "middle" }, xLabel(i)))
      : [s("text", { class: "ax", x: x(0), y: H - 3 }, xLabel(0)), s("text", { class: "ax", x: x(N - 1), y: H - 3, "text-anchor": "end" }, xLabel(N - 1))]),
  );
  series.forEach((ser, k) => {
    const color = SERIES_COLORS[k % SERIES_COLORS.length];
    svg.append(s("polyline", {
      points: Array.from(ser.values, (v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "),
      stroke: color, fill: "none", "stroke-width": 1.8, "stroke-linejoin": "round",
    }));
    if (points) for (let i = 0; i < N; i++) svg.append(s("circle", { cx: x(i), cy: y(ser.values[i]), r: 3, fill: color }));
    if (highlight >= 0) svg.append(s("circle", { class: "ring", cx: x(highlight), cy: y(ser.values[highlight]), r: 7, stroke: color }));
  });
  const hair = s("line", { class: "hair", y1: padT, y2: H - padB, visibility: "hidden" });
  svg.append(hair);
  svg.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round(((((ev.clientX - r.left) / r.width) * W - padL) / (W - padL - padR)) * (N - 1));
    if (i < 0 || i >= N) return;
    hair.setAttribute("x1", x(i));
    hair.setAttribute("x2", x(i));
    hair.setAttribute("visibility", "visible");
    showTip(`${tipLabel(i)}<br>` + series.map((ser, k) => `<span style="color:${SERIES_COLORS[k]}">●</span> ${ser.label}: <b>${yfmt(ser.values[i])}</b>`).join("<br>"), ev);
  });
  svg.addEventListener("pointerleave", () => {
    hair.setAttribute("visibility", "hidden");
    hideTip();
  });
  return h("div", null,
    h("div", { class: "line-legend" },
      series.map((ser, k) => h("span", null, h("i", { style: { background: SERIES_COLORS[k % SERIES_COLORS.length] } }), ser.label)),
      mark ? h("span", null, h("i", { class: "mark-swatch" }), markLabel) : null),
    svg);
}

// ───────────── small shared UI pieces ─────────────
// replaceChildren() that skips null/false children (replaceChildren would print "null").
export const fill = (el, ...children) => el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
export const mono = (text) => h("span", { class: "mono" }, text);

// Play/pause that advances a step counter every 750 ms.
export function player() {
  let timer = null;
  return {
    get playing() {
      return timer !== null;
    },
    stop() {
      clearInterval(timer);
      timer = null;
    },
    toggle(get, set, K, render) {
      if (timer) {
        this.stop();
        return render();
      }
      if (get() >= K) set(0);
      timer = setInterval(() => {
        if (get() >= K) this.stop();
        else set(get() + 1);
        render();
      }, 750);
      render();
    },
  };
}

// ⏮ ◀ ▶/❚❚ ▶ ⏭ + slider for stepping through 0..K.
export function stepper({ k, K, go, playing, onPlay }) {
  const slider = h("input", { type: "range", min: 0, max: K, value: k, step: 1, "aria-label": T.aStep, disabled: K === 0 });
  slider.addEventListener("input", () => go(Number(slider.value)));
  return h("div", { class: "bpe-controls" },
    h("button", { class: "btn", type: "button", onclick: () => go(0), disabled: k === 0, "aria-label": T.aFirst }, "⏮"),
    h("button", { class: "btn", type: "button", onclick: () => go(k - 1), disabled: k === 0, "aria-label": T.aPrev }, "◀"),
    h("button", { class: "btn primary", type: "button", onclick: onPlay, disabled: K === 0 }, playing ? T.pause : T.play),
    h("button", { class: "btn", type: "button", onclick: () => go(k + 1), disabled: k === K, "aria-label": T.aNext }, "▶"),
    h("button", { class: "btn", type: "button", onclick: () => go(K), disabled: k === K, "aria-label": T.aLast }, "⏭"),
    slider);
}

// ───────────── attention lines (query tokens on the left, keys on the right) ─────────────
export function attentionLines(tokens, probs, focus, { onFocus = null } = {}) {
  const n = tokens.length;
  const rowH = n > 24 ? 16 : 22;
  const top = 26;
  const W = 420;
  const H = top + n * rowH + 6;
  const lx = 150;
  const rx = W - 150;
  const svg = s("svg", { class: "lines-svg", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": T.linesAria });
  svg.append(
    s("text", { class: "hdr", x: lx - 6, y: 10, "text-anchor": "end" }, T.linesFrom),
    s("text", { class: "hdr", x: rx + 6, y: 10 }, T.linesTo),
  );
  const y = (i) => top + i * rowH + rowH / 2;
  const paths = s("g");
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const p = probs[i][j];
      if (!(p > 0.004)) continue;
      const on = i === focus;
      paths.append(s("path", {
        d: `M ${lx} ${y(i)} C ${(lx + rx) / 2} ${y(i)}, ${(lx + rx) / 2} ${y(j)}, ${rx} ${y(j)}`,
        "stroke-width": on ? 1 + p * 9 : 0.6 + p * 3,
        "stroke-opacity": on ? 0.25 + 0.75 * p : 0.05 + 0.25 * p,
      }));
    }
  }
  svg.append(paths);
  const label = (tok, x, anchor, i) => {
    const txt = tokPlain(tok.text);
    const w = Math.min(130, 8 + txt.length * 7.3);
    const g = s("g", { "data-tok": tok.index });
    g.append(
      s("rect", { class: "tbg", x: anchor === "end" ? x - w : x, y: y(i) - rowH / 2 + 2, width: w, height: rowH - 4, rx: 4, style: `fill: var(--tok${tok.index % 10})` }),
      s("text", { x: anchor === "end" ? x - 4 : x + 4, y: y(i), "text-anchor": anchor, "font-weight": i === focus ? 700 : 400 }, txt.length > 16 ? txt.slice(0, 15) + "…" : txt),
    );
    return g;
  };
  tokens.forEach((t, i) => {
    const left = label(t, lx - 6, "end", i);
    if (onFocus) {
      left.style.cursor = "pointer";
      left.addEventListener("click", () => onFocus(i));
    }
    svg.append(left, label(t, rx + 6, "start", i));
  });
  return svg;
}
