// SVG line charts (with crosshair + tooltip), small sparkline charts and heatmap canvases.

import { h, s, fmt, showTip, hideTip, diverging, rgbCss, clamp } from "./util.js";

export function niceTicks(min, max, count = 5) {
  if (!(max > min)) return [min];
  const span = max - min;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => span / st <= count + 0.5) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function extent(arrays, pad = 0.06) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays) for (const v of a) if (Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi - lo < 1e-9) (lo -= 0.5), (hi += 0.5);
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

const pathOf = (xs, ys, X, Y) => {
  let d = "";
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(ys[i])) continue;
    d += `${d ? "L" : "M"}${X(xs[i]).toFixed(1)},${Y(ys[i]).toFixed(1)}`;
  }
  return d;
};

/**
 * series: [{ xs, ys, cls: "s1"|"s2"|"s3"|"ink"|"past", label, thin, opacity }]
 * bands:  [{ xs, lo, hi, color, opacity }]
 * marks:  [{ x, cls }] vertical rules;  hlines: [{ y, label }]
 */
export function lineChart({
  width, height = 220, series = [], bands = [], marks = [], hlines = [], x0, x1, y0, y1, xTicks, yTicks,
  xFmt = (v) => fmt(v, 0), yFmt = (v) => fmt(v, 1), xLabel = "", hover = true, tipFmt = (v) => fmt(v, 3), tipTitle,
  margin = { l: 42, r: 12, t: 8, b: xLabel ? 34 : 22 },
}) {
  const m = margin;
  const W = width;
  const H = height;
  const allX = series.map((q) => q.xs);
  const allY = [...series.map((q) => q.ys), ...bands.flatMap((b) => [b.lo, b.hi]), hlines.map((l) => l.y)];
  const [ex0, ex1] = extent(allX, 0);
  const [ey0, ey1] = extent(allY);
  x0 ??= ex0;
  x1 ??= ex1;
  y0 ??= ey0;
  y1 ??= ey1;
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);

  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
  const yt = yTicks || niceTicks(y0, y1, Math.max(2, Math.round(H / 55)));
  for (const v of yt) {
    if (v < y0 || v > y1) continue;
    svg.append(s("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }));
    svg.append(s("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, yFmt(v)));
  }
  const xt = xTicks || niceTicks(x0, x1, Math.max(2, Math.round(W / 90)));
  svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
  for (const v of xt) {
    if (v < x0 - 1e-9 || v > x1 + 1e-9) continue;
    svg.append(s("text", { x: X(v), y: H - m.b + 14, "text-anchor": "middle" }, xFmt(v)));
  }
  if (xLabel) svg.append(s("text", { class: "lbl", x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle" }, xLabel));

  for (const b of bands) {
    let d = "";
    for (let i = 0; i < b.xs.length; i++) d += `${i ? "L" : "M"}${X(b.xs[i]).toFixed(1)},${Y(b.hi[i]).toFixed(1)}`;
    for (let i = b.xs.length - 1; i >= 0; i--) d += `L${X(b.xs[i]).toFixed(1)},${Y(b.lo[i]).toFixed(1)}`;
    svg.append(s("path", { class: "band", d: d + "Z", fill: b.color, "fill-opacity": b.opacity ?? 0.2 }));
  }
  for (const mk of marks) svg.append(s("line", { class: mk.cls || "now", x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b }));
  for (const l of hlines) {
    svg.append(s("line", { class: "axis", x1: m.l, x2: W - m.r, y1: Y(l.y), y2: Y(l.y) }));
    if (l.label) svg.append(s("text", { class: "lbl", x: W - m.r - 2, y: Y(l.y) - 4, "text-anchor": "end" }, l.label));
  }
  const paths = series.map((q) => {
    const p = s("path", { class: `line ${q.cls || "s1"}${q.thin ? " thin" : ""}`, d: pathOf(q.xs, q.ys, X, Y) });
    if (q.opacity != null) p.setAttribute("stroke-opacity", q.opacity);
    svg.append(p);
    return p;
  });

  if (hover && series.length) {
    const hair = s("line", { class: "hair", y1: m.t, y2: H - m.b, visibility: "hidden" });
    const dots = series.map((q) => s("circle", { class: `dot`, r: 4, visibility: "hidden", fill: `var(--${q.cls === "past" ? "muted" : q.cls === "ink" ? "ink" : q.cls || "s1"})` }));
    svg.append(hair, ...dots);
    const hit = s("rect", { class: "hit", x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b });
    svg.append(hit);
    const move = (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const xv = x0 + ((px - m.l) / (W - m.l - m.r)) * (x1 - x0);
      const rows = [];
      let hx = null;
      series.forEach((q, si) => {
        if (q.noTip) return dots[si].setAttribute("visibility", "hidden");
        const i = nearestIndex(q.xs, xv);
        if (i < 0 || Math.abs(q.xs[i] - xv) > (x1 - x0) / 40 + 1e-9 || !Number.isFinite(q.ys[i])) {
          dots[si].setAttribute("visibility", "hidden");
          return;
        }
        hx ??= q.xs[i];
        dots[si].setAttribute("cx", X(q.xs[i]));
        dots[si].setAttribute("cy", Y(q.ys[i]));
        dots[si].setAttribute("visibility", "visible");
        rows.push(`<div>${q.label ?? ""}: <span class="mono">${tipFmt(q.ys[i])}</span></div>`);
      });
      if (hx == null) return leave();
      hair.setAttribute("x1", X(hx));
      hair.setAttribute("x2", X(hx));
      hair.setAttribute("visibility", "visible");
      showTip(`${tipTitle ? `<b>${tipTitle(hx)}</b>` : ""}${rows.join("")}`, ev);
    };
    const leave = () => {
      hair.setAttribute("visibility", "hidden");
      for (const d of dots) d.setAttribute("visibility", "hidden");
      hideTip();
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", leave);
  }
  svg.scales = { X, Y, x0, x1, y0, y1 };
  svg.paths = paths;
  return svg;
}

function nearestIndex(xs, v) {
  let lo = 0;
  let hi = xs.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < v) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
}

export const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);

// Tiny curve with a zero line, for small multiples.
export function spark(ys, { cls = "s1", width = 120, height = 40, scale } = {}) {
  const n = ys.length;
  const lim = scale ?? Math.max(1e-9, ...Array.from(ys, Math.abs));
  const X = (i) => (i / (n - 1)) * width;
  const Y = (v) => height / 2 - (clamp(v / lim, -1, 1) * (height / 2 - 2));
  let d = "";
  for (let i = 0; i < n; i++) d += `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(ys[i]).toFixed(1)}`;
  return s("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none" },
    s("line", { x1: 0, x2: width, y1: height / 2, y2: height / 2, stroke: "var(--line-strong)", "stroke-width": 1, "vector-effect": "non-scaling-stroke" }),
    s("path", { d, fill: "none", stroke: `var(--${cls})`, "stroke-width": 1.6, "vector-effect": "non-scaling-stroke" }));
}

/** Heatmap of a rows × cols matrix (row-major) on a canvas, diverging colors. */
export function heatmap(M, rows, cols, scale, { height = 220, onHover, onLeave, onClick } = {}) {
  const canvas = h("canvas", { class: "heat", width: cols, height: rows });
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const c = diverging(M[i], scale);
    img.data[i * 4] = c[0];
    img.data[i * 4 + 1] = c[1];
    img.data[i * 4 + 2] = c[2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const at = (ev) => {
    const r = canvas.getBoundingClientRect();
    const c = clamp(Math.floor(((ev.clientX - r.left) / r.width) * cols), 0, cols - 1);
    const rr = clamp(Math.floor(((ev.clientY - r.top) / r.height) * rows), 0, rows - 1);
    return [rr, c];
  };
  if (onHover) canvas.addEventListener("pointermove", (ev) => onHover(...at(ev), ev));
  canvas.addEventListener("pointerleave", () => (onLeave ? onLeave() : hideTip()));
  if (onClick) canvas.addEventListener("click", (ev) => onClick(...at(ev), ev));
  return canvas;
}

export function swatchCss(cls) {
  return `bg-${cls}`;
}

export function legendRow(items) {
  return h("div", { class: "legend-row" }, items.map(([cls, label, kind]) =>
    h("span", null, h("i", { class: `${kind || ""} bg-${cls}`, style: kind === "band" ? { opacity: 0.45 } : null }), label)));
}

export { rgbCss };
