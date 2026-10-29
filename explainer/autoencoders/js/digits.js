// Parts 2 and 3 — the digits autoencoder and the VAEs: latent maps, drawing, the decoder's map,
// the KL term, the reparameterization trick, sampling, β and interpolation.

import {
  h, fmt, pct, clamp, colors, rgbCss, fitCanvas, widthOf, rng, segToggle, ctl, showTip, hideTip, drawDigit, digitCanvas, f32,
} from "./util.js";
import { sequential, sigmoid } from "./nn.js";
import { lineChart, legendRow } from "./charts.js";
import { T } from "./i18n.js";

const ALPHABET = "0123456789abcdefg";
const MONO = (typeof document !== "undefined" && getComputedStyle(document.documentElement).getPropertyValue("--mono").trim()) || "monospace";
export const SIDE = 64; // the models see 64 × 64 images
const NPX = SIDE * SIDE;

// ───────────── data & models ─────────────
// The images ship at MNIST's 28 × 28 and are resized like the training data (tools/build_assets.py).
// Each model ships the codes of every image, so the page never has to run 2000 images through five encoders.
export function prepareDigits(dj, mj) {
  const n = dj.n;
  const side = dj.side;
  const small = new Float32Array(n * side * side);
  for (let i = 0; i < small.length; i++) small[i] = ALPHABET.indexOf(dj.images[i]) / 16;
  const labels = Uint8Array.from(dj.labels, Number);
  const layers = (ls) => ls.map((l) => ({ type: l.type, W: l.w && f32(l.w), b: l.b && f32(l.b), stride: l.stride, pad: l.pad, shape: l.shape }));
  const models = {};
  for (const [key, m] of Object.entries(mj.models)) {
    models[key] = {
      key, kind: m.kind, beta: m.beta, metrics: m.metrics, enc: layers(m.encoder), dec: layers(m.decoder),
      codes: f32(m.codes), logvar: m.logvar ? f32(m.logvar) : null,
    };
  }
  return { n, side, small, labels, models, cache: {}, images: new Map() };
}

// F.interpolate(mode="bilinear", align_corners=False) from a square sIn image to sOut, then clamped to [0, 1]
export function resize(src, sIn, sOut, out = new Float32Array(sOut * sOut)) {
  const scale = sIn / sOut;
  const axis = Array.from({ length: sOut }, (_, o) => {
    const x = Math.max(0, (o + 0.5) * scale - 0.5);
    const x0 = Math.min(Math.floor(x), sIn - 1);
    return [x0, Math.min(x0 + 1, sIn - 1), x - x0];
  });
  for (let oy = 0; oy < sOut; oy++) {
    const [y0, y1, ty] = axis[oy];
    for (let ox = 0; ox < sOut; ox++) {
      const [x0, x1, tx] = axis[ox];
      const top = (1 - tx) * src[y0 * sIn + x0] + tx * src[y0 * sIn + x1];
      const bot = (1 - tx) * src[y1 * sIn + x0] + tx * src[y1 * sIn + x1];
      out[oy * sOut + ox] = clamp((1 - ty) * top + ty * bot, 0, 1);
    }
  }
  return out;
}

// image i at 64 × 64 (a few hundred are kept around)
export function img(D, i) {
  let im = D.images.get(i);
  if (!im) {
    if (D.images.size > 400) D.images.clear();
    im = resize(D.small.subarray(i * D.side * D.side, (i + 1) * D.side * D.side), D.side, SIDE);
    D.images.set(i, im);
  }
  return im;
}

export function encode(model, x) {
  const out = sequential(model.enc, x, [1, SIDE, SIDE]);
  return model.kind === "vae" ? { mu: out.subarray(0, 2), logvar: out.subarray(2, 4) } : { mu: out, logvar: null };
}
export function decode(model, z) {
  const out = sequential(model.dec, z);
  for (let i = 0; i < out.length; i++) out[i] = sigmoid(out[i]);
  return out;
}

// codes (μ for a VAE) and log-variances of every image, with their bounding box
function codes(D, key) {
  if (D.cache[key]) return D.cache[key];
  const { codes: mu, logvar } = D.models[key];
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < D.n; i++) {
    x0 = Math.min(x0, mu[2 * i]);
    x1 = Math.max(x1, mu[2 * i]);
    y0 = Math.min(y0, mu[2 * i + 1]);
    y1 = Math.max(y1, mu[2 * i + 1]);
  }
  return (D.cache[key] = { mu, logvar, box: [x0, x1, y0, y1] });
}

function padBox([x0, x1, y0, y1], p = 0.06) {
  const dx = (x1 - x0) * p;
  const dy = (y1 - y0) * p;
  return [x0 - dx, x1 + dx, y0 - dy, y1 + dy];
}

// share of uniformly drawn latent points whose nearest training code belongs to each digit
function territory(D, key, sampler, n = 3000) {
  const { mu } = codes(D, key);
  const counts = new Float64Array(10);
  const r = rng(11);
  for (let s = 0; s < n; s++) {
    const [a, b] = sampler(r);
    let best = 0;
    let bd = Infinity;
    for (let i = 0; i < D.n; i++) {
      const d = (mu[2 * i] - a) ** 2 + (mu[2 * i + 1] - b) ** 2;
      if (d < bd) (bd = d), (best = i);
    }
    counts[D.labels[best]]++;
  }
  return Array.from(counts, (c) => c / n);
}

// ───────────── latent map component ─────────────
// Each image is drawn as its own digit at its code; identity never depends on color.
function latentMap(D, opts) {
  const canvas = h("canvas", { class: "plot crosshair", role: "img", "aria-label": opts.label || T.latentMapLabel });
  const st = { geo: null, hover: -1, mapCache: null };

  function layout() {
    const W = Math.max(220, Math.floor(canvas.parentElement ? canvas.parentElement.getBoundingClientRect().width : 400));
    const H = Math.round(clamp(W * (opts.aspect ?? 0.78), 220, opts.maxH ?? 460));
    const ctx = fitCanvas(canvas, W, H);
    canvas.style.width = `${W}px`;
    // one scale for both axes, so distances (and the prior's circles) are not distorted
    let [x0, x1, y0, y1] = opts.bounds();
    const s = Math.max((x1 - x0) / W, (y1 - y0) / H);
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    [x0, x1, y0, y1] = [cx - (s * W) / 2, cx + (s * W) / 2, cy - (s * H) / 2, cy + (s * H) / 2];
    st.geo = { W, H, ctx, x0, x1, y0, y1 };
  }
  const toPx = (x, y) => {
    const g = st.geo;
    return [((x - g.x0) / (g.x1 - g.x0)) * g.W, g.H - ((y - g.y0) / (g.y1 - g.y0)) * g.H];
  };
  const toZ = (px, py) => {
    const g = st.geo;
    return [g.x0 + (px / g.W) * (g.x1 - g.x0), g.y0 + ((g.H - py) / g.H) * (g.y1 - g.y0)];
  };

  function decoderMap(model) {
    const g = st.geo;
    const key = `${model.key}:${g.W}:${g.H}:${g.x0}:${g.y0}`;
    if (st.mapCache?.key === key) return st.mapCache.canvas;
    const cell = 34;
    const cols = Math.max(4, Math.floor(g.W / cell));
    const rows = Math.max(4, Math.floor(g.H / cell));
    const off = h("canvas", { width: cols * SIDE, height: rows * SIDE });
    const octx = off.getContext("2d");
    const img = octx.createImageData(cols * SIDE, rows * SIDE);
    const c = colors();
    for (let r = 0; r < rows; r++) {
      for (let q = 0; q < cols; q++) {
        const z = toZ(((q + 0.5) / cols) * g.W, ((r + 0.5) / rows) * g.H);
        paintCell(img, decode(model, z), q * SIDE, r * SIDE, c);
      }
    }
    octx.putImageData(img, 0, 0);
    st.mapCache = { key, canvas: off };
    return off;
  }

  function draw() {
    if (!st.geo) layout();
    const g = st.geo;
    const { ctx } = g;
    const c = colors();
    const key = opts.modelKey();
    const { mu } = codes(D, key);
    ctx.clearRect(0, 0, g.W, g.H);
    if (opts.showMap?.()) {
      ctx.save();
      ctx.imageSmoothingQuality = "high";
      ctx.globalAlpha = 0.55;
      ctx.drawImage(decoderMap(D.models[key]), 0, 0, g.W, g.H);
      ctx.restore();
    }
    // integer grid
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.fillStyle = c.muted;
    ctx.font = `10px ${MONO}`;
    const step = niceStep(g.x1 - g.x0);
    ctx.beginPath();
    for (let v = Math.ceil(g.x0 / step) * step; v <= g.x1; v += step) {
      const [x] = toPx(v, 0);
      ctx.moveTo(x, 0);
      ctx.lineTo(x, g.H);
    }
    for (let v = Math.ceil(g.y0 / step) * step; v <= g.y1; v += step) {
      const [, y] = toPx(0, v);
      ctx.moveTo(0, y);
      ctx.lineTo(g.W, y);
    }
    ctx.stroke();
    for (let v = Math.ceil(g.x0 / step) * step; v <= g.x1; v += step) ctx.fillText(fmt(v, step < 1 ? 1 : 0), toPx(v, 0)[0] + 2, g.H - 4);
    for (let v = Math.ceil(g.y0 / step) * step; v <= g.y1; v += step) if (Math.abs(v) > 1e-9) ctx.fillText(fmt(v, step < 1 ? 1 : 0), 3, toPx(0, v)[1] - 3);
    opts.underlay?.(ctx, toPx);
    // glyphs
    const emph = opts.emph?.() ?? null;
    ctx.font = `600 ${opts.glyph ?? 10}px ${MONO}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const ink = c.ink;
    for (let i = 0; i < D.n; i++) {
      const d = D.labels[i];
      const on = emph == null || emph === d;
      ctx.fillStyle = on && emph != null ? c.s1 : rgbCss(ink, emph == null ? opts.alpha ?? 0.5 : 0.12);
      const [x, y] = toPx(mu[2 * i], mu[2 * i + 1]);
      ctx.fillText(String(d), x, y);
    }
    if (st.hover >= 0) {
      const [x, y] = toPx(mu[2 * st.hover], mu[2 * st.hover + 1]);
      ctx.strokeStyle = c.inkCss;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.stroke();
    }
    opts.overlay?.(ctx, toPx);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  function nearest(mx, my, radius = 9) {
    const { mu } = codes(D, opts.modelKey());
    let best = -1;
    let bd = radius * radius;
    for (let i = 0; i < D.n; i++) {
      const [x, y] = toPx(mu[2 * i], mu[2 * i + 1]);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd) (bd = d), (best = i);
    }
    return best;
  }
  const local = (ev) => {
    const r = canvas.getBoundingClientRect();
    return [ev.clientX - r.left, ev.clientY - r.top];
  };
  canvas.addEventListener("pointermove", (ev) => {
    const [mx, my] = local(ev);
    const i = nearest(mx, my);
    if (i !== st.hover) {
      st.hover = i;
      draw();
    }
    opts.onMove?.(i, toZ(mx, my), ev);
  });
  canvas.addEventListener("pointerleave", () => {
    st.hover = -1;
    hideTip();
    draw();
    opts.onLeave?.();
  });
  canvas.addEventListener("click", (ev) => {
    const [mx, my] = local(ev);
    opts.onClick?.(nearest(mx, my, 14), toZ(mx, my), ev);
  });

  return {
    el: canvas,
    draw,
    relayout() {
      layout();
      st.mapCache = null;
      draw();
    },
    toPx: (x, y) => toPx(x, y),
  };
}

// write a 64 × 64 image into an ImageData at (x, y)
function paintCell(im, px, x, y, c) {
  for (let p = 0; p < NPX; p++) {
    const v = px[p];
    const o = ((y + Math.floor(p / SIDE)) * im.width + x + (p % SIDE)) * 4;
    im.data[o] = c.paper[0] + (c.ink[0] - c.paper[0]) * v;
    im.data[o + 1] = c.paper[1] + (c.ink[1] - c.paper[1]) * v;
    im.data[o + 2] = c.paper[2] + (c.ink[2] - c.paper[2]) * v;
    im.data[o + 3] = 255;
  }
}

function niceStep(span) {
  const raw = span / 6;
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
}

const firstOf = (D, digit, k = 0) => {
  let seen = 0;
  for (let i = 0; i < D.n; i++) if (D.labels[i] === digit && seen++ === k) return i;
  return 0;
};

function bigDigit(caption) {
  const c = h("canvas", { class: "digit digit-big", width: SIDE, height: SIDE });
  return { canvas: c, fig: h("figure", null, c, h("figcaption", null, caption)) };
}

function digitPick(current, onPick, withAll = true) {
  const wrap = h("div", { class: "digit-pick", role: "group" });
  const opts = [...(withAll ? [[null, T.all]] : []), ...Array.from({ length: 10 }, (_, d) => [d, String(d)])];
  for (const [v, label] of opts) {
    const b = h("button", { type: "button", "aria-pressed": String(v === current) }, label);
    if (v === null) b.style.padding = "3px 8px";
    b.addEventListener("click", () => {
      for (const x of wrap.children) x.setAttribute("aria-pressed", String(x === b));
      onPick(v);
    });
    wrap.append(b);
  }
  return wrap;
}

function strip(images, cls = "") {
  return h("div", { class: `strip ${cls}` }, images.map((im) => digitCanvas(im)));
}

function territoryBars(series, W = 520) {
  // series: [{label, cls, values[10]}] → grouped vertical bars (SVG)
  const H = 150;
  const m = { l: 34, r: 8, t: 8, b: 22 };
  const max = Math.max(0.25, ...series.flatMap((s) => s.values));
  const bw = (W - m.l - m.r) / 10;
  const Y = (v) => H - m.b - (v / max) * (H - m.t - m.b);
  const sv = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  sv.setAttribute("viewBox", `0 0 ${W} ${H}`);
  sv.setAttribute("class", "chart");
  const add = (tag, attrs, text) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    sv.append(e);
    return e;
  };
  for (const v of [0, 0.1, 0.2]) {
    add("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) });
    add("text", { x: m.l - 5, y: Y(v) + 3.5, "text-anchor": "end" }, pct(v));
  }
  add("line", { class: "axis", x1: m.l, x2: W - m.r, y1: Y(0.1), y2: Y(0.1), "stroke-width": 1.5 });
  const gw = (bw - 6) / series.length;
  for (let d = 0; d < 10; d++) {
    add("text", { x: m.l + bw * d + bw / 2, y: H - 6, "text-anchor": "middle" }, String(d));
    series.forEach((s, k) => {
      const x = m.l + bw * d + 3 + k * gw;
      const v = s.values[d];
      const r = add("rect", { x: x + 1, y: Y(v), width: Math.max(1, gw - 2), height: Math.max(0, Y(0) - Y(v)), rx: 2, fill: `var(--${s.cls})` });
      r.addEventListener("pointermove", (ev) => showTip(`${s.label}<br>${T.digit} ${d}: <span class="mono">${pct(v, 1)}</span>`, ev));
      r.addEventListener("pointerleave", hideTip);
    });
  }
  return h("div", null, legendRow(series.map((s) => [s.cls, s.label])), sv);
}

// ───────────── sections ─────────────
export function initDigits(D, roots) {
  const parts = [
    mapSection(D, roots.map),
    drawSection(D, roots.draw),
    gridSection(D, roots.grid),
    klSection(roots.kl),
    cloudSection(D, roots.cloud),
    vaeMapSection(D, roots.vaeMap),
    betaSection(D, roots.beta),
    interpSection(D, roots.interp),
  ];
  return { redraw: () => parts.forEach((p) => p?.redraw()) };
}

// 2.1
function mapSection(D, root) {
  const st = { emph: null, shown: firstOf(D, 4) };
  const ae = D.models.ae;
  const box = () => padBox(codes(D, "ae").box);
  const big = bigDigit(T.decodedHere);
  const orig = bigDigit(T.original);
  const recon = bigDigit(T.reconstruction);
  const zLine = h("div", { class: "sub" });
  const map = latentMap(D, {
    modelKey: () => "ae", bounds: box, emph: () => st.emph,
    onMove(i, z, ev) {
      if (i >= 0) {
        st.shown = i;
        showPoint(i);
        showTip(`${T.image} #${i} · ${T.digit} ${D.labels[i]}`, ev);
      } else {
        hideTip();
        drawDigit(big.canvas, decode(ae, z));
        zLine.textContent = `${T.decodedAt} z = (${fmt(z[0], 2)}, ${fmt(z[1], 2)})`;
      }
    },
  });
  function showPoint(i) {
    const { mu } = codes(D, "ae");
    const z = mu.subarray(2 * i, 2 * i + 2);
    const rec = decode(ae, z);
    drawDigit(orig.canvas, img(D, i));
    drawDigit(recon.canvas, rec);
    drawDigit(big.canvas, rec);
    zLine.textContent = `${T.image} #${i} (${T.digit} ${D.labels[i]}) → z = (${fmt(z[0], 2)}, ${fmt(z[1], 2)})`;
  }
  root.replaceChildren(
    h("div", { class: "viz-row" },
      h("div", { class: "main-plot" }, map.el),
      h("div", { class: "side" },
        h("div", null, h("div", { class: "panel-title" }, T.highlight), digitPick(null, (d) => {
          st.emph = d;
          map.draw();
        })),
        h("div", { class: "digit-pair" }, orig.fig, recon.fig),
        zLine,
        h("div", { class: "digit-pair" }, big.fig),
        h("p", { class: "sub" }, T.mapHint, " ", T.aeMetrics(ae.metrics)))),
  );
  requestAnimationFrame(() => {
    map.relayout();
    showPoint(st.shown);
  });
  return { redraw: () => (map.relayout(), showPoint(st.shown)) };
}

// 2.2
// MNIST's digits were scaled to fit a 20 × 20 box and centered by their center of mass in 28 × 28;
// at 64 × 64 that box is 46 pixels wide. Drawings get the same treatment before the encoder sees them.
const BOX = Math.round((20 / 28) * SIDE);

function mnistNormalize(src, out = new Float32Array(NPX)) {
  let [x0, x1, y0, y1, m, mx, my] = [SIDE, -1, SIDE, -1, 0, 0, 0];
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      const v = src[y * SIDE + x];
      if (v > 0.05) (x0 = Math.min(x0, x)), (x1 = Math.max(x1, x)), (y0 = Math.min(y0, y)), (y1 = Math.max(y1, y));
      (m += v), (mx += v * (x + 0.5)), (my += v * (y + 0.5));
    }
  }
  out.fill(0);
  if (x1 < 0 || m < 1e-6) return out;
  const s = BOX / Math.max(x1 - x0 + 1, y1 - y0 + 1);
  const [cx, cy] = [mx / m, my / m];
  const at = (x, y) => (x < 0 || y < 0 || x >= SIDE || y >= SIDE ? 0 : src[y * SIDE + x]);
  const bilinear = (x, y) => {
    const [fx, fy] = [Math.floor(x), Math.floor(y)];
    const [tx, ty] = [x - fx, y - fy];
    return (1 - ty) * ((1 - tx) * at(fx, fy) + tx * at(fx + 1, fy)) + ty * ((1 - tx) * at(fx, fy + 1) + tx * at(fx + 1, fy + 1));
  };
  // 2 × 2 samples per output pixel, so shrinking a big drawing doesn't alias
  for (let v = 0; v < SIDE; v++) {
    for (let u = 0; u < SIDE; u++) {
      let acc = 0;
      for (const [du, dv] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) {
        acc += bilinear(cx + (u + du - SIDE / 2) / s - 0.5, cy + (v + dv - SIDE / 2) / s - 0.5);
      }
      out[v * SIDE + u] = clamp(acc / 4, 0, 1);
    }
  }
  return out;
}

function drawSection(D, root) {
  const pad = new Float32Array(NPX);
  const input = new Float32Array(NPX);
  const st = { erase: false, painting: false, last: null, normalize: true };
  const ae = D.models.ae;
  const vae = D.models["vae-1"];
  const paint = h("canvas", { class: "digit paint", width: SIDE, height: SIDE, "aria-label": T.paintLabel });
  const seen = bigDigit(T.encoderSees);
  const recAE = bigDigit(T.aeRebuilds);
  const recVAE = bigDigit(T.vaeRebuilds);
  const verdict = h("p", { class: "sub" });
  let mine = null;
  const map = latentMap(D, {
    modelKey: () => "ae", bounds: () => padBox(codes(D, "ae").box), alpha: 0.28, aspect: 0.62, maxH: 320,
    overlay(ctx, toPx) {
      if (!mine) return;
      const [x, y] = toPx(mine[0], mine[1]);
      const c = colors();
      ctx.fillStyle = c.s2;
      ctx.strokeStyle = c.surface;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    },
  });
  function update() {
    drawDigit(paint, pad);
    if (st.normalize) mnistNormalize(pad, input);
    else input.set(pad);
    drawDigit(seen.canvas, input);
    const z = encode(ae, input).mu;
    mine = [z[0], z[1]];
    drawDigit(recAE.canvas, decode(ae, z));
    drawDigit(recVAE.canvas, decode(vae, encode(vae, input).mu));
    // vote of the 10 nearest training codes
    const { mu } = codes(D, "ae");
    const near = [];
    for (let i = 0; i < D.n; i++) near.push([(mu[2 * i] - z[0]) ** 2 + (mu[2 * i + 1] - z[1]) ** 2, D.labels[i]]);
    near.sort((a, b) => a[0] - b[0]);
    const votes = new Array(10).fill(0);
    for (const [, l] of near.slice(0, 10)) votes[l]++;
    const top = votes.indexOf(Math.max(...votes));
    const empty = pad.every((v) => v < 1e-3);
    verdict.textContent = empty ? T.drawEmpty : T.drawVerdict(top, votes[top]);
    map.draw();
  }
  // a round, anti-aliased brush stamped along the segment since the last pointer event
  const at = (ev) => {
    const r = paint.getBoundingClientRect();
    return [((ev.clientX - r.left) / r.width) * SIDE, ((ev.clientY - r.top) / r.height) * SIDE];
  };
  function stamp(cx, cy) {
    const rad = st.erase ? 6 : 3.2;
    for (let y = Math.max(0, Math.floor(cy - rad - 1)); y <= Math.min(SIDE - 1, Math.ceil(cy + rad + 1)); y++) {
      for (let x = Math.max(0, Math.floor(cx - rad - 1)); x <= Math.min(SIDE - 1, Math.ceil(cx + rad + 1)); x++) {
        const v = clamp(rad + 0.5 - Math.hypot(x + 0.5 - cx, y + 0.5 - cy), 0, 1);
        const k = y * SIDE + x;
        pad[k] = st.erase ? Math.min(pad[k], 1 - v) : Math.max(pad[k], v);
      }
    }
  }
  function stroke(ev) {
    const [x, y] = at(ev);
    const [x0, y0] = st.last || [x, y];
    const steps = Math.max(1, Math.ceil(Math.hypot(x - x0, y - y0)));
    for (let s = 1; s <= steps; s++) stamp(x0 + ((x - x0) * s) / steps, y0 + ((y - y0) * s) / steps);
    st.last = [x, y];
    update();
  }
  const stop = () => ((st.painting = false), (st.last = null));
  paint.addEventListener("pointerdown", (ev) => {
    st.painting = true;
    st.last = null;
    paint.setPointerCapture(ev.pointerId);
    stroke(ev);
  });
  paint.addEventListener("pointermove", (ev) => st.painting && stroke(ev));
  paint.addEventListener("pointerup", stop);
  paint.addEventListener("pointercancel", stop);
  let sample = 0;
  const load = () => {
    pad.set(img(D, (sample++ * 37) % D.n));
    update();
  };
  const normToggle = h("input", { type: "checkbox", checked: true });
  normToggle.addEventListener("change", () => ((st.normalize = normToggle.checked), update()));
  root.replaceChildren(
    h("div", { class: "controls" },
      segToggle([{ value: false, label: T.brush }, { value: true, label: T.eraser }], false, (v) => (st.erase = v), T.tool),
      h("button", { class: "btn", type: "button", onclick: () => (pad.fill(0), update()) }, T.clear),
      h("button", { class: "btn", type: "button", onclick: load }, T.loadTestDigit),
      h("label", { class: "toggle" }, normToggle, T.normalizeLikeMnist)),
    h("div", { class: "viz-row" },
      h("div", { class: "paint-wrap" },
        h("div", { class: "digit-pair" }, h("figure", null, paint, h("figcaption", null, T.paintCaption))),
        h("div", { class: "digit-pair" }, seen.fig, recAE.fig, recVAE.fig)),
      h("div", { class: "grow" }, h("div", { class: "panel-title" }, T.whereItLands), map.el, verdict)),
  );
  load();
  requestAnimationFrame(() => map.relayout());
  return { redraw: () => (map.relayout(), update()) };
}

// 2.3
function gridSection(D, root) {
  const ae = D.models.ae;
  const n = 15;
  const cell = SIDE + 2;
  const mosaic = h("canvas", { class: "digit", width: n * cell + 2, height: n * cell + 2, style: { width: "100%", maxWidth: "440px", height: "auto", aspectRatio: "1", borderRadius: "8px" } });
  const [x0, x1, y0, y1] = codes(D, "ae").box;
  function drawMosaic() {
    const c = colors();
    const ctx = mosaic.getContext("2d");
    const im = ctx.createImageData(mosaic.width, mosaic.height);
    for (let k = 0; k < im.data.length; k += 4) (im.data[k] = c.paper[0] * 0.92 + c.ink[0] * 0.08), (im.data[k + 1] = c.paper[1] * 0.92 + c.ink[1] * 0.08), (im.data[k + 2] = c.paper[2] * 0.92 + c.ink[2] * 0.08), (im.data[k + 3] = 255);
    for (let r = 0; r < n; r++) {
      for (let q = 0; q < n; q++) {
        const z = [x0 + ((x1 - x0) * q) / (n - 1), y1 - ((y1 - y0) * r) / (n - 1)];
        paintCell(im, decode(ae, z), 2 + q * cell, 2 + r * cell, c);
      }
    }
    ctx.putImageData(im, 0, 0);
  }
  mosaic.addEventListener("pointermove", (ev) => {
    const r = mosaic.getBoundingClientRect();
    const q = clamp(Math.floor(((ev.clientX - r.left) / r.width) * n), 0, n - 1);
    const rr = clamp(Math.floor(((ev.clientY - r.top) / r.height) * n), 0, n - 1);
    showTip(`z = <span class="mono">(${fmt(x0 + ((x1 - x0) * q) / (n - 1), 1)}, ${fmt(y1 - ((y1 - y0) * rr) / (n - 1), 1)})</span>`, ev);
  });
  mosaic.addEventListener("pointerleave", hideTip);
  const share = territory(D, "ae", (r) => [x0 + (x1 - x0) * r.uniform(), y0 + (y1 - y0) * r.uniform()]);
  const sorted = share.map((v, d) => [v, d]).sort((a, b) => b[0] - a[0]);
  root.replaceChildren(h("div", { class: "viz-row" },
    h("div", { style: { flex: "1 1 300px", maxWidth: "440px" } }, h("div", { class: "panel-title" }, T.gridTitle(n)), mosaic),
    h("div", { class: "grow" }, h("div", { class: "panel-title" }, T.territoryTitle),
      territoryBars([{ label: T.aeUniformBox, cls: "s1", values: share }]),
      h("p", { class: "sub" }, T.territoryNote(sorted[0], sorted[sorted.length - 1])))));
  drawMosaic();
  return { redraw: drawMosaic };
}

// 3.1
function klSection(root) {
  const st = { mu: 1, logSigma: Math.log(0.5) };
  const chartBox = h("div");
  const terms = h("div", { class: "formula" });
  const muIn = h("input", { type: "range", min: -3, max: 3, step: 0.05, value: st.mu, "aria-label": "μ" });
  const sgIn = h("input", { type: "range", min: Math.log(0.1), max: Math.log(3), step: 0.01, value: st.logSigma, "aria-label": "σ" });
  const muOut = h("output");
  const sgOut = h("output");
  muIn.addEventListener("input", () => ((st.mu = +muIn.value), render()));
  sgIn.addEventListener("input", () => ((st.logSigma = +sgIn.value), render()));
  root.replaceChildren(
    h("div", { class: "controls" },
      h("div", { class: "ctl wide" }, h("span", { class: "lbl sym" }, "μ"), muIn, muOut),
      h("div", { class: "ctl wide" }, h("span", { class: "lbl sym" }, "σ"), sgIn, sgOut),
      h("button", { class: "btn", type: "button", onclick: () => ((muIn.value = st.mu = 0), (sgIn.value = st.logSigma = 0), render()) }, T.matchPrior)),
    legendRow([["muted", T.prior], ["s1", T.encoderCloud]]),
    chartBox, terms);
  const xs = Array.from({ length: 241 }, (_, i) => -6 + i * 0.05);
  const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  function render() {
    const s = Math.exp(st.logSigma);
    const m = st.mu;
    muOut.textContent = fmt(m, 2);
    sgOut.textContent = fmt(s, 2);
    const q = xs.map((x) => pdf(x, m, s));
    const p = xs.map((x) => pdf(x, 0, 1));
    chartBox.replaceChildren(lineChart({
      width: widthOf(root), height: 190, x0: -6, x1: 6, y0: 0, y1: Math.max(0.45, Math.min(4.2, Math.max(...q) * 1.08)),
      series: [{ xs, ys: p, cls: "past", label: T.prior }, { xs, ys: q, cls: "s1", label: T.encoderCloud }],
      bands: [{ xs, lo: xs.map(() => 0), hi: q, color: "var(--s1)", opacity: 0.12 }],
      xFmt: (v) => fmt(v, 0), yFmt: (v) => fmt(v, 1), xLabel: "z", tipTitle: (x) => `z = ${fmt(x, 2)}`,
    }));
    const a = 0.5 * m * m;
    const b = 0.5 * (s * s - Math.log(s * s) - 1);
    const tex = `\\mathrm{KL} = \\tfrac12\\big(\\underbrace{${fmt(m * m, 2)}}_{\\mu^2} + \\underbrace{${fmt(s * s, 2)}}_{\\sigma^2} - \\underbrace{(${fmt(Math.log(s * s), 2)})}_{\\ln\\sigma^2} - 1\\big) = \\underbrace{${fmt(a, 3)}}_{\\text{${T.klMean}}} + \\underbrace{${fmt(b, 3)}}_{\\text{${T.klWidth}}} = \\mathbf{${fmt(a + b, 3)}}\\ \\text{${T.nats}}`;
    renderTex(terms, tex.replaceAll("−", "-"));
  }
  render();
  return { redraw: render };
}

// KaTeX may still be loading from the CDN: show the source, then typeset once it arrives.
export function renderTex(el, tex) {
  el.dataset.tex = tex;
  if (window.katex) {
    try {
      window.katex.render(tex, el, { displayMode: true, throwOnError: false, strict: "ignore" });
      return;
    } catch {
      /* fall through */
    }
  }
  el.textContent = tex;
  document.getElementById("katex-js")?.addEventListener("load", () => renderTex(el, el.dataset.tex), { once: true });
}

// 3.2
function cloudSection(D, root) {
  const key = "vae-1";
  const vae = D.models[key];
  const st = { i: firstOf(D, 2), samples: [], eps: null, seed: 3 };
  const orig = bigDigit(T.original);
  const mean = bigDigit(T.decodedMean);
  const samplesBox = h("div");
  const readout = h("dl", { class: "kv" });
  const map = latentMap(D, {
    modelKey: () => key, bounds: () => [-3.2, 3.2, -3.2, 3.2], alpha: 0.22, aspect: 0.8, maxH: 420,
    onMove(i, z, ev) {
      if (i >= 0) showTip(`${T.image} #${i} · ${T.digit} ${D.labels[i]} · ${T.clickToPick}`, ev);
      else hideTip();
    },
    onClick(i) {
      if (i < 0) return;
      st.i = i;
      st.samples = [];
      drawEps(12);
    },
    overlay(ctx, toPx) {
      const c = colors();
      const { mu, logvar } = codes(D, key);
      const m = [mu[2 * st.i], mu[2 * st.i + 1]];
      const s = [Math.exp(0.5 * logvar[2 * st.i]), Math.exp(0.5 * logvar[2 * st.i + 1])];
      const [cx, cy] = toPx(m[0], m[1]);
      const [ex] = toPx(m[0] + 2 * s[0], m[1]);
      const [, ey] = toPx(m[0], m[1] + 2 * s[1]);
      ctx.fillStyle = rgbCss(c.s1rgb, 0.16);
      ctx.strokeStyle = c.s1;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(cx, cy, Math.abs(ex - cx), Math.abs(cy - ey), 0, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = c.s2;
      ctx.strokeStyle = c.surface;
      for (const z of st.samples) {
        const [x, y] = toPx(z[0], z[1]);
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
      ctx.fillStyle = c.inkCss;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fill();
    },
  });
  function drawEps(k = 12) {
    const r = rng(st.seed++);
    const { mu, logvar } = codes(D, key);
    const m = [mu[2 * st.i], mu[2 * st.i + 1]];
    const s = [Math.exp(0.5 * logvar[2 * st.i]), Math.exp(0.5 * logvar[2 * st.i + 1])];
    st.samples = [];
    for (let j = 0; j < k; j++) {
      const eps = [r.normal(), r.normal()];
      st.eps = eps;
      st.samples.push([m[0] + s[0] * eps[0], m[1] + s[1] * eps[1]]);
    }
    const last = st.samples[st.samples.length - 1];
    readout.replaceChildren(
      h("dt", null, "μ"), h("dd", null, `(${fmt(m[0], 2)}, ${fmt(m[1], 2)})`),
      h("dt", null, "σ"), h("dd", null, `(${fmt(s[0], 2)}, ${fmt(s[1], 2)})`),
      h("dt", null, T.lastEps), h("dd", null, `(${fmt(st.eps[0], 2)}, ${fmt(st.eps[1], 2)})`),
      h("dt", null, "z = μ + σ·ε"), h("dd", null, `(${fmt(last[0], 2)}, ${fmt(last[1], 2)})`));
    drawDigit(orig.canvas, img(D, st.i));
    drawDigit(mean.canvas, decode(vae, m));
    samplesBox.replaceChildren(h("div", { class: "panel-title" }, T.decodedSamples(k)), strip(st.samples.map((z) => decode(vae, z)), "small"));
    map.draw();
  }
  root.replaceChildren(
    h("div", { class: "controls" },
      h("button", { class: "btn primary", type: "button", onclick: () => drawEps(12) }, T.drawEps),
      h("button", { class: "btn", type: "button", onclick: () => {
        st.i = Math.floor(rng(st.seed++).uniform() * D.n);
        drawEps(12);
      } }, T.randomImage)),
    h("div", { class: "viz-row" },
      h("div", { class: "main-plot" }, map.el),
      h("div", { class: "side" }, h("div", { class: "digit-pair" }, orig.fig, mean.fig), readout, samplesBox)));
  requestAnimationFrame(() => {
    map.relayout();
    drawEps(12);
  });
  return { redraw: () => (map.relayout(), drawEps(st.samples.length || 12)) };
}

// 3.3
function vaeMapSection(D, root) {
  const st = { key: "vae-1", showMap: false, samples: [], seed: 21 };
  const samplesBox = h("div");
  const box = () => (st.key === "ae" ? padBox(codes(D, "ae").box) : [-3.2, 3.2, -3.2, 3.2]);
  const map = latentMap(D, {
    modelKey: () => st.key, bounds: box, showMap: () => st.showMap, alpha: 0.4,
    onMove(i, z, ev) {
      const im = decode(D.models[st.key], z);
      const tip = h("div", null, i >= 0 ? `${T.digit} ${D.labels[i]} · ` : "", `z = (${fmt(z[0], 2)}, ${fmt(z[1], 2)})`, h("br"), digitCanvas(im));
      showTip(tip, ev);
    },
    underlay(ctx, toPx) {
      if (st.key === "ae") return;
      const c = colors();
      ctx.strokeStyle = c.muted;
      ctx.lineWidth = 1;
      for (const r of [1, 2]) {
        const [cx, cy] = toPx(0, 0);
        const [ex] = toPx(r, 0);
        ctx.beginPath();
        ctx.arc(cx, cy, Math.abs(ex - cx), 0, 2 * Math.PI);
        ctx.stroke();
      }
    },
    overlay(ctx, toPx) {
      const c = colors();
      ctx.fillStyle = c.s2;
      ctx.strokeStyle = c.surface;
      ctx.lineWidth = 2;
      for (const z of st.samples) {
        const [x, y] = toPx(z[0], z[1]);
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
    },
  });
  function sample() {
    const r = rng(st.seed++);
    const [x0, x1, y0, y1] = codes(D, "ae").box;
    st.samples = Array.from({ length: 12 }, () => (st.key === "ae" ? [x0 + (x1 - x0) * r.uniform(), y0 + (y1 - y0) * r.uniform()] : [r.normal(), r.normal()]));
    samplesBox.replaceChildren(h("div", { class: "panel-title", html: st.key === "ae" ? T.samplesAE : T.samplesVAE }),
      strip(st.samples.map((z) => decode(D.models[st.key], z))));
    map.draw();
  }
  const terrAE = territory(D, "ae", (r) => {
    const [x0, x1, y0, y1] = codes(D, "ae").box;
    return [x0 + (x1 - x0) * r.uniform(), y0 + (y1 - y0) * r.uniform()];
  });
  const terrVAE = territory(D, "vae-1", (r) => [r.normal(), r.normal()]);
  const mapToggle = h("input", { type: "checkbox" });
  mapToggle.addEventListener("change", () => ((st.showMap = mapToggle.checked), map.draw()));
  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.model, segToggle([{ value: "ae", label: "AE" }, { value: "vae-1", label: "VAE (β = 1)" }], st.key, (v) => {
        st.key = v;
        map.relayout();
        sample();
      })),
      h("label", { class: "toggle" }, mapToggle, T.showDecoderMap),
      h("button", { class: "btn primary", type: "button", onclick: sample }, T.sample12)),
    h("div", { class: "viz-row" },
      h("div", { class: "main-plot" }, map.el),
      h("div", { class: "side" }, samplesBox, h("div", null, h("div", { class: "panel-title" }, T.territoryTitle),
        territoryBars([{ label: T.aeUniformBox, cls: "s1", values: terrAE }, { label: T.vaePrior, cls: "s2", values: terrVAE }], 340)))));
  requestAnimationFrame(() => {
    map.relayout();
    sample();
  });
  return { redraw: () => (map.relayout(), sample()) };
}

// 3.4
function betaSection(D, root) {
  const keys = ["vae-0.1", "vae-1", "vae-2", "vae-4"];
  const st = { key: "vae-1" };
  const fixedZ = (() => {
    const r = rng(5);
    return Array.from({ length: 10 }, () => [r.normal(), r.normal()]);
  })();
  const statsBox = h("div");
  const samplesBox = h("div");
  const map = latentMap(D, {
    modelKey: () => st.key, bounds: () => [-3.6, 3.6, -3.6, 3.6], alpha: 0.45, aspect: 0.8, maxH: 400,
    underlay(ctx, toPx) {
      const c = colors();
      ctx.strokeStyle = c.muted;
      for (const r of [1, 2]) {
        const [cx, cy] = toPx(0, 0);
        const [ex] = toPx(r, 0);
        ctx.beginPath();
        ctx.arc(cx, cy, Math.abs(ex - cx), 0, 2 * Math.PI);
        ctx.stroke();
      }
    },
  });
  function render() {
    const m = D.models[st.key];
    statsBox.replaceChildren(h("div", { class: "stats" },
      h("span", null, h("b", null, fmt(m.metrics.rec, 1)), T.recNats),
      h("span", null, h("b", null, fmt(m.metrics.kl, 2)), T.klNats),
      h("span", null, h("b", null, fmt(m.metrics.mse, 4)), T.pixelMse)),
    m.metrics.kl < 0.3 ? h("div", { class: "callout warn" }, T.collapse) : h("p", { class: "sub" }, T.betaNote(m.beta)));
    samplesBox.replaceChildren(h("div", { class: "panel-title", html: T.priorSamples }), strip(fixedZ.map((z) => decode(m, z)), "small"));
    map.draw();
  }
  root.replaceChildren(
    h("div", { class: "controls" }, ctl("β", segToggle(keys.map((k) => ({ value: k, label: String(D.models[k].beta) })), st.key, (v) => ((st.key = v), render())))),
    h("div", { class: "viz-row" }, h("div", { class: "main-plot" }, map.el), h("div", { class: "side" }, statsBox, samplesBox)));
  requestAnimationFrame(() => {
    map.relayout();
    render();
  });
  return { redraw: () => (map.relayout(), render()) };
}

// 3.5
function interpSection(D, root) {
  const st = { a: 0, b: 7, ka: 0, kb: 0 };
  const steps = 12;
  const rows = h("div");
  const maps = {};
  const pathOf = (key) => {
    const { mu } = codes(D, key);
    const [ia, ib] = [firstOf(D, st.a, st.ka), firstOf(D, st.b, st.kb)];
    const za = [mu[2 * ia], mu[2 * ia + 1]];
    const zb = [mu[2 * ib], mu[2 * ib + 1]];
    return Array.from({ length: steps }, (_, i) => {
      const t = i / (steps - 1);
      return [za[0] * (1 - t) + zb[0] * t, za[1] * (1 - t) + zb[1] * t];
    });
  };
  for (const key of ["ae", "vae-1"]) {
    maps[key] = latentMap(D, {
      modelKey: () => key, bounds: () => (key === "ae" ? padBox(codes(D, "ae").box) : [-3.2, 3.2, -3.2, 3.2]), alpha: 0.25, aspect: 0.75, maxH: 280, glyph: 9,
      overlay(ctx, toPx) {
        const c = colors();
        const p = pathOf(key);
        ctx.strokeStyle = c.s2;
        ctx.lineWidth = 2;
        ctx.beginPath();
        p.forEach((z, i) => ctx[i ? "lineTo" : "moveTo"](...toPx(z[0], z[1])));
        ctx.stroke();
        ctx.fillStyle = c.s2;
        for (const z of p) {
          const [x, y] = toPx(z[0], z[1]);
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      },
    });
  }
  function render() {
    const row = (key, label) => h("div", { class: "strip-row" }, h("span", null, label),
      strip([img(D, firstOf(D, st.a, st.ka)), ...pathOf(key).map((z) => decode(D.models[key], z)), img(D, firstOf(D, st.b, st.kb))], "small"));
    rows.replaceChildren(row("ae", "AE"), row("vae-1", "VAE (β = 1)"), h("p", { class: "sub" }, T.interpNote));
    maps.ae.draw();
    maps["vae-1"].draw();
  }
  root.replaceChildren(
    h("div", { class: "controls" },
      ctl(T.from, digitPick(st.a, (d) => ((st.a = d), (st.ka = 0), render()), false),
        h("button", { class: "btn", type: "button", onclick: () => (st.ka++, render()) }, T.another)),
      ctl(T.to, digitPick(st.b, (d) => ((st.b = d), (st.kb = 0), render()), false),
        h("button", { class: "btn", type: "button", onclick: () => (st.kb++, render()) }, T.another))),
    rows,
    h("div", { class: "viz-row" },
      h("div", { class: "grow" }, h("div", { class: "panel-title" }, T.pathIn("AE")), maps.ae.el),
      h("div", { class: "grow" }, h("div", { class: "panel-title" }, T.pathIn("VAE")), maps["vae-1"].el)));
  requestAnimationFrame(() => {
    maps.ae.relayout();
    maps["vae-1"].relayout();
    render();
  });
  return { redraw: () => (maps.ae.relayout(), maps["vae-1"].relayout(), render()) };
}

