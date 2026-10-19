// GPT-2 small: embeddings + the first attention sub-layer, with every
// intermediate kept. Mirrors GPT2Model.forward and GPT2Attention in
// src/transformers/models/gpt2/modeling_gpt2.py (eager attention path).

export const CFG = {
  nEmbd: 768,
  nHead: 12,
  headDim: 64,
  nVocab: 50257,
  nCtx: 1024,
  eps: 1e-5,
};

const D = CFG.nEmbd;

// y = x·W + b with W stored [in, out] row-major (GPT-2 "Conv1D" layout).
function linear(x, W, b, outDim) {
  const y = new Float32Array(outDim);
  y.set(b);
  for (let d = 0; d < x.length; d++) {
    const xd = x[d];
    if (xd === 0) continue;
    const off = d * outDim;
    for (let j = 0; j < outDim; j++) y[j] += xd * W[off + j];
  }
  return y;
}

function layerNorm(x, gamma, beta) {
  let mean = 0;
  for (let d = 0; d < D; d++) mean += x[d];
  mean /= D;
  let variance = 0;
  for (let d = 0; d < D; d++) variance += (x[d] - mean) ** 2;
  variance /= D; // biased, like torch.nn.LayerNorm
  const std = Math.sqrt(variance + CFG.eps);
  const norm = new Float32Array(D);
  const out = new Float32Array(D);
  for (let d = 0; d < D; d++) {
    norm[d] = (x[d] - mean) / std;
    out[d] = norm[d] * gamma[d] + beta[d];
  }
  return { mean, std, norm, out };
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function cosine(a, b) {
  return dot(a, b) / (Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b)) || 1);
}

function softmaxRow(row, upto) {
  // Positions > upto are masked (−∞) and get probability 0.
  const p = new Float64Array(row.length);
  let max = -Infinity;
  for (let j = 0; j <= upto; j++) max = Math.max(max, row[j]);
  let sum = 0;
  for (let j = 0; j <= upto; j++) {
    p[j] = Math.exp(row[j] - max);
    sum += p[j];
  }
  for (let j = 0; j <= upto; j++) p[j] /= sum;
  return p;
}

// Sinusoidal encoding from "Attention Is All You Need" (shown for comparison only;
// GPT-2 itself uses the learned table wpe).
export function sinusoidal(pos, d = D) {
  const v = new Float32Array(d);
  for (let i = 0; i < d; i += 2) {
    const angle = pos / Math.pow(10000, i / d);
    v[i] = Math.sin(angle);
    if (i + 1 < d) v[i + 1] = Math.cos(angle);
  }
  return v;
}

// tokEmb: array of Float32Array(768) rows of wte for each token.
// W: { lnW, lnB, attnW, attnB, projW, projB, wpe }.
export function forward(tokEmb, W) {
  const n = tokEmb.length;
  const { nHead, headDim } = CFG;
  const posEmb = [];
  const x = [];
  for (let i = 0; i < n; i++) {
    const p = W.wpe.subarray(i * D, (i + 1) * D);
    posEmb.push(p);
    const xi = new Float32Array(D);
    for (let d = 0; d < D; d++) xi[d] = tokEmb[i][d] + p[d];
    x.push(xi);
  }

  const ln = x.map((xi) => layerNorm(xi, W.lnW, W.lnB));

  // One matmul for all three projections, then split into Q | K | V.
  const q = [];
  const k = [];
  const v = [];
  for (let i = 0; i < n; i++) {
    const qkv = linear(ln[i].out, W.attnW, W.attnB, 3 * D);
    q.push(qkv.subarray(0, D));
    k.push(qkv.subarray(D, 2 * D));
    v.push(qkv.subarray(2 * D, 3 * D));
  }

  const scale = Math.sqrt(headDim);
  const heads = [];
  for (let h = 0; h < nHead; h++) {
    const lo = h * headDim;
    const hi = lo + headDim;
    const qh = q.map((r) => r.subarray(lo, hi));
    const kh = k.map((r) => r.subarray(lo, hi));
    const vh = v.map((r) => r.subarray(lo, hi));
    const scores = qh.map((qi) => Float64Array.from(kh, (kj) => dot(qi, kj)));
    const scaled = scores.map((row) => row.map((s) => s / scale));
    const probs = scaled.map((row, i) => softmaxRow(row, i));
    const probsNoMask = scaled.map((row) => softmaxRow(row, n - 1));
    const out = probs.map((row) => {
      const z = new Float32Array(headDim);
      for (let j = 0; j < n; j++) {
        if (!row[j]) continue;
        for (let d = 0; d < headDim; d++) z[d] += row[j] * vh[j][d];
      }
      return z;
    });
    heads.push({ q: qh, k: kh, v: vh, scores, scaled, probs, probsNoMask, out });
  }

  const concat = [];
  const attnOut = [];
  const resid = [];
  for (let i = 0; i < n; i++) {
    const c = new Float32Array(D);
    for (let h = 0; h < nHead; h++) c.set(heads[h].out[i], h * headDim);
    concat.push(c);
    const o = linear(c, W.projW, W.projB, D);
    attnOut.push(o);
    const r = new Float32Array(D);
    for (let d = 0; d < D; d++) r[d] = x[i][d] + o[d];
    resid.push(r);
  }

  return { n, tokEmb, posEmb, x, ln, q, k, v, scale, heads, concat, attnOut, resid };
}

// What each head mostly does on this sentence: average weight a token puts on
// itself, on the previous token, on the first token, and on everything else.
// Uses rows i >= 2, where those three positions are all different.
export function headProfile(probs) {
  const n = probs.length;
  const rows = [];
  for (let i = n >= 3 ? 2 : 1; i < n; i++) rows.push(i);
  if (!rows.length) return { self: 1, prev: 0, first: 0, other: 0, label: "self" };
  const acc = { self: 0, prev: 0, first: 0, other: 0 };
  for (const i of rows) {
    const self = probs[i][i];
    const prev = probs[i][i - 1];
    const first = i - 1 > 0 ? probs[i][0] : 0;
    acc.self += self;
    acc.prev += prev;
    acc.first += first;
    acc.other += Math.max(0, 1 - self - prev - first);
  }
  for (const key in acc) acc[key] /= rows.length;
  const label = Object.entries(acc).sort((a, b) => b[1] - a[1])[0][0];
  return { ...acc, label };
}
