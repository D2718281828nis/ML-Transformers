// Small network math in plain Float32Arrays.
//  * forward passes for the PyTorch weights exported by tools/build_assets.py (dense and convolutional layers,
//    weights in PyTorch's layout, row-major);
//  * a trainable Dense/Net with backprop and Adam for the live demos (toy autoencoder, forecasting lab);
//  * ridge least squares (Cholesky) for the lab's full-rank reference.

import { rng } from "./util.js";

// y = W x + b for one vector. W: Float32Array(out*in).
export function affine(W, b, x, nOut, nIn, out = new Float32Array(nOut)) {
  for (let o = 0; o < nOut; o++) {
    let acc = b ? b[o] : 0;
    const row = o * nIn;
    for (let i = 0; i < nIn; i++) acc += W[row + i] * x[i];
    out[o] = acc;
  }
  return out;
}

export const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// Run an exported nn.Sequential(Linear, act, Linear, ...) on one vector. layers: [{W, b, nIn, nOut}]
export function mlp(layers, x, act = "relu") {
  let a = x;
  layers.forEach((L, i) => {
    a = affine(L.W, L.b, a, L.nOut, L.nIn);
    if (i < layers.length - 1) {
      if (act === "relu") for (let j = 0; j < a.length; j++) a[j] = a[j] > 0 ? a[j] : 0;
      else if (act === "tanh") for (let j = 0; j < a.length; j++) a[j] = Math.tanh(a[j]);
    }
  });
  return a;
}

// nn.Conv2d on one [cIn, h, w] image. W: [cOut, cIn, k, k].
export function conv2d(x, cIn, h, w, W, b, cOut, k, stride, pad) {
  const ho = Math.floor((h + 2 * pad - k) / stride) + 1;
  const wo = Math.floor((w + 2 * pad - k) / stride) + 1;
  const out = new Float32Array(cOut * ho * wo);
  for (let o = 0; o < cOut; o++) {
    for (let oy = 0; oy < ho; oy++) {
      for (let ox = 0; ox < wo; ox++) {
        let acc = b[o];
        for (let c = 0; c < cIn; c++) {
          const wBase = (o * cIn + c) * k * k;
          const xBase = c * h * w;
          for (let ky = 0; ky < k; ky++) {
            const iy = oy * stride - pad + ky;
            if (iy < 0 || iy >= h) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = ox * stride - pad + kx;
              if (ix >= 0 && ix < w) acc += W[wBase + ky * k + kx] * x[xBase + iy * w + ix];
            }
          }
        }
        out[(o * ho + oy) * wo + ox] = acc;
      }
    }
  }
  return { out, h: ho, w: wo };
}

// nn.ConvTranspose2d on one [cIn, h, w] image. W: [cIn, cOut, k, k]. Each input pixel scatters a k × k stamp.
export function convTranspose2d(x, cIn, h, w, W, b, cOut, k, stride, pad) {
  const ho = (h - 1) * stride - 2 * pad + k;
  const wo = (w - 1) * stride - 2 * pad + k;
  const out = new Float32Array(cOut * ho * wo);
  for (let o = 0; o < cOut; o++) out.fill(b[o], o * ho * wo, (o + 1) * ho * wo);
  for (let c = 0; c < cIn; c++) {
    for (let iy = 0; iy < h; iy++) {
      for (let ix = 0; ix < w; ix++) {
        const v = x[(c * h + iy) * w + ix];
        if (v === 0) continue; // inputs come out of a ReLU: mostly zeros
        for (let o = 0; o < cOut; o++) {
          const wBase = (c * cOut + o) * k * k;
          const oBase = o * ho * wo;
          for (let ky = 0; ky < k; ky++) {
            const oy = iy * stride - pad + ky;
            if (oy < 0 || oy >= ho) continue;
            for (let kx = 0; kx < k; kx++) {
              const ox = ix * stride - pad + kx;
              if (ox >= 0 && ox < wo) out[oBase + oy * wo + ox] += v * W[wBase + ky * k + kx];
            }
          }
        }
      }
    }
  }
  return { out, h: ho, w: wo };
}

// Run an exported nn.Sequential of Linear / Conv2d / ConvTranspose2d / ReLU / Flatten / Unflatten on one input.
// layers: [{type, W, b, shape, stride, pad}] as prepared from build_assets.py; shape: [c, h, w] of an image input, or null for a vector.
export function sequential(layers, x, shape = null) {
  let a = x;
  let s = shape;
  for (const L of layers) {
    if (L.type === "linear") a = affine(L.W, L.b, a, L.W.shape[0], L.W.shape[1]);
    else if (L.type === "relu") for (let j = 0; j < a.length; j++) a[j] = a[j] > 0 ? a[j] : 0;
    else if (L.type === "flatten") s = null;
    else if (L.type === "unflatten") s = L.shape;
    else {
      const [cA, cB, k] = L.W.shape;
      const r = L.type === "conv"
        ? conv2d(a, s[0], s[1], s[2], L.W, L.b, cA, k, L.stride, L.pad)
        : convTranspose2d(a, s[0], s[1], s[2], L.W, L.b, cB, k, L.stride, L.pad);
      a = r.out;
      s = [L.type === "conv" ? cA : cB, r.h, r.w];
    }
  }
  return a;
}

// ───────────── trainable layers ─────────────
const ACT = {
  linear: { f: (z) => z, df: () => 1 },
  tanh: { f: Math.tanh, df: (a) => 1 - a * a }, // derivative written in terms of the output a
};

export class Dense {
  constructor(nIn, nOut, act = "linear", seed = 1) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.act = ACT[act];
    const r = rng(seed);
    const bound = 1 / Math.sqrt(nIn); // PyTorch's default nn.Linear init
    this.W = Float32Array.from({ length: nIn * nOut }, () => (2 * r.uniform() - 1) * bound);
    this.b = Float32Array.from({ length: nOut }, () => (2 * r.uniform() - 1) * bound);
    this.gW = new Float32Array(this.W.length);
    this.gb = new Float32Array(nOut);
    this.mW = new Float32Array(this.W.length);
    this.vW = new Float32Array(this.W.length);
    this.mb = new Float32Array(nOut);
    this.vb = new Float32Array(nOut);
  }

  // X: Float32Array(n * nIn) → A: Float32Array(n * nOut)
  forward(X, n) {
    const { nIn, nOut, W, b } = this;
    const A = new Float32Array(n * nOut);
    for (let r = 0; r < n; r++) {
      const xo = r * nIn;
      for (let o = 0; o < nOut; o++) {
        let acc = b[o];
        const wo = o * nIn;
        for (let i = 0; i < nIn; i++) acc += W[wo + i] * X[xo + i];
        A[r * nOut + o] = this.act.f(acc);
      }
    }
    this.X = X;
    this.A = A;
    this.n = n;
    return A;
  }

  // dA: gradient w.r.t. this layer's output. Accumulates gW, gb; returns gradient w.r.t. input.
  backward(dA) {
    const { nIn, nOut, W, X, A, n } = this;
    const dX = new Float32Array(n * nIn);
    for (let r = 0; r < n; r++) {
      for (let o = 0; o < nOut; o++) {
        const k = r * nOut + o;
        const dz = dA[k] * this.act.df(A[k]);
        if (dz === 0) continue;
        this.gb[o] += dz;
        const wo = o * nIn;
        const xo = r * nIn;
        for (let i = 0; i < nIn; i++) {
          this.gW[wo + i] += dz * X[xo + i];
          dX[xo + i] += dz * W[wo + i];
        }
      }
    }
    return dX;
  }

  zeroGrad() {
    this.gW.fill(0);
    this.gb.fill(0);
  }

  adam(lr, t, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    const c1 = 1 - b1 ** t;
    const c2 = 1 - b2 ** t;
    const upd = (p, g, m, v) => {
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    };
    upd(this.W, this.gW, this.mW, this.vW);
    upd(this.b, this.gb, this.mb, this.vb);
  }
}

export class Net {
  constructor(layers) {
    this.layers = layers;
    this.t = 0;
  }
  forward(X, n) {
    let a = X;
    for (const l of this.layers) a = l.forward(a, n);
    return a;
  }
  backward(dOut) {
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) d = this.layers[i].backward(d);
    return d;
  }
  zeroGrad() {
    for (const l of this.layers) l.zeroGrad();
  }
  step(lr) {
    this.t++;
    for (const l of this.layers) l.adam(lr, this.t);
  }
}

// mean squared error and its gradient for pred vs target (same length)
export function mseGrad(pred, target) {
  const g = new Float32Array(pred.length);
  let loss = 0;
  const scale = 2 / pred.length;
  for (let i = 0; i < pred.length; i++) {
    const d = pred[i] - target[i];
    loss += d * d;
    g[i] = scale * d;
  }
  return { loss: loss / pred.length, grad: g };
}

// ───────────── least squares ─────────────
// Solve (XᵀX + λI) B = XᵀY for B (p × q). X: n × p, Y: n × q, row-major. Returns Float64Array(p*q).
export function ridge(X, Y, n, p, q, lambda = 1e-3) {
  const A = new Float64Array(p * p);
  const R = new Float64Array(p * q);
  for (let r = 0; r < n; r++) {
    const xo = r * p;
    const yo = r * q;
    for (let i = 0; i < p; i++) {
      const xi = X[xo + i];
      if (xi === 0) continue;
      for (let j = i; j < p; j++) A[i * p + j] += xi * X[xo + j];
      for (let j = 0; j < q; j++) R[i * q + j] += xi * Y[yo + j];
    }
  }
  for (let i = 0; i < p; i++) {
    A[i * p + i] += lambda * n;
    for (let j = 0; j < i; j++) A[i * p + j] = A[j * p + i];
  }
  // Cholesky A = C Cᵀ (lower)
  const C = new Float64Array(p * p);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * p + j];
      for (let k = 0; k < j; k++) sum -= C[i * p + k] * C[j * p + k];
      C[i * p + j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / C[j * p + j];
    }
  }
  const B = new Float64Array(p * q);
  const tmp = new Float64Array(p);
  for (let col = 0; col < q; col++) {
    for (let i = 0; i < p; i++) {
      let sum = R[i * q + col];
      for (let k = 0; k < i; k++) sum -= C[i * p + k] * tmp[k];
      tmp[i] = sum / C[i * p + i];
    }
    for (let i = p - 1; i >= 0; i--) {
      let sum = tmp[i];
      for (let k = i + 1; k < p; k++) sum -= C[k * p + i] * B[k * q + col];
      B[i * q + col] = sum / C[i * p + i];
    }
  }
  return B;
}
