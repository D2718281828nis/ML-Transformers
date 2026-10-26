// Small dense-network math in plain Float32Arrays.
//  * forward passes for the PyTorch weights exported by tools/build_assets.py (weights are [out, in], row-major);
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
