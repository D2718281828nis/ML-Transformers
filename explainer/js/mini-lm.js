// A tiny next-word model, trained in the browser to show how an embedding matrix learns.
//
// Same structure as GPT-2, at toy size:
//   e = E[current word]          (look up a row of the embedding matrix E, V × 2)
//   h = A · e                    (A, 2 × 2, stands in for GPT-2's 12 blocks)
//   scores_w = E[w] · h + b_w    (E used a second time: weight tying, like GPT-2's lm_head)
//   p = softmax(scores),  loss = −ln p[next word]
// Initialized like GPT-2 (normal, std 0.02) and trained with Adam on all examples at once.

export const DIM = 2;
export const MAX_VOCAB = 30;
export const INIT_STD = 0.02;

// Whole words (and punctuation) are the tokens here.
export function wordsOf(text) {
  return text.toLowerCase().match(/[\p{L}\p{N}']+|[^\s\p{L}\p{N}]/gu) || [];
}

// Vocabulary, (current, next) examples, and the best loss any model that sees only the
// current word could reach (the conditional entropy of the next word).
export function prepare(text) {
  const all = wordsOf(text);
  const vocab = [];
  const index = new Map();
  for (const w of all) {
    if (!index.has(w) && vocab.length < MAX_VOCAB) {
      index.set(w, vocab.length);
      vocab.push(w);
    }
  }
  const ids = all.map((w) => index.get(w));
  const pairs = [];
  for (let k = 0; k + 1 < ids.length; k++) if (ids[k] !== undefined && ids[k + 1] !== undefined) pairs.push([ids[k], ids[k + 1]]);
  const counts = new Map();
  for (const [i, j] of pairs) {
    const row = counts.get(i) || new Map();
    row.set(j, (row.get(j) || 0) + 1);
    counts.set(i, row);
  }
  let floor = 0;
  for (const [i, j] of pairs) {
    const row = counts.get(i);
    const total = [...row.values()].reduce((a, b) => a + b, 0);
    floor -= Math.log(row.get(j) / total);
  }
  return { vocab, pairs, floor: pairs.length ? floor / pairs.length : 0, truncated: new Set(all).size > vocab.length };
}

// Small seeded PRNG (mulberry32) + Box–Muller, so a "random start" can be replayed.
export function gaussian(seed) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
}

export function initModel(V, seed) {
  const g = gaussian(seed);
  const params = {
    E: Float64Array.from({ length: V * DIM }, () => INIT_STD * g()),
    A: Float64Array.from({ length: DIM * DIM }, () => INIT_STD * g()),
    b: new Float64Array(V),
  };
  const zeros = () => Object.fromEntries(Object.entries(params).map(([k, v]) => [k, new Float64Array(v.length)]));
  return { V, params, m: zeros(), v: zeros(), t: 0, losses: [], zeros };
}

export function forward(P, V, i) {
  const e = [P.E[i * 2], P.E[i * 2 + 1]];
  const h = [P.A[0] * e[0] + P.A[1] * e[1], P.A[2] * e[0] + P.A[3] * e[1]];
  const z = new Float64Array(V);
  let max = -Infinity;
  for (let w = 0; w < V; w++) {
    z[w] = P.E[w * 2] * h[0] + P.E[w * 2 + 1] * h[1] + P.b[w];
    max = Math.max(max, z[w]);
  }
  const p = new Float64Array(V);
  let sum = 0;
  for (let w = 0; w < V; w++) sum += p[w] = Math.exp(z[w] - max);
  for (let w = 0; w < V; w++) p[w] /= sum;
  return { e, h, z, p };
}

// Gradients of −ln p[j] for one example (current word i, next word j), split into the two
// uses of E: as output scores (every row) and as the looked-up input (row i).
export function exampleGrads(P, V, i, j) {
  const { e, h, p } = forward(P, V, i);
  const gOut = new Float64Array(V * 2);
  const gb = new Float64Array(V);
  const dh = [0, 0];
  for (let w = 0; w < V; w++) {
    const dz = p[w] - (w === j ? 1 : 0);
    gb[w] = dz;
    gOut[w * 2] = dz * h[0];
    gOut[w * 2 + 1] = dz * h[1];
    dh[0] += dz * P.E[w * 2];
    dh[1] += dz * P.E[w * 2 + 1];
  }
  const gA = [dh[0] * e[0], dh[0] * e[1], dh[1] * e[0], dh[1] * e[1]];
  const gIn = [P.A[0] * dh[0] + P.A[2] * dh[1], P.A[1] * dh[0] + P.A[3] * dh[1]];
  return { loss: -Math.log(p[j]), p, h, gOut, gIn, gA, gb };
}

// Average loss and gradients over all examples.
export function lossAndGrads(model, pairs) {
  const { V, params: P } = model;
  const G = model.zeros();
  let loss = 0;
  for (const [i, j] of pairs) {
    const g = exampleGrads(P, V, i, j);
    loss += g.loss;
    for (let q = 0; q < V * 2; q++) G.E[q] += g.gOut[q];
    G.E[i * 2] += g.gIn[0];
    G.E[i * 2 + 1] += g.gIn[1];
    for (let q = 0; q < 4; q++) G.A[q] += g.gA[q];
    for (let w = 0; w < V; w++) G.b[w] += g.gb[w];
  }
  const n = pairs.length || 1;
  for (const k in G) for (let q = 0; q < G[k].length; q++) G[k][q] /= n;
  return { loss: loss / n, G };
}

// One Adam step on the full set of examples. Returns the loss before the step.
export function trainStep(model, pairs, lr = 0.05) {
  const { loss, G } = lossAndGrads(model, pairs);
  const t = ++model.t;
  for (const k in model.params) {
    const P = model.params[k];
    const m = model.m[k];
    const v = model.v[k];
    for (let q = 0; q < P.length; q++) {
      m[q] = 0.9 * m[q] + 0.1 * G[k][q];
      v[q] = 0.999 * v[q] + 0.001 * G[k][q] ** 2;
      P[q] -= (lr * (m[q] / (1 - 0.9 ** t))) / (Math.sqrt(v[q] / (1 - 0.999 ** t)) + 1e-8);
    }
  }
  model.losses.push(loss);
  return loss;
}

export function currentLoss(model, pairs) {
  return lossAndGrads(model, pairs).loss;
}
