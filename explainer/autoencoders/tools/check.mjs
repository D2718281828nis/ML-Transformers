// Run the page's JavaScript math in Node and compare it with the PyTorch outputs that
// build_assets.py stored in assets/checks.json.
//
//     node explainer/autoencoders/tools/check.mjs

import { readFileSync } from "node:fs";
import { prepareDigits, encode, decode, img } from "../js/digits.js";
import { prepareForecast, nlinear, aeForecast, cvaeMean } from "../js/forecast.js";
import { Dense, Net, mseGrad } from "../js/nn.js";

const asset = (name) => JSON.parse(readFileSync(new URL(`../assets/${name}`, import.meta.url), "utf8"));
const checks = asset("checks.json");
let worst = 0;
let failed = 0;

function compare(label, got, want, tol = 1e-4) {
  const g = Array.from(got).flat(Infinity);
  const w = want.flat(Infinity);
  if (g.length !== w.length) throw new Error(`${label}: length ${g.length} vs ${w.length}`);
  const diff = Math.max(...g.map((v, i) => Math.abs(v - w[i])));
  worst = Math.max(worst, diff);
  const ok = diff <= tol;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label.padEnd(34)} max |Δ| = ${diff.toExponential(2)}`);
}

// digits: the page resizes the shipped 28 × 28 images itself, then runs the conv encoders and decoders
const D = prepareDigits(asset("digits.json"), asset("digit-models.json"));
for (const key of Object.keys(D.models)) {
  const m = D.models[key];
  const enc = [0, 1, 2, 3, 4].map((i) => encode(m, img(D, i)));
  compare(`digits ${key} encoder μ (5 images)`, enc.map((e) => Array.from(e.mu)), checks.digits[key].mu);
  if (m.logvar) compare(`digits ${key} encoder log σ²`, enc.map((e) => Array.from(e.logvar)), checks.digits[key].logvar);
  compare(`digits ${key} decoder (2 codes)`, [[0.5, -1], [-1.5, 1.2]].map((z) => Array.from(decode(m, z))), checks.digits[key].decode);
}

// ETTh1 forecasting
const F = prepareForecast(asset("etth1.json"));
const probes = checks.forecast.windows.map(([i, v]) => F.series[v].subarray(i, i + F.L));
compare("NLinear (3 test windows)", probes.map((x) => Array.from(nlinear(F, x))), checks.forecast.nlinear);
compare("forecasting AE k=8", probes.map((x) => Array.from(aeForecast(F, 8, x).y)), checks.forecast.ae8);
compare("conditional VAE prior mean", probes.map((x) => Array.from(cvaeMean(F, x))), checks.forecast.cvaeMean);

// backprop of the trainable net used by the live demos, against central finite differences
{
  const net = new Net([new Dense(3, 5, "tanh", 1), new Dense(5, 2, "tanh", 2), new Dense(2, 3, "linear", 3)]);
  const X = Float32Array.from({ length: 4 * 3 }, (_, i) => Math.sin(i * 1.7));
  const loss = () => mseGrad(net.forward(X, 4), X).loss;
  net.zeroGrad();
  net.backward(mseGrad(net.forward(X, 4), X).grad);
  const analytic = [];
  const numeric = [];
  for (const layer of net.layers) {
    for (const [p, g] of [[layer.W, layer.gW], [layer.b, layer.gb]]) {
      for (let i = 0; i < p.length; i++) {
        const keep = p[i];
        p[i] = keep + 1e-3;
        const up = loss();
        p[i] = keep - 1e-3;
        const down = loss();
        p[i] = keep;
        analytic.push(g[i]);
        numeric.push((up - down) / 2e-3);
      }
    }
  }
  compare("trainable net: backprop vs finite diff.", analytic, numeric, 2e-3);
}

console.log(`\n${failed ? `${failed} check(s) FAILED` : "all checks passed"} · largest difference ${worst.toExponential(2)}`);
process.exit(failed ? 1 : 0);
