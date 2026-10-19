// Runs the explainer's own tokenizer + model code in Node and prints the results as JSON.
// Used by verify.py. Usage: node dump_js.mjs '<json {"tok": [...], "model": [...]}>'
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const { GPT2Tokenizer } = await import(`${root}js/tokenizer.js`);
const { RemoteSafetensors, loadStaticWeights, EmbeddingRows } = await import(`${root}js/weights.js`);
const { forward } = await import(`${root}js/model.js`);

const input = JSON.parse(process.argv[2]);
const tok = new GPT2Tokenizer(
  JSON.parse(fs.readFileSync(`${root}assets/gpt2/vocab.json`, "utf8")),
  fs.readFileSync(`${root}assets/gpt2/merges.txt`, "utf8"),
);
const out = { tok: input.tok.map((t) => tok.encode(t).ids), model: [] };
const st = new RemoteSafetensors();
const W = await loadStaticWeights(st);
const emb = new EmbeddingRows(st);
for (const text of input.model) {
  const { ids } = tok.encode(text);
  const r = forward(await emb.get(ids), W);
  const rows = (a) => a.map((v) => Array.from(v));
  out.model.push({ ids, x: rows(r.x), probs: r.heads.map((h) => rows(h.probs)), attnOut: rows(r.attnOut) });
}
process.stdout.write(JSON.stringify(out));
