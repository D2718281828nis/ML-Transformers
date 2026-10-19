// Interlude "How the tokenizer is trained" (T.1–T.6): byte-level BPE training on a
// small editable text, round by round, and GPT-2's real merges.txt cut at N rules.

import { bpeMerge, mergeAll, pairPositions, vocabJsonKey } from "./tokenizer.js";
import { T } from "./i18n.js";
import { h, int, fill, mono, stepper, player, lineChart, tokText } from "./viz.js";

const MAX_RULES = 40;
const GPT2_STOPS = [0, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 50000];
const MAX_CHUNK_ROWS = 24;

let tok = null; // GPT2Tokenizer
const st = {
  preset: 0,
  corpus: T.presets[0].text,
  test: T.presets[0].test,
  k: 0, // rounds done (T.2, T.3)
  word: null, // chunk picked in T.4
  stop: GPT2_STOPS.length - 1, // T.6 slider
  enc: null, // the sentence typed at the top of the page
};
const rounds = player();
const ui = {}; // persistent inputs
let cache = null;
let scaleCache = null;
let rebuilt = null;

const $ = (id) => document.getElementById(id);
const shown = (sym) => tok.displayToken(sym).text; // byte-level symbol → readable text
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function initTraining(tokenizer) {
  tok = tokenizer;
  renderAllTraining();
}

// Called whenever the sentence at the top changes (only T.6 uses it).
export function renderTraining(enc, all = false) {
  if (enc) st.enc = enc;
  if (!tok) return;
  if (all) renderAllTraining();
  else renderScale();
}

function renderAllTraining() {
  renderStart();
  renderRound();
  renderLoop();
  renderWhy();
  renderApply();
  renderScale();
}

function setCorpus(text, preset = st.preset) {
  st.corpus = text;
  st.preset = preset;
  if (preset >= 0 && text === T.presets[preset].text) {
    st.test = T.presets[preset].test;
    if (ui.test) ui.test.value = st.test;
  }
  if (ui.area && ui.area.value !== text) ui.area.value = text;
  st.k = 0;
  st.word = null;
  rounds.stop();
  renderStart();
  renderRound();
  renderLoop();
  renderWhy();
  renderApply();
}

// ───────────── the training itself ─────────────
// Count every neighbouring pair over all chunks, weighted by how often each chunk
// occurs. `parts` remembers which chunks a count comes from (chunk index → occurrences).
function countPairs(words, counts) {
  const pairs = new Map();
  words.forEach((w, wi) => {
    for (let i = 0; i < w.length - 1; i++) {
      const key = w[i] + " " + w[i + 1];
      let e = pairs.get(key);
      if (!e) pairs.set(key, (e = { key, a: w[i], b: w[i + 1], count: 0, parts: new Map() }));
      e.count += counts[wi];
      e.parts.set(wi, (e.parts.get(wi) || 0) + 1);
    }
  });
  return [...pairs.values()].sort((x, y) => y.count - x.count); // stable: ties keep first-seen order
}

function train(text) {
  if (cache?.text === text) return cache;
  const freq = new Map();
  for (const { text: piece } of tok.pretokenize(text)) {
    const b = tok.toByteString(piece);
    freq.set(b, (freq.get(b) || 0) + 1);
  }
  const chunks = [...freq.keys()];
  const counts = [...freq.values()];
  let words = chunks.map((c) => Array.from(c));
  const snaps = [words];
  const steps = [];
  let ranked = countPairs(words, counts);
  while (steps.length < MAX_RULES && ranked.length && ranked[0].count >= 2) {
    const best = ranked[0];
    steps.push({ ...best, top: ranked.slice(0, 10) });
    words = words.map((w) => mergeAll(w, best.a, best.b).merged);
    snaps.push(words);
    ranked = countPairs(words, counts);
  }
  const ranks = new Map(steps.map((s, i) => [s.a + " " + s.b, i]));
  const tokenRank = new Map();
  steps.forEach((s, i) => tokenRank.has(s.a + s.b) || tokenRank.set(s.a + s.b, i));
  cache = {
    text, chunks, counts, snaps, steps, ranks, tokenRank,
    chunkSet: new Set(chunks),
    total: counts.reduce((a, b) => a + b, 0),
    symbols: snaps.map((sn) => sn.reduce((s, w, i) => s + w.length * counts[i], 0)),
    stop: ranked.length ? (steps.length >= MAX_RULES ? "limit" : "done") : steps.length ? "all" : "nothing",
  };
  return cache;
}

const stopMessage = (R) =>
  R.stop === "limit" ? T.trainLimit(MAX_RULES) : R.stop === "done" ? T.trainDone : R.stop === "all" ? T.trainAllMerged : T.trainNothing;

// Byte-level symbol boxes (as written in merges.txt); the tooltip shows the readable text.
const symBoxes = (syms, { hot = new Set(), fresh = new Set() } = {}) =>
  h("div", { class: "bpe-symbols small" },
    syms.map((sym, i) => h("span", { class: `sym${hot.has(i) ? " pair" : ""}${fresh.has(i) ? " just-merged" : ""}`, title: shown(sym) }, sym)));

// Readable token chips with their IDs.
const pieceChips = (pieces, idOf) =>
  h("div", { class: "tok-row" },
    pieces.map((p, i) =>
      h("span", { class: "tok static", style: `--tc: var(--tok${i % 10})`, title: p },
        h("span", null, tokText(shown(p))), idOf ? h("span", { class: "tid" }, idOf(p)) : null)));

// Keep a persistent top part (inputs) and re-render only the part below it,
// so typing never loses focus.
function slot(box, build) {
  let dyn = box.querySelector(":scope > .dyn");
  if (!dyn) {
    dyn = h("div", { class: "dyn" });
    fill(box, build(), dyn);
  }
  return dyn;
}

// ───────────── T.1 start ─────────────
function renderStart() {
  const box = $("viz-train-start");
  if (!box) return;
  const dyn = slot(box, () => {
    ui.presets = T.presets.map((p, i) => h("button", { type: "button", onclick: () => setCorpus(p.text, i) }, p.label));
    ui.area = h("textarea", { class: "corpus", rows: 3, spellcheck: "false", maxlength: 3000, "aria-label": T.corpusLabel });
    ui.area.value = st.corpus;
    let tmr = 0;
    ui.area.addEventListener("input", () => {
      clearTimeout(tmr);
      tmr = setTimeout(() => setCorpus(ui.area.value, -1), 250);
    });
    const reset = h("button", { class: "btn", type: "button", onclick: () => setCorpus(T.presets[Math.max(0, st.preset)].text, Math.max(0, st.preset)) }, T.reset);
    return h("div", null,
      h("div", { class: "examples presets" }, h("span", { class: "panel-title", style: { margin: 0 } }, T.presetsLabel), ui.presets),
      h("div", { class: "corpus-row" }, ui.area, reset));
  });
  ui.presets.forEach((b, i) => b.setAttribute("aria-pressed", String(i === st.preset)));

  const R = train(st.corpus);
  const order = R.chunks.map((_, i) => i).sort((a, b) => R.counts[b] - R.counts[a]);
  const rows = order.slice(0, MAX_CHUNK_ROWS);
  fill(dyn,
    h("p", { class: "sub" }, T.startStats(R.chunks.length, R.total, R.symbols[0])),
    R.chunks.some((c) => tok.tokenBytes(c).some((b) => b >= 0x80)) ? h("div", { class: "callout", style: { margin: "6px 0 10px" } }, T.multibyteNote) : null,
    h("div", { class: "scroll-x" },
      h("table", { class: "sm-table chunk-table" },
        h("thead", null, h("tr", null, T.chunkCols.map((c) => h("th", null, c)))),
        h("tbody", null, rows.map((i) =>
          h("tr", null,
            h("td", null, h("span", { class: "chunk-text" }, tokText(shown(R.chunks[i])))),
            h("td", null, symBoxes(R.snaps[0][i])),
            h("td", null, T.times(R.counts[i]))))))),
    order.length > rows.length ? h("p", { class: "sub" }, T.moreChunks(order.length - rows.length)) : null,
  );
}

// ───────────── T.2 one round, T.3 repeat ─────────────
function roundControls(R) {
  const K = R.steps.length;
  const k = Math.min(st.k, K);
  const both = () => {
    renderRound();
    renderLoop();
  };
  const go = (v) => {
    rounds.stop();
    st.k = clamp(v, 0, K);
    both();
  };
  return [
    h("div", { class: "panel-title" }, T.rulesLearned(k, K)),
    stepper({ k, K, go, playing: rounds.playing, onPlay: () => rounds.toggle(() => st.k, (v) => (st.k = v), K, both) }),
  ];
}

function breakdown(R, e) {
  const parts = [...e.parts].map(([wi, occ]) => ({ wi, occ, c: occ * R.counts[wi] })).sort((a, b) => b.c - a.c);
  const head = parts.slice(0, 3).map((p) => `${p.occ > 1 ? `${p.occ}·` : ""}${R.counts[p.wi]} (${shown(R.chunks[p.wi]).replace(/ /g, "␣")})`);
  return head.join(" + ") + (parts.length > 3 ? " + …" : "");
}

function renderRound() {
  const box = $("viz-train-round");
  if (!box) return;
  const R = train(st.corpus);
  const K = R.steps.length;
  const k = Math.min(st.k, K);
  const step = R.steps[k];
  if (!step) {
    fill(box, roundControls(R), h("div", { class: "bpe-msg" }, stopMessage(R)));
    return;
  }
  const ab = step.a + step.b;
  const table = h("table", { class: "sm-table pair-table" },
    h("thead", null, h("tr", null, h("th", null, T.colPair), h("th", null, T.colCount), h("th", null, T.colFrom), h("th", null, T.colGpt2))),
    h("tbody", null, step.top.map((e, i) => {
      const r = tok.ranks.get(e.key);
      return h("tr", { class: i === 0 ? "win" : null },
        h("td", null, mono(`${e.a} + ${e.b}`)),
        h("td", null, e.count),
        h("td", { class: "from" }, breakdown(R, e)),
        h("td", null, r === undefined ? h("span", { class: "muted" }, T.notInGpt2) : `#${int(r + 1)}`));
    })));
  const affected = [...step.parts.keys()].slice(0, 6).map((wi) => {
    const before = R.snaps[k][wi];
    const hot = new Set();
    for (const p of pairPositions(before, [step.a, step.b])) hot.add(p).add(p + 1);
    const after = R.snaps[k + 1][wi];
    const fresh = new Set(mergeAll(before, step.a, step.b).positions);
    return h("div", { class: "merge-row" }, symBoxes(before, { hot }), h("span", { class: "arrow" }, "→"), symBoxes(after, { fresh }), h("span", { class: "cnt" }, T.times(R.counts[wi])));
  });
  fill(box,
    roundControls(R),
    h("div", { class: "bpe-msg" }, T.nextRule(k + 1, mono(step.a), mono(step.b), mono(ab), step.count)),
    h("div", { class: "round-grid" },
      h("div", null,
        h("div", { class: "panel-title" }, T.countTitle),
        h("div", { class: "scroll-x" }, table),
        h("p", { class: "sub" }, T.tiesNote)),
      h("div", null,
        h("div", { class: "panel-title" }, T.pickTitle),
        h("div", { class: "pick" }, h("div", { class: "pick-pair" }, mono(step.a), " + ", mono(step.b), " → ", mono(ab)), h("div", { class: "sub" }, T.pickCount(step.count))),
        h("div", { class: "panel-title", style: { marginTop: "14px" } }, T.mergeTitle),
        h("div", { class: "merge-rows" }, affected),
        h("div", { class: "panel-title", style: { marginTop: "14px" } }, T.writeTitle),
        h("div", { class: "write" },
          h("div", { class: "sub" }, T.writeMerges(k + 1)), h("pre", { class: "merges-file mini" }, `${step.a} ${step.b}`),
          h("div", { class: "sub" }, T.writeVocab(256 + k)), h("pre", { class: "merges-file mini" }, `${vocabJsonKey(ab)}: ${256 + k}`)))),
  );
}

function renderLoop() {
  const box = $("viz-train-loop");
  if (!box) return;
  const R = train(st.corpus);
  const K = R.steps.length;
  const k = Math.min(st.k, K);
  const merges = h("pre", { class: "merges-file" },
    h("span", { class: "muted" }, "#version: 0.2\n"),
    R.steps.slice(0, k).map((s, i) => h("span", { class: i === k - 1 ? "fresh" : null }, h("span", { class: "ln" }, i + 1), `${s.a} ${s.b}\n`)),
    k === 0 ? h("span", { class: "muted" }, "…") : null);
  const vocab = h("pre", { class: "merges-file" },
    "{\n", h("span", { class: "muted" }, `  ${T.bytesCollapsed}\n`),
    R.steps.slice(0, k).map((s, i) => h("span", { class: i === k - 1 ? "fresh" : null }, `  ${vocabJsonKey(s.a + s.b)}: ${256 + i},\n`)),
    "}");
  fill(box,
    roundControls(R),
    lineChart([{ label: T.chartSymbols, values: R.symbols }], {
      height: 150, positive: true, digits: 0, points: R.symbols.length <= 41,
      xLabel: (i) => String(i), tipLabel: T.afterRounds, mark: k + 1, markLabel: T.roundsDone,
    }),
    h("div", { class: "stats" },
      h("span", null, h("b", null, T.shrink(R.symbols[0], R.symbols[k], k))),
      k > 0 ? h("span", null, T.lastShrink(R.symbols[k - 1] - R.symbols[k])) : null),
    h("div", { class: "files-row" },
      h("div", null, h("div", { class: "panel-title" }, "merges.txt"), merges),
      h("div", null, h("div", { class: "panel-title" }, "vocab.json"), vocab, h("p", { class: "sub" }, T.vocabGrowth(k, 256 + k)))),
    h("p", { class: "sub" }, T.lineIdNote),
  );
}

// ───────────── T.4 why a list of rules ─────────────
// All ways to cut `word` into pieces from `vocab`: the number of ways (dynamic
// programming) and up to `limit` of them, fewest pieces first.
function segmentations(word, vocab, limit = 8) {
  const ch = Array.from(word);
  const n = ch.length;
  const ways = new Array(n + 1).fill(0);
  ways[n] = 1;
  for (let p = n - 1; p >= 0; p--) for (let q = p + 1; q <= n; q++) if (vocab.has(ch.slice(p, q).join(""))) ways[p] += ways[q];
  const list = [];
  const cur = [];
  const dfs = (p) => {
    if (list.length >= limit) return;
    if (p === n) return void list.push([...cur]);
    for (let q = n; q > p; q--) {
      const piece = ch.slice(p, q).join("");
      if (!vocab.has(piece)) continue;
      cur.push(piece);
      dfs(q);
      cur.pop();
    }
  };
  dfs(0);
  return { count: ways[0], list };
}

function rebuildCheck() {
  if (rebuilt) return rebuilt;
  let id = 0;
  let ok = 0;
  for (const ch of tok.byteEncoder.values()) ok += tok.encoder[ch] === id++ ? 1 : 0;
  for (const m of tok.merges) ok += tok.encoder[m.replace(" ", "")] === id++ ? 1 : 0;
  ok += tok.encoder["<|endoftext|>"] === id++ ? 1 : 0;
  return (rebuilt = { ok, total: id });
}

// What a vocabulary alone suggests: always take the longest token that fits, from the left.
function longestFirst(word, vocab) {
  const ch = Array.from(word);
  const out = [];
  for (let p = 0; p < ch.length; ) {
    let q = ch.length;
    while (q > p + 1 && !vocab.has(ch.slice(p, q).join(""))) q--;
    out.push(ch.slice(p, q).join(""));
    p = q;
  }
  return out;
}

function renderWhy() {
  const box = $("viz-train-why");
  if (!box) return;
  const R = train(st.corpus);
  const K = R.steps.length;
  const vocab = new Set([...tok.byteEncoder.values(), ...R.steps.map((s) => s.a + s.b)]);
  const rankOf = (pair) => R.ranks.get(pair);
  const key = (p) => p.join("\u0000");
  // Candidates: chunks of the training text and new chunks from the T.5 text.
  const multi = (w) => Array.from(w).length > 1;
  const trained = R.chunks.map((_, i) => i).filter((i) => multi(R.chunks[i])).sort((a, b) => R.counts[b] - R.counts[a]).slice(0, 10).map((i) => R.chunks[i]);
  const fresh = [...new Set(tok.pretokenize(st.test).map((c) => tok.toByteString(c.text)))].filter((w) => multi(w) && !R.chunkSet.has(w)).slice(0, 6);
  const cands = [...trained, ...fresh];
  const differs = (w) => key(bpeMerge(Array.from(w), rankOf).pieces) !== key(longestFirst(w, vocab));
  if (st.word == null || !cands.includes(st.word)) {
    // default: a word where replaying the rules and longest-match disagree, else the one with most ways to cut it
    st.word = cands.find(differs) ?? cands.reduce((best, w) => (segmentations(w, vocab, 0).count > segmentations(best, vocab, 0).count ? w : best), cands[0] ?? null);
  }
  const check = rebuildCheck();
  const rebuild = h("div", { class: "callout", style: { marginTop: "16px" } },
    h("b", null, T.rebuildTitle), h("br"), h("span", { class: "sub" }, T.rebuildRecipe), h("br"),
    h("span", { class: check.ok === check.total ? "ok-mark" : "bad-mark" }, check.ok === check.total ? "✓ " : "✗ "), T.rebuildResult(check.ok, check.total));
  if (!K || st.word == null) {
    fill(box, h("div", { class: "bpe-msg" }, T.noRulesYet), rebuild);
    return;
  }
  const word = st.word;
  const replay = bpeMerge(Array.from(word), rankOf).pieces;
  const longest = longestFirst(word, vocab);
  const seg = segmentations(word, vocab, 8);
  for (const extra of [replay, longest]) if (!seg.list.some((p) => key(p) === key(extra))) seg.list.push(extra);
  seg.list.sort((a, b) => a.length - b.length);
  const chip = (w) =>
    h("button", { class: "chunk", type: "button", "aria-pressed": String(w === word), onclick: () => ((st.word = w), renderWhy()) },
      h("span", { class: "chunk-text" }, tokText(shown(w))),
      fresh.includes(w) ? h("span", { class: "chunk-kind new" }, T.newWord) : null);
  fill(box,
    h("p", { class: "sub", style: { marginTop: 0 } }, T.usingRules(K)),
    h("div", { class: "panel-title" }, T.pickWord),
    h("div", { class: "chunks", style: { marginBottom: "14px" } }, cands.map(chip)),
    h("div", { class: "panel-title" }, T.waysTitle(seg.count), " ", seg.count > seg.list.length ? h("span", { class: "muted" }, T.showingFirst(seg.list.length)) : null),
    h("div", { class: "ways" },
      seg.list.map((p) => {
        const isReplay = key(p) === key(replay);
        const isLongest = key(p) === key(longest);
        return h("div", { class: `way${isReplay ? " replay" : ""}${isLongest && !isReplay ? " longest" : ""}` },
          h("span", { class: "star" }, (isReplay ? "★" : "") + (isLongest ? "◆" : "")), pieceChips(p));
      })),
    h("p", { class: "sub" }, T.replayMark, h("br"), T.longestMark),
    key(replay) !== key(longest) ? h("div", { class: "callout warn" }, T.differs) : null,
    rebuild,
  );
}

// ───────────── T.5 rules on new text ─────────────
function renderApply() {
  const box = $("viz-train-apply");
  if (!box) return;
  const dyn = slot(box, () => {
    ui.test = h("input", { type: "text", class: "test-input", spellcheck: "false", maxlength: 200, "aria-label": T.testLabel });
    ui.test.value = st.test;
    let tmr = 0;
    ui.test.addEventListener("input", () => {
      clearTimeout(tmr);
      tmr = setTimeout(() => {
        st.test = ui.test.value;
        renderApply();
        renderWhy();
      }, 200);
    });
    return h("label", { class: "test-row" }, h("span", { class: "panel-title", style: { margin: 0 } }, T.testLabel), ui.test);
  });
  const R = train(st.corpus);
  const rankOf = (pair) => R.ranks.get(pair);
  const idOf = (p) => (Array.from(p).length === 1 ? tok.encoder[p] : 256 + R.tokenRank.get(p));
  const rows = tok.pretokenize(st.test).map(({ text }) => {
    const byteStr = tok.toByteString(text);
    const { pieces, steps } = bpeMerge(Array.from(byteStr), rankOf);
    const seen = R.chunkSet.has(byteStr);
    return h("tr", null,
      h("td", null, h("span", { class: "chunk-text" }, tokText(text)), h("div", { class: seen ? "muted tag" : "tag new" }, seen ? T.seen : T.unseen)),
      h("td", { class: "rules" }, steps.length ? steps.map((s) => h("span", { class: "rule" }, `#${s.rank + 1} ${s.pair[0]} + ${s.pair[1]}`)) : h("span", { class: "muted" }, T.noneApplied)),
      h("td", null, pieceChips(pieces, idOf)));
  });
  fill(dyn,
    h("div", { class: "scroll-x", style: { marginTop: "10px" } },
      h("table", { class: "sm-table apply-table" },
        h("thead", null, h("tr", null, h("th", null, T.chunkCols[0]), h("th", null, T.rulesApplied), h("th", null, ""))),
        h("tbody", null, rows))),
  );
}

// ───────────── T.6 GPT-2's own rules, cut at N ─────────────
function renderScale() {
  const box = $("viz-train-scale");
  if (!box) return;
  const dyn = slot(box, () => {
    ui.scale = h("input", { type: "range", min: 0, max: GPT2_STOPS.length - 1, step: 1, value: st.stop, "aria-label": T.scaleAria, class: "scale-slider" });
    ui.scale.addEventListener("input", () => ((st.stop = Number(ui.scale.value)), renderScale()));
    ui.scaleLabel = h("div", { class: "panel-title", style: { margin: 0 } });
    return h("div", { class: "scale-row" }, ui.scaleLabel, ui.scale);
  });
  const N = GPT2_STOPS[st.stop];
  ui.scaleLabel.textContent = T.scaleLabel(N);
  const chunks = st.enc?.chunks || [];
  if (!chunks.length) return fill(dyn, h("div", { class: "placeholder" }, T.scaleEmpty));
  if (scaleCache?.text !== st.enc.text) {
    scaleCache = { text: st.enc.text, counts: GPT2_STOPS.map((n) => chunks.reduce((s, c) => s + tok.bpeFirst(c.byteStr, n).length, 0)) };
  }
  const pieces = chunks.flatMap((c) => tok.bpeFirst(c.byteStr, N));
  const V = 256 + N + 1;
  fill(dyn,
    h("div", { style: { margin: "10px 0" } }, pieceChips(pieces, (p) => tok.encoder[p])),
    h("div", { class: "stats" }, h("span", null, h("b", null, T.scaleStats(pieces.length, V, V * 768)))),
    lineChart([{ label: T.scaleChart, values: scaleCache.counts }], {
      height: 150, positive: true, digits: 0, points: true, highlight: st.stop, ticks: true,
      xLabel: (i) => int(GPT2_STOPS[i]), tipLabel: (i) => T.scaleTip(GPT2_STOPS[i]),
    }),
    h("p", { class: "sub" }, T.scaleNote),
  );
}
