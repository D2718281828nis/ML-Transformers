import { GPT2Tokenizer } from "./tokenizer.js";
import { RemoteSafetensors, loadStaticWeights, EmbeddingRows, POSITION_ROWS } from "./weights.js";
import { CFG, forward, headProfile, cosine, sinusoidal, dot } from "./model.js";
import {
  h, s, fmt, pct, int, tokChip, tokText, tokTip, tokPlain, heatRows, heatImage, matrixGrid, cellSize, bars, lineChart,
  attentionLines, legend, seqLegend, robustScale, divergingRGB, sequentialRGB, showTip, hideTip, resetPalette,
} from "./viz.js";

const MAX_TOKENS = 48;
const CODE_BASE =
  "https://github.com/D2718281828nis/ML-Transformers/blob/0ade1caa356dce6b70ef8293addeb0898f177206/src/transformers/models/gpt2/";
const EXAMPLES = [
  "Attention is all you need",
  "The chef cooked the meal and then she served it.",
  "I love tokenization!",
  "I saw the cat and the dog",
  "dog bites man",
  "Data visualization empowers users to",
  "Кошка сидит на ковре",
];
const HEAD_TAG = { self: "self", prev: "prev", first: "1st", other: "mix" };
const HEAD_DESC = {
  self: "mostly attends to the token itself",
  prev: "mostly attends to the previous token",
  first: "parks most attention on the first token",
  other: "spreads attention over other tokens",
};
const D = CFG.nEmbd;
const HD = CFG.headDim;
const utf8 = new TextEncoder();

const state = {
  enc: null,
  tokens: [],
  truncated: false,
  res: null,
  focus: 0,
  head: 7,
  mask: true,
  bpeChunk: 0,
  bpeStep: 0,
  posView: "learned",
  scoreCell: null,
  computeMs: 0,
};
let tokenizer = null;
let st = null;
let W = null;
let emb = null;
let weightsError = null;
let runId = 0;
let playTimer = null;

const $ = (id) => document.getElementById(id);
const input = $("text");

// ───────────── boot ─────────────
main();

async function main() {
  for (const a of document.querySelectorAll("a.code[data-code]")) a.href = CODE_BASE + a.dataset.code;
  setupTex();
  setupTopbar();
  setupHoverAndFocus();
  setupScrollSpy();
  renderExamples();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    resetPalette();
    renderAll();
  });

  setStatus("loading", "Loading tokenizer…");
  try {
    tokenizer = await GPT2Tokenizer.load("assets/gpt2");
  } catch (e) {
    setStatus("error", `Couldn't load the tokenizer files (${e.message}).`);
    return;
  }
  let timer = 0;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => update(input.value), 160);
  });
  update(input.value);
  loadWeights();
}

async function loadWeights() {
  weightsError = null;
  W = null;
  renderModelSections();
  setStatus("loading", "Downloading GPT-2 weights…", 0);
  st = new RemoteSafetensors(undefined, {
    onProgress: (loaded, expected) => {
      if (!W) setStatus("loading", `GPT-2 weights ${(loaded / 1e6).toFixed(1)} / ${(expected / 1e6).toFixed(1)} MB`, expected ? loaded / expected : 0);
    },
  });
  emb = new EmbeddingRows(st);
  try {
    W = await loadStaticWeights(st);
  } catch (e) {
    weightsError = e;
    setStatus("error", `Couldn't download GPT-2 weights from huggingface.co (${e.message}). Tokenization still works.`, null, true);
    renderModelSections();
    return;
  }
  compute();
}

function update(text) {
  const enc = tokenizer.encode(text);
  state.enc = enc;
  state.truncated = enc.tokens.length > MAX_TOKENS;
  state.tokens = enc.tokens.slice(0, MAX_TOKENS);
  state.focus = Math.max(0, state.tokens.length - 1);
  state.scoreCell = null;
  let best = 0;
  enc.chunks.forEach((c, i) => {
    if (c.steps.length > enc.chunks[best].steps.length) best = i;
  });
  state.bpeChunk = best;
  state.bpeStep = 0;
  stopPlay();
  const n = enc.tokens.length;
  $("input-meta").textContent = n ? `${n} token${n === 1 ? "" : "s"}` : "";
  for (const b of document.querySelectorAll("#examples button")) b.setAttribute("aria-pressed", String(b.dataset.ex === text));
  setFocusStyle();
  renderTokenization();
  state.res = null;
  if (W) compute();
  else renderModelSections();
}

async function compute() {
  const id = ++runId;
  if (!state.tokens.length) {
    state.res = null;
    renderModelSections();
    setStatus("ready", "GPT-2 ready");
    return;
  }
  const ids = state.tokens.map((t) => t.id);
  if (ids.some((i) => !emb.rows.has(i))) setStatus("loading", "Fetching token embeddings…", null);
  let rows;
  try {
    rows = await emb.get(ids);
  } catch (e) {
    if (id !== runId) return;
    weightsError = e;
    setStatus("error", `Couldn't fetch token embeddings (${e.message}).`, null, true);
    renderModelSections();
    return;
  }
  if (id !== runId) return;
  weightsError = null;
  const t0 = performance.now();
  state.res = forward(rows, W);
  state.computeMs = performance.now() - t0;
  setStatus("ready", `GPT-2 ready · ${(st.loaded / 1e6).toFixed(1)} MB loaded · ${state.computeMs.toFixed(0)} ms`);
  renderModelSections();
}

function setText(t) {
  input.value = t;
  update(t);
}

function setFocus(i) {
  if (i === state.focus || i >= state.tokens.length) return;
  state.focus = i;
  state.scoreCell = null;
  setFocusStyle();
  renderModelSections();
}

function setHead(hd) {
  if (hd === state.head) return;
  state.head = hd;
  state.scoreCell = null;
  renderAttention();
}

function renderAll() {
  if (!state.enc) return;
  renderTokenization();
  renderModelSections();
}

function renderModelSections() {
  renderEmbedding();
  renderPosition();
  renderAttention();
}

// ───────────── chrome: status, examples, hover, scroll spy, KaTeX ─────────────
function setStatus(kind, text, progress = null, retry = false) {
  const el = $("status");
  el.dataset.state = kind;
  el.querySelector(".status-text").textContent = text;
  el.title = text;
  el.querySelector(".status-bar > span").style.width = progress == null ? "100%" : `${Math.round(progress * 100)}%`;
  el.querySelector("button")?.remove();
  if (retry) el.append(h("button", { type: "button", onclick: () => (W ? compute() : loadWeights()) }, "Retry"));
}

function renderExamples() {
  $("examples").append(...EXAMPLES.map((ex) => h("button", { type: "button", "data-ex": ex, onclick: () => setText(ex) }, ex)));
  for (const el of document.querySelectorAll(".example-link")) {
    el.setAttribute("role", "button");
    el.tabIndex = 0;
    const go = () => setText(el.dataset.example);
    el.addEventListener("click", go);
    el.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), go()));
  }
}

function setupHoverAndFocus() {
  let hover = null;
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest?.("[data-tok]");
    const v = t ? t.dataset.tok : null;
    if (v === hover) return;
    hover = v;
    $("hover-style").textContent =
      v == null ? "" : `.tok[data-tok="${v}"]{box-shadow:0 0 0 2px var(--ink)} g[data-tok="${v}"] rect{stroke:var(--ink);stroke-width:1.5px}`;
  });
  document.addEventListener("click", (e) => {
    const t = e.target.closest?.("[data-tok]");
    if (t && state.res) setFocus(Number(t.dataset.tok));
  });
}

function setFocusStyle() {
  $("focus-style").textContent = `.tok[data-tok="${state.focus}"]{outline:2px solid var(--accent);outline-offset:1px}`;
}

function setupTopbar() {
  const bar = $("topbar");
  const set = () => document.documentElement.style.setProperty("--topbar-h", `${bar.offsetHeight}px`);
  new ResizeObserver(set).observe(bar);
  set();
}

function setupScrollSpy() {
  const nav = $("pipeline");
  const links = [...nav.querySelectorAll("a")];
  const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const obs = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const a = byId.get(e.target.id);
        links.forEach((l) => l.classList.toggle("active", l === a));
        if (a) nav.scrollTo({ left: a.offsetLeft - nav.clientWidth / 2 + a.offsetWidth / 2, behavior: "smooth" });
      }
    },
    { rootMargin: "-40% 0px -55% 0px" },
  );
  for (const id of byId.keys()) if ($(id)) obs.observe($(id));
}

function setupTex() {
  const render = () => {
    if (!window.katex) return;
    for (const el of document.querySelectorAll(".tex[data-tex]:not([data-done])")) {
      try {
        window.katex.render(el.dataset.tex, el, { throwOnError: false, displayMode: !!el.closest(".formula") });
        el.dataset.done = "1";
      } catch {
        /* keep the plain-text fallback */
      }
    }
  };
  $("katex-js")?.addEventListener("load", render);
  render();
}

// ───────────── small builders ─────────────
const labelFor = (t, extra) => h("span", { class: "tok-label" }, tokChip(t), extra ? h("span", { class: "mono" }, extra) : null);

function placeholder(box) {
  box.replaceChildren();
  if (!state.tokens.length) {
    box.append(h("div", { class: "placeholder" }, "Type a sentence above to see this step."));
  } else if (weightsError) {
    box.append(
      h("div", { class: "placeholder error" },
        `Couldn't load the GPT-2 weights (${weightsError.message}). Tokenization above still works.`,
        h("button", { class: "btn", type: "button", onclick: () => (W ? compute() : loadWeights()) }, "Retry")),
    );
  } else {
    box.append(h("div", { class: "placeholder" }, h("span", { class: "spinner" }), "Downloading GPT-2 weights (about 10 MB, only once)…"));
  }
}

// Header row of a viz box: title (text and/or nodes) on the left, controls/legend on the right.
function vizHead(title, right = []) {
  return h("div", { class: "viz-head" }, h("div", { class: "panel-title", style: { margin: 0 } }, title), h("div", { class: "viz-head-right" }, right));
}

// replaceChildren() that skips null/false children (replaceChildren would print "null").
const fill = (el, ...children) => el.replaceChildren(...children.flat().filter((c) => c != null && c !== false));

const argmax = (arr) => {
  let b = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[b]) b = i;
  return b;
};
const norm = (v) => Math.sqrt(dot(v, v));

// ═════════════ 1. Tokenization ═════════════
function renderTokenization() {
  renderPretok();
  renderBPE();
  renderIds();
}

function chunkKind(text) {
  if (/^'(s|t|re|ve|m|ll|d)$/.test(text)) return "contraction";
  if (/^\s+$/.test(text)) return "spaces";
  const sp = text.startsWith(" ") ? "␣" : "";
  if (/^ ?\p{L}+$/u.test(text)) return `${sp}letters`;
  if (/^ ?\p{N}+$/u.test(text)) return `${sp}digits`;
  return `${sp}symbols`;
}

function selectChunk(i) {
  state.bpeChunk = i;
  state.bpeStep = 0;
  stopPlay();
  renderPretok();
  renderBPE();
}

function renderPretok() {
  const box = $("viz-pretok");
  box.replaceChildren();
  const { chunks } = state.enc;
  if (!chunks.length) return box.append(h("div", { class: "placeholder" }, "Type a sentence above to start."));
  box.append(
    h("div", { class: "panel-title" }, `${chunks.length} chunk${chunks.length === 1 ? "" : "s"} · click one to see its merges below`),
    h("div", { class: "chunks" },
      chunks.map((c, i) =>
        h("button", { class: "chunk", type: "button", "aria-pressed": String(i === state.bpeChunk), onclick: () => selectChunk(i) },
          h("span", { class: "chunk-text" }, tokText(c.text)),
          h("span", { class: "chunk-kind" }, chunkKind(c.text))))),
  );
}

function stopPlay() {
  clearInterval(playTimer);
  playTimer = null;
}

function togglePlay(K) {
  if (playTimer) return stopPlay(), renderBPE();
  if (state.bpeStep >= K) state.bpeStep = 0;
  playTimer = setInterval(() => {
    if (state.bpeStep >= K) {
      stopPlay();
    } else {
      state.bpeStep++;
    }
    renderBPE();
  }, 750);
  renderBPE();
}

function pairPositions(word, [a, b]) {
  const pos = [];
  for (let i = 0; i < word.length - 1; i++) {
    if (word[i] === a && word[i + 1] === b) {
      pos.push(i);
      i++;
    }
  }
  return pos;
}

function renderBPE() {
  const box = $("viz-bpe");
  box.replaceChildren();
  const { chunks } = state.enc;
  if (!chunks.length) return;
  const chunk = chunks[state.bpeChunk];
  const K = chunk.steps.length;
  const k = Math.min(state.bpeStep, K);
  const word = k === 0 ? Array.from(chunk.byteStr) : chunk.steps[k - 1].after;
  const next = chunk.steps[k];

  // chunk picker
  box.append(
    h("div", { class: "panel-title" }, "Chunk"),
    h("div", { class: "chunks", style: { marginBottom: "14px" } },
      chunks.map((c, i) =>
        h("button", { class: "chunk", type: "button", "aria-pressed": String(i === state.bpeChunk), onclick: () => selectChunk(i) },
          h("span", { class: "chunk-text" }, tokText(c.text)),
          h("span", { class: "chunk-kind" }, `${c.steps.length} merge${c.steps.length === 1 ? "" : "s"}`)))),
  );

  // characters → bytes → byte symbols
  const groups = Array.from(chunk.text).map((ch) => {
    const bytes = [...utf8.encode(ch)];
    return h("div", { class: "byte-group", style: { flexDirection: "column", alignItems: "center" } },
      h("span", { class: "byte-col" }, h("span", { class: "ch" }, ch === " " ? "␣" : ch === "\n" ? "↵" : ch === "\t" ? "⇥" : ch)),
      h("div", { style: { display: "flex", gap: "3px" } },
        bytes.map((b) =>
          h("div", { class: "byte-col" },
            h("span", { class: "hex" }, b.toString(16).toUpperCase().padStart(2, "0")),
            h("span", { class: "arrow-d" }, "↓"),
            h("span", { class: "bu" }, tokenizer.byteEncoder.get(b))))));
  });
  box.append(
    h("div", { class: "panel-title" }, "Characters → UTF-8 bytes (hex) → GPT-2's printable stand-in for each byte"),
    h("div", { class: "bpe-bytes" }, groups),
  );

  // stepper
  const go = (v) => {
    stopPlay();
    state.bpeStep = Math.max(0, Math.min(K, v));
    renderBPE();
  };
  const slider = h("input", { type: "range", min: 0, max: K, value: k, step: 1, "aria-label": "Merge step", disabled: K === 0 });
  slider.addEventListener("input", () => go(Number(slider.value)));
  box.append(
    h("div", { class: "panel-title" }, `Merge step ${k} of ${K}`),
    h("div", { class: "bpe-controls" },
      h("button", { class: "btn", type: "button", onclick: () => go(0), disabled: k === 0, "aria-label": "First step" }, "⏮"),
      h("button", { class: "btn", type: "button", onclick: () => go(k - 1), disabled: k === 0, "aria-label": "Previous step" }, "◀"),
      h("button", { class: "btn primary", type: "button", onclick: () => togglePlay(K), disabled: K === 0 }, playTimer ? "❚❚ Pause" : "▶ Play"),
      h("button", { class: "btn", type: "button", onclick: () => go(k + 1), disabled: k === K, "aria-label": "Next step" }, "▶"),
      h("button", { class: "btn", type: "button", onclick: () => go(K), disabled: k === K, "aria-label": "Last step" }, "⏭"),
      slider),
  );

  // symbols
  const hot = new Set();
  if (next) for (const p of pairPositions(word, next.pair)) hot.add(p).add(p + 1);
  const fresh = new Set(k > 0 ? chunk.steps[k - 1].positions : []);
  const done = k === K;
  box.append(
    h("div", { class: "bpe-symbols" },
      word.map((sym, i) => {
        const tok = done ? chunk.tokens[i] : null;
        return h("span", {
          class: `sym${hot.has(i) ? " pair" : ""}${fresh.has(i) && !hot.has(i) && !done ? " just-merged" : ""}${tok ? " final" : ""}`,
          style: tok ? `--tc: var(--tok${tok.index % 10})` : null,
          "data-tok": tok ? tok.index : null,
          title: `bytes: ${tokenizer.tokenBytes(sym).map((b) => b.toString(16).toUpperCase().padStart(2, "0")).join(" ")}`,
        }, sym);
      })),
  );

  // explanation of this step
  const msg = h("div", { class: "bpe-msg" });
  if (next) {
    const where = pairPositions(word, next.pair).length;
    msg.append(
      "Next: rule ", h("b", null, `#${int(next.rank + 1)}`), " of 50,000 merges ",
      h("span", { class: "mono" }, next.pair[0]), " + ", h("span", { class: "mono" }, next.pair[1]), " → ",
      h("span", { class: "mono" }, next.pair.join("")),
      where > 1 ? ` (in ${where} places at once)` : "",
      ". Of all neighbouring pairs it has the lowest rule number, meaning it was learned earliest.",
    );
  } else if (word.length === 1 && K === 0) {
    msg.append("A single byte is already a token. Nothing to merge.");
  } else {
    msg.append(
      K ? "Done: none of the remaining neighbouring pairs is in the merge list. " : "None of the neighbouring pairs is in the merge list, so nothing merges. ",
      `This chunk becomes ${chunk.tokens.length} token${chunk.tokens.length === 1 ? "" : "s"}: `,
      h("span", { class: "tok-row", style: { display: "inline-flex", verticalAlign: "middle" } }, chunk.tokens.map((t) => tokChip(t, { showId: true }))),
    );
  }
  box.append(msg);

  // all candidate pairs at this step
  if (word.length > 1) {
    const cands = [];
    for (let i = 0; i < word.length - 1; i++) {
      const r = tokenizer.ranks.get(word[i] + " " + word[i + 1]);
      cands.push({ pair: `${word[i]} ${word[i + 1]}`, r });
    }
    const minR = Math.min(...cands.map((c) => c.r ?? Infinity));
    box.append(
      h("div", { class: "panel-title", style: { marginTop: "12px" } }, "Neighbouring pairs right now and their rule numbers"),
      h("ul", { class: "merge-list" },
        cands.map((c) =>
          h("li", null, h("button", {
            type: "button",
            "aria-current": String(c.r === minR && c.r !== undefined),
            class: c.r === undefined ? "done" : null,
            tabindex: -1,
          }, `${c.pair}  ${c.r === undefined ? "not in list" : "#" + int(c.r + 1)}`)))),
    );
  }

  // merge history
  if (K) {
    box.append(
      h("div", { class: "panel-title", style: { marginTop: "12px" } }, "All merges for this chunk (click to jump)"),
      h("ol", { class: "merge-list" },
        chunk.steps.map((stp, i) =>
          h("li", null, h("button", {
            type: "button",
            class: i < k ? "done" : null,
            "aria-current": String(i === k - 1),
            onclick: () => go(i + 1),
          }, `${i + 1}. ${stp.pair[0]} + ${stp.pair[1]}`)))),
    );
  }
}

function renderIds() {
  const box = $("viz-ids");
  box.replaceChildren();
  const { enc } = state;
  const toks = state.tokens;
  if (!toks.length) return;
  box.append(h("div", { class: "tok-row" }, toks.map((t) => tokChip(t, { showId: true }))));
  const chars = Array.from(enc.text).length;
  const bytes = utf8.encode(enc.text).length;
  box.append(
    h("div", { class: "stats" },
      h("span", null, h("b", null, chars), "characters"),
      h("span", null, h("b", null, bytes), "bytes"),
      h("span", null, h("b", null, enc.chunks.length), "chunks"),
      h("span", null, h("b", null, enc.tokens.length), "tokens"),
      h("span", null, h("b", null, (chars / enc.tokens.length).toFixed(1)), "characters per token")),
    h("div", { class: "ids-array" }, h("span", { class: "k" }, "model input = "), `[${toks.map((t) => t.id).join(", ")}]`),
  );
  if (toks.some((t) => t.partial)) {
    box.append(h("div", { class: "callout" },
      "Some tokens hold only part of a character's bytes (shown as ‹hex›). GPT-2 was trained mostly on English text, so other scripts often break into small byte-level pieces. That is why the same sentence costs more tokens in other languages."));
  }
  if (state.truncated) {
    box.append(h("div", { class: "callout warn" }, `This text has ${enc.tokens.length} tokens. To keep the pictures readable, the steps below use only the first ${MAX_TOKENS}.`));
  }
}

// ═════════════ 2. Token embedding ═════════════
function renderEmbedding() {
  const wte = $("viz-wte");
  const sim = $("viz-sim");
  const { res } = state;
  if (!res) return placeholder(wte), placeholder(sim);
  const toks = state.tokens;
  const n = toks.length;
  const scale = robustScale(res.tokEmb);

  fill(wte,
    vizHead(`${n} row${n === 1 ? "" : "s"} of W_E, one per token`, legend(scale)),
    h("div", { class: "emb-layout" }, matrixSchematic(toks), h("div", null,
      heatRows(res.tokEmb, {
        labels: toks.map((t) => labelFor(t, `#${t.id}`)),
        scale,
        describe: (r, d, v) => `${tokTip(toks[r])} · row ${int(toks[r].id)} of W<sub>E</sub><br>dim ${d}: <b>${fmt(v, 4)}</b>`,
      }),
      focusValues("e", res.tokEmb[state.focus], toks[state.focus], `W_E[${toks[state.focus].id}]`),
      h("p", { class: "sub" },
        `Each row is 768 float32 numbers = 3,072 bytes. This page downloaded exactly these rows from model.safetensors with HTTP Range requests, not the whole 154 MB matrix.`))),
  );

  const size = cellSize(n, 520);
  let best = null;
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
    if (toks[i].id === toks[j].id) continue;
    const c = cosine(res.tokEmb[i], res.tokEmb[j]);
    if (!best || c > best.c) best = { i, j, c };
  }
  fill(sim,
    vizHead("Cosine similarity between token embeddings", legend(1, { digits: 0 })),
    matrixGrid({
      n, size,
      rowLabels: toks.map((t) => tokChip(t)),
      colLabels: toks.map((t) => tokChip(t)),
      cell: (i, j) => {
        const c = cosine(res.tokEmb[i], res.tokEmb[j]);
        return {
          bg: divergingRGB(c, 1),
          text: fmt(c, 2),
          tip: `cos(${tokTip(toks[i])}, ${tokTip(toks[j])}) = <b>${fmt(c, 3)}</b>${toks[i].id === toks[j].id ? "<br>same token, so the same vector" : ""}`,
        };
      },
    }),
    best ? h("p", { class: "sub" }, "Most similar pair of different tokens here: ", tokChip(toks[best.i]), " and ", tokChip(toks[best.j]), ` (${fmt(best.c, 2)}).`) : null,
  );
}

function matrixSchematic(toks) {
  const uniq = [...new Map(toks.map((t) => [t.id, t])).values()].sort((a, b) => a.id - b.id);
  const W0 = 220;
  const top = 34;
  const matX = 18;
  const matW = 54;
  const H = Math.max(300, uniq.length * 17 + top + 26);
  const matH = H - top - 22;
  const svg = s("svg", { class: "matrix-svg", viewBox: `0 0 ${W0} ${H}`, role: "img", "aria-label": "Embedding matrix with the looked-up rows marked" });
  svg.append(
    s("text", { class: "caption", x: matX, y: 12 }, "W_E  (50,257 × 768)"),
    s("text", { class: "dimtext", x: matX, y: 27 }, "768 columns →"),
    s("rect", { class: "mat", x: matX, y: top, width: matW, height: matH, rx: 3 }),
    s("text", { class: "dimtext", x: matX + matW + 4, y: top + 8 }, "row 0"),
    s("text", { class: "dimtext", x: matX + matW + 4, y: top + matH }, "row 50,256"),
  );
  // place labels without overlap
  const ys = uniq.map((t) => top + (t.id / CFG.nVocab) * matH);
  const lab = [];
  let prev = top + 14;
  for (const y of ys) {
    const ly = Math.max(y, prev + 16);
    lab.push(ly);
    prev = ly;
  }
  const overflow = prev - (top + matH - 12);
  if (overflow > 0) for (let i = lab.length - 1, lim = top + matH - 12; i >= 0; i--, lim -= 16) lab[i] = Math.min(lab[i], lim);
  uniq.forEach((t, i) => {
    const y = ys[i];
    const ly = lab[i];
    const txt = `#${t.id} ${tokPlain(t.text)}`;
    const label = txt.length > 18 ? txt.slice(0, 17) + "…" : txt;
    const g = s("g", { "data-tok": t.index, style: "cursor:pointer" });
    g.append(
      s("line", { class: "rowline", x1: matX, x2: matX + matW, y1: y, y2: y, style: "stroke: var(--ink)" }),
      s("path", { class: "leader", d: `M ${matX + matW} ${y} C ${matX + matW + 14} ${y}, ${matX + matW + 14} ${ly}, ${matX + matW + 26} ${ly}` }),
      s("rect", { x: matX + matW + 26, y: ly - 7.5, width: 8 + label.length * 6.7, height: 15, rx: 3, style: `fill: var(--tok${t.index % 10})` }),
      s("text", { x: matX + matW + 30, y: ly + 4 }, label),
    );
    svg.append(g);
  });
  return svg;
}

function focusValues(sym, vec, tok, expr) {
  const first = Array.from(vec.slice(0, 8), (v) => fmt(v, 3)).join(", ");
  return h("div", { class: "vals" }, tokChip(tok), " ", h("span", { class: "k" }, `${sym} = ${expr} = `), `[${first}, … ${vec.length - 8} more]`);
}

// ═════════════ 3. Positional encoding ═════════════
const SIN_DIMS = [0, 64, 192, 400]; // fast → slow waves
const posCache = {};

// Everything about one position table (learned or sinusoidal) that 3.1 shows.
function positionTable(kind) {
  if (posCache[kind]) return posCache[kind];
  const rows = Array.from({ length: POSITION_ROWS }, (_, r) => (kind === "learned" ? W.wpe.subarray(r * D, (r + 1) * D) : sinusoidal(r)));
  const P = rows.length;
  // How much each dimension varies along positions 1..P-1, and how smoothly (lag-1 autocorrelation).
  const dims = [];
  for (let d = 0; d < D; d++) {
    let m = 0;
    for (let r = 1; r < P; r++) m += rows[r][d];
    m /= P - 1;
    let v = 0;
    let ac = 0;
    for (let r = 1; r < P; r++) {
      const c = rows[r][d] - m;
      v += c * c;
      if (r > 1) ac += c * (rows[r - 1][d] - m);
    }
    dims.push({ d, std: Math.sqrt(v / (P - 1)), ac: v ? ac / v : 0 });
  }
  const top = kind === "learned" ? [...dims].sort((a, b) => b.std - a.std).slice(0, 4).map((x) => x.d) : SIN_DIMS;
  const norms = rows.map(norm);
  const sorted = norms.slice(1).sort((a, b) => a - b);
  const sim = rows.map((a) => rows.map((b) => cosine(a, b)));
  posCache[kind] = {
    rows, top, sim,
    scale: kind === "learned" ? robustScale(rows, 0.95) : 1,
    smoothDims: dims.filter((x) => x.ac > 0.9 && x.std > 0.02).length,
    row0Ratio: norms[0] / sorted[sorted.length >> 1],
  };
  return posCache[kind];
}

function renderPosition() {
  const wpe = $("viz-wpe");
  const sum = $("viz-sum");
  const { res } = state;
  if (!res) return placeholder(wpe), placeholder(sum);
  const toks = state.tokens;
  const n = toks.length;
  const i = state.focus;
  const learned = state.posView === "learned";
  const T = positionTable(state.posView);
  const used = Math.min(n, POSITION_ROWS);
  const viewBtn = (key, label) =>
    h("button", { type: "button", "aria-pressed": String(state.posView === key), onclick: () => ((state.posView = key), renderPosition()) }, label);

  const simCanvas = heatImage((r) => T.sim[r], POSITION_ROWS, POSITION_ROWS, {
    scale: 1,
    rowPx: 2,
    describe: (r, c, v) => `cos(position ${r}, position ${c}) = <b>${fmt(v, 3)}</b>`,
  });
  simCanvas.style.maxWidth = "256px";

  fill(wpe,
    vizHead(learned ? `GPT-2's W_P, first ${POSITION_ROWS} of 1,024 rows` : `Sinusoidal encoding, positions 0–${POSITION_ROWS - 1}`,
      [h("span", { class: "seg-toggle" }, viewBtn("learned", "Learned (GPT-2)"), viewBtn("sin", "Sinusoidal (2017)")), legend(T.scale)]),
    h("div", { class: "pos-image" },
      h("div", { class: "pos-axis" }, h("span", null, "pos 0"), h("span", null, `pos ${POSITION_ROWS - 1}`)),
      heatImage((r) => T.rows[r], POSITION_ROWS, D, {
        scale: T.scale,
        rowPx: 2,
        bracket: [0, used],
        describe: (r, d, v) => `position ${r} · dim ${d}<br><b>${fmt(v, 4)}</b>${r < n ? `<br>used by ${tokTip(toks[r])}` : ""}`,
      }),
      h("span"), h("div", { class: "heat-axis" }, h("span", null, "dim 0"), h("span", null, "dim 767"))),
    h("p", { class: "sub" },
      learned
        ? `Each row is one position and each column one of the 768 dimensions. The bracket on the left marks the ${used} position${used === 1 ? "" : "s"} your sentence uses. Most numbers are close to zero. About ${T.smoothDims} dimensions carry the position signal, and they change smoothly from row to row (plotted below).`
        : "Every column is a sine or cosine wave. Columns on the left oscillate quickly and columns on the right very slowly, so every row gets a unique combination. GPT-2 does not use this formula. It is shown for comparison."),
    h("div", { class: "viz-row", style: { marginTop: "16px" } },
      h("div", { class: "grow" },
        h("div", { class: "panel-title" }, learned ? "The 4 dimensions that change most with position" : "4 of the 768 waves"),
        lineChart(T.top.map((d) => ({ label: `dim ${d}`, values: T.rows.map((r) => r[d]) })), { mark: used }),
        h("p", { class: "sub" }, learned
          ? "GPT-2 was never told to use waves. These smooth curves came out of training."
          : "Each dimension pair uses a different wavelength, from about 6 positions (dim 0) to about 61,000 (dim 767).")),
      h("div", { style: { flex: "0 1 280px" } },
        h("div", { class: "panel-title" }, "Similarity between positions"),
        simCanvas,
        h("p", { class: "sub" },
          "Cosine similarity of every pair of position vectors (row = one position, column = another). The strong band along the diagonal means nearby positions get similar vectors, and similarity fades with distance.",
          learned ? ` Row 0 is special: its vector is ${T.row0Ratio.toFixed(1)}× longer than a typical row.` : ""))),
    learned
      ? h("div", { style: { marginTop: "14px" } },
          h("div", { class: "panel-title" }, "The rows your sentence uses"),
          heatRows(res.posEmb, {
            labels: toks.map((t, r) => labelFor(t, `pos ${r}`)),
            scale: robustScale(res.posEmb, 0.95),
            describe: (r, d, v) => `position ${r} (${tokTip(toks[r])}) · dim ${d}<br><b>${fmt(v, 4)}</b>`,
          }))
      : null,
  );

  // 3.2 sum, for the selected token
  const trio = [res.tokEmb[i], res.posEmb[i], res.x[i]];
  const sc = robustScale(trio);
  const arith = (d) =>
    `${tokTip(toks[i])} · dim ${d}<br>e ${fmt(trio[0][d], 4)} + p ${fmt(trio[1][d], 4)} = <b>x ${fmt(trio[2][d], 4)}</b>`;
  const dup = findDuplicate(toks);
  fill(sum,
    vizHead(["Selected token ", tokChip(toks[i]), ` at position ${i}`], legend(sc)),
    heatRows(trio, {
      labels: [h("span", null, "token  e", h("sub", null, "i")), h("span", null, "+ position  p", h("sub", null, "i")), h("b", null, "= input  x", h("sub", null, "i"))],
      scale: sc,
      rowHeight: 22,
      describe: (_r, d) => arith(d),
    }),
    h("div", { class: "panel-title", style: { marginTop: "18px" } }, `All tokens: X has shape ${n} × 768`),
    heatRows(res.x, {
      labels: toks.map((t, r) => labelFor(t, `pos ${r}`)),
      scale: robustScale(res.x),
      describe: (r, d, v) => `${tokTip(toks[r])} · dim ${d}<br>e ${fmt(res.tokEmb[r][d], 4)} + p ${fmt(res.posEmb[r][d], 4)} = <b>${fmt(v, 4)}</b>`,
    }),
    dup
      ? h("div", { class: "callout" }, tokChip(toks[dup[0]]), ` appears at positions ${dup[0]} and ${dup[1]}. Their token embeddings are identical (cosine 1.000). After adding positions, the cosine is `,
          h("b", null, fmt(cosine(res.x[dup[0]], res.x[dup[1]]), 3)), ", so the model can now tell the two apart.")
      : h("div", { class: "callout" }, "Try a sentence with a repeated word, like ",
          h("span", { class: "example-link", role: "button", tabindex: 0, onclick: () => setText("I saw the cat and the dog") }, "I saw the cat and the dog"),
          ", to see the same token become two different vectors."),
  );
}

function findDuplicate(toks) {
  const seen = new Map();
  for (const t of toks) {
    if (seen.has(t.id)) return [seen.get(t.id), t.index];
    seen.set(t.id, t.index);
  }
  return null;
}

// ═════════════ 4. Self-attention ═════════════
function renderAttention() {
  const ids = ["viz-ln", "viz-qkv", "viz-scores", "viz-mask", "viz-softmax", "viz-weighted", "viz-heads", "viz-proj"];
  const { res } = state;
  const ctl = $("attn-controls");
  if (!res) {
    fill(ctl, );
    ids.forEach((id) => placeholder($(id)));
    return;
  }
  const profiles = res.heads.map((hd) => headProfile(hd.probs));
  renderControls(profiles);
  renderLN();
  renderQKV();
  renderScores();
  renderMask();
  renderSoftmax();
  renderWeighted();
  renderHeads(profiles);
  renderProj();
}

function renderControls(profiles) {
  const toks = state.tokens;
  fill($("attn-controls"),
    h("div", { class: "ctl" },
      h("span", { class: "lbl" }, "Head"),
      h("div", { class: "head-pick" },
        profiles.map((p, hd) =>
          h("button", { type: "button", "aria-pressed": String(hd === state.head), title: `Head ${hd}: ${HEAD_DESC[p.label]}`, onclick: () => setHead(hd) },
            hd, h("small", null, HEAD_TAG[p.label]))))),
    h("div", { class: "ctl tokens" },
      h("span", { class: "lbl" }, "Query"),
      h("div", { class: "tok-row tight" }, toks.map((t) => tokChip(t)))),
  );
}

function renderLN() {
  const { res } = state;
  const i = state.focus;
  const t = state.tokens[i];
  const L = res.ln[i];
  const x = res.x[i];
  const describe = (_r, d) =>
    `${tokTip(t)} · dim ${d}<br>(x − μ) / σ = (${fmt(x[d], 3)} − ${fmt(L.mean, 3)}) / ${fmt(L.std, 3)} = ${fmt(L.norm[d], 3)}<br>× γ ${fmt(W.lnW[d], 3)} + β ${fmt(W.lnB[d], 3)} = <b>${fmt(L.out[d], 3)}</b>`;
  const one = (vec, label) => {
    const sc = robustScale([vec]);
    return h("div", { style: { marginBottom: "10px" } },
      h("div", { class: "viz-head", style: { marginBottom: "4px" } }, h("span", { style: { fontSize: "14px" } }, label), legend(sc)),
      heatRows([vec], { scale: sc, rowHeight: 20, describe, axis: false }));
  };
  const range = (v) => `${fmt(Math.min(...v), 2)} … ${fmt(Math.max(...v), 2)}`;
  fill($("viz-ln"),
    vizHead(["Selected token ", tokChip(t)]),
    one(x, h("span", null, "x", h("sub", null, "i"), ` · range ${range(x)}`)),
    one(L.norm, h("span", null, "(x", h("sub", null, "i"), " − μ) / σ", ` · mean 0, std 1`)),
    one(L.out, h("b", null, "x̂", h("sub", null, "i"), ` = γ ⊙ … + β · range ${range(L.out)}`)),
    h("div", { class: "stats" },
      h("span", null, h("b", null, fmt(L.mean, 4)), "mean μ"),
      h("span", null, h("b", null, fmt(L.std, 4)), "std σ"),
      h("span", null, h("b", null, "768"), "dims, normalized separately for each token")),
  );
}

function renderQKV() {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const hd = state.head;
  const seg = [hd * HD, (hd + 1) * HD];
  const rowHeight = n > 16 ? 8 : n > 10 ? 11 : 14;
  const shape = (cls, name, dims, sub) => h("div", { class: `shape ${cls}` }, name, h("small", null, dims), sub ? h("small", null, sub) : null);
  const group = (key, name, cls, mats) => {
    const sc = robustScale(mats);
    return h("div", { class: "qkv-group" },
      h("div", { class: "panel-title" }, h("span", { class: `${cls}-ink` }, `${key} · ${name}`), legend(sc)),
      heatRows(mats, {
        labels: toks.map((t) => tokChip(t)),
        scale: sc,
        rowHeight,
        boundaries: HD,
        segment: seg,
        describe: (r, d, v) => `${key.toLowerCase()} of ${tokTip(toks[r])} · dim ${d}<br>head ${Math.floor(d / HD)}, slot ${d % HD}: <b>${fmt(v, 3)}</b>`,
        onPick: (_r, d) => setHead(Math.floor(d / HD)),
      }));
  };
  fill($("viz-qkv"),
    h("div", { class: "shapes" },
      shape("", "X̂", `${n} × 768`),
      h("span", { class: "op" }, "·"),
      shape("", "W_qkv", "768 × 2304", "c_attn"),
      h("span", { class: "op" }, "+ b ="),
      shape("", "QKV", `${n} × 2304`),
      h("span", { class: "op" }, "→"),
      shape("q", "Q", `${n} × 768`),
      shape("k", "K", `${n} × 768`),
      shape("v", "V", `${n} × 768`),
      h("span", { class: "op" }, "→"),
      shape("", "12 heads", `each ${n} × 64`)),
    h("p", { class: "sub", style: { marginTop: 0 } }, `Thin lines split the 768 dimensions into 12 head slices of 64. Head ${hd} uses dims ${seg[0]}–${seg[1] - 1}.`),
    group("Q", "queries", "q", res.q),
    group("K", "keys", "k", res.k),
    group("V", "values", "v", res.v),
  );
}

function scoreGrid(n, size, cell) {
  const toks = state.tokens;
  return matrixGrid({
    n, size,
    corner: h("span", null, "query ↓", h("br"), "key →"),
    rowLabels: toks.map((t) => tokChip(t)),
    colLabels: toks.map((t) => tokChip(t)),
    cell,
  });
}

function renderScores() {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const hd = res.heads[state.head];
  const size = cellSize(n, 460);
  const sc = robustScale(hd.scores, 1);
  const pinned = state.scoreCell || [state.focus, argmax(hd.probs[state.focus])];
  const inspector = h("div", { class: "inspector" });
  const inspect = (i, j) => fillInspector(inspector, i, j);
  const grid = scoreGrid(n, size, (i, j) => {
    const v = hd.scores[i][j];
    return {
      bg: divergingRGB(v, sc),
      text: fmt(v, 1),
      cls: `${j > i ? "future" : ""}${i === pinned[0] && j === pinned[1] ? " sel" : ""}`,
      tip: `q of ${tokTip(toks[i])} · k of ${tokTip(toks[j])} = <b>${fmt(v, 3)}</b>${j > i ? "<br>a future token, masked in the next step" : ""}`,
      onEnter: () => inspect(i, j),
      onClick: () => {
        state.scoreCell = [i, j];
        renderScores();
      },
    };
  });
  grid.addEventListener("pointerleave", () => inspect(...pinned));
  inspect(...pinned);
  fill($("viz-scores"),
    vizHead(`Head ${state.head}: raw scores S = Q·Kᵀ (${n} × ${n})`, legend(sc, { digits: 1 })),
    h("div", { class: "viz-row" }, h("div", null, grid, h("p", { class: "axis-note" }, "Faded cells are future tokens. They are computed, then masked in 4.4.")), inspector),
  );
}

function fillInspector(box, i, j) {
  const { res } = state;
  const toks = state.tokens;
  const hd = res.heads[state.head];
  const q = hd.q[i];
  const k = hd.k[j];
  const prod = Float32Array.from(q, (v, d) => v * k[d]);
  const sum = hd.scores[i][j];
  const qkScale = robustScale([q, k], 1);
  const pScale = robustScale([prod], 1);
  const top = [...prod.keys()].sort((a, b) => Math.abs(prod[b]) - Math.abs(prod[a])).slice(0, 3);
  const tip = (d) => `dim ${d}: q ${fmt(q[d], 3)} × k ${fmt(k[d], 3)} = <b>${fmt(prod[d], 3)}</b>`;
  fill(box,
    h("div", { class: "panel-title" }, "Dot-product inspector"),
    h("div", { class: "eq" }, "q of ", tokChip(toks[i]), " · k of ", tokChip(toks[j]), " = ", h("b", null, fmt(sum, 3))),
    h("div", { class: "bars-block" }, h("div", { class: "lbl" }, h("span", { class: "q-ink" }, `q${i} · head ${state.head} (64 numbers)`)), bars(q, { scale: qkScale, describe: tip })),
    h("div", { class: "bars-block" }, h("div", { class: "lbl" }, h("span", { class: "k-ink" }, `k${j} · head ${state.head} (64 numbers)`)), bars(k, { scale: qkScale, describe: tip })),
    h("div", { class: "bars-block" },
      h("div", { class: "lbl" }, h("span", null, "products q", h("sub", null, "d"), " × k", h("sub", null, "d")), h("span", null, `sum = ${fmt(sum, 3)}`)),
      bars(prod, { scale: pScale, describe: tip })),
    h("div", { class: "contrib" }, "largest terms: ", top.map((d) => `dim ${d} (${fmt(prod[d], 2)})`).join(", "),
      `. Positive products (red) push the score up, negative ones (blue) pull it down. Next step: ${fmt(sum, 2)} ÷ 8 = ${fmt(sum / 8, 3)}.`),
  );
}

function renderMask() {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const hd = res.heads[state.head];
  const size = cellSize(n, 460);
  const finite = hd.scaled.map((row, i) => row.filter((_, j) => !state.mask || j <= i));
  const sc = robustScale(finite, 1);
  const grid = scoreGrid(n, size, (i, j) => {
    if (state.mask && j > i) {
      return { cls: "masked", text: "−∞", tip: `${tokTip(toks[i])} can't see the future token ${tokTip(toks[j])}<br>score → <b>−∞</b> → weight 0` };
    }
    const v = hd.scaled[i][j];
    return {
      bg: divergingRGB(v, sc),
      text: fmt(v, 2),
      cls: i === state.focus ? "rowsel" : "",
      tip: `${fmt(hd.scores[i][j], 3)} ÷ 8 = <b>${fmt(v, 3)}</b>`,
    };
  });
  // Why scale: compare the top weight in the selected row with and without ÷√64.
  const i = state.focus;
  const raw = hd.scores[i];
  const upto = state.mask ? i : n - 1;
  const softTop = (row, div) => {
    let m = -Infinity;
    for (let j = 0; j <= upto; j++) m = Math.max(m, row[j] / div);
    let sum = 0;
    for (let j = 0; j <= upto; j++) sum += Math.exp(row[j] / div - m);
    return 1 / sum;
  };
  const toggle = h("label", { class: "toggle" }, h("input", { type: "checkbox", checked: state.mask }), "causal mask");
  toggle.querySelector("input").addEventListener("change", (e) => {
    state.mask = e.target.checked;
    renderMask();
    renderSoftmax();
    renderWeighted();
  });
  fill($("viz-mask"),
    vizHead(`Head ${state.head}: S / √64 with the mask applied`, [toggle, legend(sc, { digits: 1 })]),
    grid,
    !state.mask
      ? h("div", { class: "callout warn" },
          "Mask off: GPT-2 never computes this. It is shown only so you can see what the mask removes. In 4.5 and 4.6, tokens now also mix in the tokens after them.")
      : null,
    n > 1
      ? h("div", { class: "callout" }, "Why scale? In the row of ", tokChip(toks[i]), ", the largest attention weight is ",
          h("b", null, pct(softTop(raw, 8))), " with ÷8, but it would be ", h("b", null, pct(softTop(raw, 1))),
          " without it. Unscaled scores make softmax much closer to all-or-nothing.")
      : null,
  );
}

const probsOf = (hd) => (state.mask ? hd.probs : hd.probsNoMask);
const maskOffTag = () => (state.mask ? null : h("span", { class: "maskoff" }, "mask off · not what GPT-2 computes"));

function renderSoftmax() {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const hd = res.heads[state.head];
  const P = probsOf(hd);
  const size = cellSize(n, 420);
  const i = state.focus;
  const grid = scoreGrid(n, size, (r, j) => {
    const p = P[r][j];
    if (state.mask && j > r) return { cls: "masked", text: "0", tip: "masked → weight 0" };
    return {
      bg: sequentialRGB(p),
      text: p >= 0.995 ? "1" : fmt(p, 2).replace(/^0/, ""),
      cls: r === i ? "rowsel" : "",
      tip: `${tokTip(toks[r])} → ${tokTip(toks[j])}: <b>${pct(p, 1)}</b>`,
    };
  });
  // Row breakdown for the selected token.
  const upto = state.mask ? i : n - 1;
  const ex = [];
  let total = 0;
  for (let j = 0; j <= upto; j++) {
    ex.push(Math.exp(hd.scaled[i][j]));
    total += ex[j];
  }
  const fmtE = (v) => (v >= 1e4 || v < 1e-3 ? v.toExponential(2) : v.toPrecision(3));
  const table = h("table", { class: "sm-table" },
    h("thead", null, h("tr", null, h("th", null, "key"), h("th", null, "score s′"), h("th", null, "e^s′"), h("th", null, "weight a"), h("th", null, ""))),
    h("tbody", null, ex.map((e, j) =>
      h("tr", null,
        h("td", null, tokChip(toks[j])),
        h("td", null, fmt(hd.scaled[i][j], 3)),
        h("td", null, fmtE(e)),
        h("td", null, pct(e / total, 1)),
        h("td", { class: "barcell" }, h("div", { class: "wbar", style: { width: `${(e / total) * 100}%` } }))))),
    h("tfoot", null, h("tr", null, h("td", null, "sum"), h("td", null, ""), h("td", null, fmtE(total)), h("td", null, "100%"), h("td", null, ""))),
  );
  fill($("viz-softmax"),
    vizHead([`Head ${state.head}: attention weights A = softmax(S′)`, maskOffTag()], seqLegend()),
    h("div", { class: "viz-row" },
      h("div", null, grid, h("p", { class: "axis-note" }, "Each row adds up to 1.")),
      h("div", { class: "grow", style: { maxWidth: "440px" } },
        h("div", { class: "panel-title" }, "Who listens to whom (click a query on the left)"),
        attentionLines(toks, P, i, { onFocus: setFocus }))),
    h("div", { style: { marginTop: "16px" } },
      h("div", { class: "panel-title" }, "Softmax for the row of ", tokChip(toks[i])),
      h("div", { class: "scroll-x" }, table),
      h("p", { class: "sub" }, "Exponentiating makes every score positive and stretches the differences. Dividing by the sum makes the row add up to 1. Real implementations subtract the row maximum first so e^s′ can't overflow, which gives the same weights.")),
  );
}

function renderWeighted() {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const hd = res.heads[state.head];
  const P = probsOf(hd);
  const i = state.focus;
  const js = [];
  for (let j = 0; j < n; j++) if (P[i][j] > 0) js.push(j);
  const weighted = js.map((j) => Float32Array.from(hd.v[j], (v) => v * P[i][j]));
  const z = new Float32Array(HD);
  for (const w of weighted) for (let d = 0; d < HD; d++) z[d] += w[d];
  const sc = robustScale(js.map((j) => hd.v[j]).concat([z]));
  const describe = (r, d) => {
    if (r < js.length) {
      const j = js[r];
      return `${pct(P[i][j], 1)} × v of ${tokTip(toks[j])} [dim ${d}] = ${fmt(P[i][j], 3)} × ${fmt(hd.v[j][d], 3)} = <b>${fmt(weighted[r][d], 3)}</b>`;
    }
    const terms = js.map((j, r2) => ({ j, v: weighted[r2][d] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 4);
    return `z dim ${d} = <b>${fmt(z[d], 3)}</b><br>biggest terms: ${terms.map((t) => `${tokTip(toks[t.j])} ${fmt(t.v, 3)}`).join(", ")}`;
  };
  fill($("viz-weighted"),
    vizHead([`Head ${state.head}: output for `, tokChip(toks[i]), maskOffTag()], legend(sc)),
    heatRows([...weighted, z], {
      labels: [
        ...js.map((j) => h("span", { class: "tok-label" }, h("span", { class: "mono" }, pct(P[i][j], 1)), "×", tokChip(toks[j]))),
        h("b", null, "= z", h("sub", null, i), " (64)"),
      ],
      scale: sc,
      rowHeight: js.length > 16 ? 10 : 16,
      describe,
    }),
    h("p", { class: "sub" },
      "Each row is one token's value vector (this head's 64 numbers) multiplied by its attention weight, so low weights fade out. Adding the rows gives z. Hover a column to see the sum."),
  );
}

function renderHeads(profiles) {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const i = state.focus;
  const cards = res.heads.map((hd, k) => {
    const cv = h("canvas", { width: n, height: n });
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(n, n);
    const masked = getComputedStyle(document.documentElement).getPropertyValue("--surface-2").trim();
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < n; j++) {
        const c = j > r ? null : sequentialRGB(hd.probs[r][j]);
        const o = (r * n + j) * 4;
        if (c) img.data.set([c[0], c[1], c[2], 255], o);
        else img.data.set([0, 0, 0, 0], o);
      }
    }
    ctx.putImageData(img, 0, 0);
    cv.style.background = masked;
    const p = profiles[k];
    const tip = `<b>Head ${k}</b> ${HEAD_DESC[p.label]}<br>self ${pct(p.self, 0)} · previous ${pct(p.prev, 0)} · first ${pct(p.first, 0)} · other ${pct(p.other, 0)}`;
    const card = h("button", { class: "head-card", type: "button", "aria-pressed": String(k === state.head), onclick: () => setHead(k) },
      h("div", { class: "ttl" }, `Head ${k}`, h("span", null, HEAD_TAG[p.label])),
      cv,
      h("div", { class: "profile" }, ["self", "prev", "first", "other"].map((key) => h("i", { class: `p-${key}`, style: { width: `${p[key] * 100}%` } }))));
    card.addEventListener("pointermove", (ev) => showTip(tip, ev));
    card.addEventListener("pointerleave", hideTip);
    return card;
  });
  const z = res.concat[i];
  const hdSel = state.head;
  fill($("viz-heads"),
    h("div", { class: "profile-legend" },
      h("span", null, h("i", { class: "p-self" }), "self"),
      h("span", null, h("i", { class: "p-prev" }), "previous token"),
      h("span", null, h("i", { class: "p-first" }), "first token"),
      h("span", null, h("i", { class: "p-other" }), "everything else"),
      h("span", { style: { color: "var(--muted)" } }, "average weight per row, on this sentence")),
    h("div", { class: "heads-grid" }, cards),
    h("div", { style: { marginTop: "18px" } },
      vizHead(["Concatenated output for ", tokChip(toks[i])], legend(robustScale([z]))),
      heatRows([z], {
        labels: [h("span", null, "z", h("sub", null, i), " (768)")],
        scale: robustScale([z]),
        rowHeight: 22,
        boundaries: HD,
        segment: [hdSel * HD, (hdSel + 1) * HD],
        describe: (_r, d, v) => `head ${Math.floor(d / HD)}, slot ${d % HD}: <b>${fmt(v, 3)}</b>`,
        onPick: (_r, d) => setHead(Math.floor(d / HD)),
      })),
  );
}

function renderProj() {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const i = state.focus;
  const trio = [res.x[i], res.attnOut[i], res.resid[i]];
  const sc = robustScale(trio);
  fill($("viz-proj"),
    vizHead(["Selected token ", tokChip(toks[i])], legend(sc)),
    heatRows(trio, {
      labels: [
        h("span", null, "input  x", h("sub", null, i)),
        h("span", null, "+ attention  z", h("sub", null, i), "W", h("sub", null, "O"), " + b"),
        h("b", null, "= h", h("sub", null, i)),
      ],
      scale: sc,
      rowHeight: 22,
      describe: (_r, d) => `dim ${d}<br>x ${fmt(trio[0][d], 3)} + attn ${fmt(trio[1][d], 3)} = <b>h ${fmt(trio[2][d], 3)}</b>`,
    }),
    h("div", { class: "stats" },
      h("span", null, h("b", null, fmt(norm(trio[0]), 2)), "‖x‖"),
      h("span", null, h("b", null, fmt(norm(trio[1]), 2)), "‖attention output‖"),
      h("span", null, h("b", null, fmt(norm(trio[2]), 2)), "‖h‖")),
    h("p", { class: "sub" },
      "A few dimensions of the attention output are very large. GPT-2 is known to have such \"outlier\" dimensions in its residual stream, and the color scale clips them."),
    h("div", { class: "panel-title", style: { marginTop: "16px" } }, `All tokens: H = X + Attention(X), ${n} × 768, the input to the MLP`),
    heatRows(res.resid, {
      labels: toks.map((t) => tokChip(t)),
      scale: robustScale(res.resid),
      describe: (r, d, v) => `${tokTip(toks[r])} · dim ${d}<br>x ${fmt(res.x[r][d], 3)} + attn ${fmt(res.attnOut[r][d], 3)} = <b>${fmt(v, 3)}</b>`,
    }),
  );
}
