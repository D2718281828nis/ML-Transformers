import { GPT2Tokenizer, PRETOKENIZE_ALTERNATIVES, pairPositions, vocabJsonKey } from "./tokenizer.js";
import { RemoteSafetensors, loadStaticWeights, EmbeddingRows, POSITION_ROWS } from "./weights.js";
import { CFG, forward, headProfile, cosine, sinusoidal, dot } from "./model.js";
import { T } from "./i18n.js";
import { initTraining, renderTraining } from "./training.js";
import { initEmbeddingTraining, renderEmbeddingTraining, renderEmbInit } from "./embedding.js";
import {
  h, s, fmt, pct, int, tokChip, tokText, tokTip, tokPlain, heatRows, heatImage, matrixGrid, cellSize, bars, lineChart,
  attentionLines, legend, seqLegend, robustScale, divergingRGB, sequentialRGB, showTip, hideTip, resetPalette,
  fill, mono, stepper, player,
} from "./viz.js";

const MAX_TOKENS = 48;
const CODE_BASE =
  "https://github.com/D2718281828nis/ML-Transformers/blob/0ade1caa356dce6b70ef8293addeb0898f177206/src/transformers/models/gpt2/";
const ASSETS = new URL("../assets/gpt2", import.meta.url).href;
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
const bpePlayer = player();

const $ = (id) => document.getElementById(id);
const input = $("text");

// ───────────── boot ─────────────
main();

async function main() {
  for (const a of document.querySelectorAll("a.code[data-code]")) a.href = CODE_BASE + a.dataset.code;
  const fromUrl = new URLSearchParams(location.search).get("text");
  if (fromUrl != null) input.value = fromUrl.slice(0, Number(input.maxLength) || 240);
  setupTex();
  setupTopbar();
  setupHoverAndFocus();
  setupScrollSpy();
  renderExamples();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    resetPalette();
    renderAll();
  });

  setStatus("loading", T.loadingTokenizer);
  try {
    tokenizer = await GPT2Tokenizer.load(ASSETS);
  } catch (e) {
    setStatus("error", T.tokenizerError(e.message));
    return;
  }
  let timer = 0;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => update(input.value), 160);
  });
  initTraining(tokenizer);
  initEmbeddingTraining();
  update(input.value);
  loadWeights();
}

async function loadWeights() {
  weightsError = null;
  W = null;
  renderModelSections();
  setStatus("loading", T.downloadingWeights, 0);
  st = new RemoteSafetensors(undefined, {
    onProgress: (loaded, expected) => {
      if (!W) setStatus("loading", T.weightsProgress((loaded / 1e6).toFixed(1), (expected / 1e6).toFixed(1)), expected ? loaded / expected : 0);
    },
  });
  emb = new EmbeddingRows(st);
  try {
    W = await loadStaticWeights(st);
  } catch (e) {
    weightsError = e;
    setStatus("error", T.weightsError(e.message), null, true);
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
  bpePlayer.stop();
  const n = enc.tokens.length;
  $("input-meta").textContent = n ? T.tokens(n) : "";
  for (const b of document.querySelectorAll("#examples button")) b.setAttribute("aria-pressed", String(b.dataset.ex === text));
  const sw = $("lang-switch");
  if (sw) sw.href = `${sw.dataset.base ||= sw.getAttribute("href")}?text=${encodeURIComponent(text)}`;
  setFocusStyle();
  renderTokenization();
  renderTraining(enc);
  state.res = null;
  if (W) compute();
  else renderModelSections();
}

async function compute() {
  const id = ++runId;
  if (!state.tokens.length) {
    state.res = null;
    renderModelSections();
    setStatus("ready", T.ready);
    return;
  }
  const ids = state.tokens.map((t) => t.id);
  if (ids.some((i) => !emb.rows.has(i))) setStatus("loading", T.fetchingEmb, null);
  let rows;
  try {
    rows = await emb.get(ids);
  } catch (e) {
    if (id !== runId) return;
    weightsError = e;
    setStatus("error", T.embError(e.message), null, true);
    renderModelSections();
    return;
  }
  if (id !== runId) return;
  weightsError = null;
  const t0 = performance.now();
  state.res = forward(rows, W);
  state.computeMs = performance.now() - t0;
  setStatus("ready", T.readyStats((st.loaded / 1e6).toFixed(1), state.computeMs.toFixed(0)));
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
  renderTraining(state.enc, true);
  renderEmbeddingTraining();
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
  if (retry) el.append(h("button", { type: "button", onclick: () => (W ? compute() : loadWeights()) }, T.retry));
}

function renderExamples() {
  $("examples").append(...T.examples.map((ex) => h("button", { type: "button", "data-ex": ex, onclick: () => setText(ex) }, ex)));
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
const hex = (b) => b.toString(16).toUpperCase().padStart(2, "0");

function placeholder(box) {
  box.replaceChildren();
  if (!state.tokens.length) {
    box.append(h("div", { class: "placeholder" }, T.phType));
  } else if (weightsError) {
    box.append(
      h("div", { class: "placeholder error" },
        T.phError(weightsError.message),
        h("button", { class: "btn", type: "button", onclick: () => (W ? compute() : loadWeights()) }, T.retry)),
    );
  } else {
    box.append(h("div", { class: "placeholder" }, h("span", { class: "spinner" }), T.phLoading));
  }
}

// Header row of a viz box: title (text and/or nodes) on the left, controls/legend on the right.
function vizHead(title, right = []) {
  return h("div", { class: "viz-head" }, h("div", { class: "panel-title", style: { margin: 0 } }, title), h("div", { class: "viz-head-right" }, right));
}

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

function selectChunk(i) {
  state.bpeChunk = i;
  state.bpeStep = 0;
  bpePlayer.stop();
  renderPretok();
  renderBPE();
}

const chunkButton = (c, i, sub) =>
  h("button", { class: "chunk", type: "button", "aria-pressed": String(i === state.bpeChunk), onclick: () => selectChunk(i) },
    h("span", { class: "chunk-text" }, tokText(c.text)),
    h("span", { class: "chunk-kind" }, sub));

function renderPretok() {
  const box = $("viz-pretok");
  box.replaceChildren();
  const { chunks } = state.enc;
  if (!chunks.length) return box.append(h("div", { class: "placeholder" }, T.phStart));
  box.append(
    h("div", { class: "panel-title" }, T.chunksTitle(chunks.length)),
    h("div", { class: "chunks" }, chunks.map((c, i) => chunkButton(c, i, `#${c.alt} ${T.altShort[c.alt]}`))),
    altTable(chunks),
  );
}

// The 12 alternatives of the pre-tokenization regex, with the chunks of this text each one caught.
function altTable(chunks) {
  const byAlt = new Map();
  for (const c of chunks) byAlt.set(c.alt, [...(byAlt.get(c.alt) || []), c]);
  return h("div", { style: { marginTop: "18px" } },
    h("div", { class: "panel-title" }, T.altTableTitle),
    h("div", { class: "scroll-x" },
      h("table", { class: "alt-table" },
        h("thead", null, h("tr", null, T.altCols.map((c) => h("th", null, c)))),
        h("tbody", null, T.altRows.map(({ alts: [a0, a1], desc, ex }) => {
          const idx = Array.from({ length: a1 - a0 + 1 }, (_, i) => a0 + i);
          const used = idx.flatMap((k) => byAlt.get(k) || []);
          return h("tr", { class: used.length ? "used" : null },
            h("td", { class: "num" }, a0 === a1 ? a0 : `${a0}–${a1}`),
            // A leading literal space is shown as ␣ so it is visible.
            h("td", null, h("code", null, idx.map((k) => PRETOKENIZE_ALTERNATIVES[k - 1].replace(/^ /, "␣")).join("  "))),
            h("td", null, desc),
            h("td", { class: "ex" }, ex),
            h("td", null, used.length
              ? h("span", { class: "chunks mini" }, used.map((c) => h("span", { class: "chunk-text" }, tokText(c.text))))
              : h("span", { class: "muted" }, T.altUnused)));
        })))),
  );
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

  box.append(
    h("div", { class: "panel-title" }, T.chunk),
    h("div", { class: "chunks", style: { marginBottom: "14px" } }, chunks.map((c, i) => chunkButton(c, i, T.merges(c.steps.length)))),
  );

  // characters → bytes → byte symbols
  const groups = Array.from(chunk.text).map((ch) =>
    h("div", { class: "byte-group", style: { flexDirection: "column", alignItems: "center" } },
      h("span", { class: "byte-col" }, h("span", { class: "ch" }, ch === " " ? "␣" : ch === "\n" ? "↵" : ch === "\t" ? "⇥" : ch)),
      h("div", { style: { display: "flex", gap: "3px" } },
        [...utf8.encode(ch)].map((b) =>
          h("div", { class: "byte-col" },
            h("span", { class: "hex" }, hex(b)),
            h("span", { class: "arrow-d" }, "↓"),
            h("span", { class: "bu" }, tokenizer.byteEncoder.get(b)))))));
  box.append(h("div", { class: "panel-title" }, T.bytesTitle), h("div", { class: "bpe-bytes" }, groups));

  const go = (v) => {
    bpePlayer.stop();
    state.bpeStep = Math.max(0, Math.min(K, v));
    renderBPE();
  };
  box.append(
    h("div", { class: "panel-title" }, T.mergeStep(k, K)),
    stepper({ k, K, go, playing: bpePlayer.playing, onPlay: () => bpePlayer.toggle(() => state.bpeStep, (v) => (state.bpeStep = v), K, renderBPE) }),
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
          title: `${T.bytesTip}: ${tokenizer.tokenBytes(sym).map(hex).join(" ")}`,
        }, sym);
      })),
  );

  const msg = h("div", { class: "bpe-msg" });
  if (next) {
    const where = pairPositions(word, next.pair).length;
    msg.append(...T.nextMerge(h("b", null, `#${int(next.rank + 1)}`), mono(next.pair[0]), mono(next.pair[1]), mono(next.pair.join("")), where > 1 ? T.inPlaces(where) : ""));
  } else if (word.length === 1 && K === 0) {
    msg.append(T.singleByte);
  } else {
    msg.append(
      K ? T.doneMerging : T.nothingMerges,
      T.becomes(chunk.tokens.length),
      h("span", { class: "tok-row", style: { display: "inline-flex", verticalAlign: "middle" } }, chunk.tokens.map((t) => tokChip(t, { showId: true }))),
    );
  }
  box.append(msg);

  if (word.length > 1) {
    const cands = [];
    for (let i = 0; i < word.length - 1; i++) cands.push({ pair: `${word[i]} ${word[i + 1]}`, r: tokenizer.ranks.get(word[i] + " " + word[i + 1]) });
    const minR = Math.min(...cands.map((c) => c.r ?? Infinity));
    box.append(
      h("div", { class: "panel-title", style: { marginTop: "12px" } }, T.pairsTitle),
      h("ul", { class: "merge-list" },
        cands.map((c) =>
          h("li", null, h("button", { type: "button", tabindex: -1, "aria-current": String(c.r === minR && c.r !== undefined), class: c.r === undefined ? "done" : null },
            `${c.pair}  ${c.r === undefined ? T.notInList : "#" + int(c.r + 1)}`)))),
    );
  }

  if (K) {
    box.append(
      h("div", { class: "panel-title", style: { marginTop: "12px" } }, T.historyTitle),
      h("ol", { class: "merge-list" },
        chunk.steps.map((stp, i) =>
          h("li", null, h("button", { type: "button", class: i < k ? "done" : null, "aria-current": String(i === k - 1), onclick: () => go(i + 1) },
            `${i + 1}. ${stp.pair[0]} + ${stp.pair[1]}`)))),
    );
  }
}

// ───────────── 1.3 token IDs ─────────────
function idOrigin(tok) {
  if (tok.id === tokenizer.vocabSize - 1) return T.originSpecial;
  if (tok.id < 256) return T.originByte(hex(tokenizer.byteDecoder.get(tok.str)));
  const [a, b] = tokenizer.merges[tok.id - 256].split(" ");
  return T.originMerge(tok.id - 255, mono(a), mono(b));
}

function renderIds() {
  const box = $("viz-ids");
  box.replaceChildren();
  const { enc } = state;
  const toks = state.tokens;
  if (!toks.length) return;
  const chars = Array.from(enc.text).length;
  const bytes = utf8.encode(enc.text).length;
  const stat = (n, label) => h("span", null, h("b", null, n), label);
  const uniq = [...new Map(toks.map((t) => [t.id, t])).values()];
  box.append(
    h("div", { class: "tok-row" }, toks.map((t) => tokChip(t, { showId: true }))),
    h("div", { class: "stats" },
      stat(chars, T.stChars(chars)),
      stat(bytes, T.stBytes(bytes)),
      stat(enc.chunks.length, T.stChunks(enc.chunks.length)),
      stat(enc.tokens.length, T.stTokens(enc.tokens.length)),
      stat((chars / enc.tokens.length).toFixed(1), T.stPerToken)),
    h("div", { class: "ids-array" }, h("span", { class: "k" }, T.modelInput), `[${toks.map((t) => t.id).join(", ")}]`),
    h("div", { class: "panel-title", style: { marginTop: "18px" } }, T.vocabTitle),
    h("div", { class: "scroll-x" },
      h("table", { class: "sm-table vocab-table" },
        h("thead", null, h("tr", null, T.vocabCols.map((c) => h("th", null, c)))),
        h("tbody", null, uniq.map((t) =>
          h("tr", null,
            h("td", null, tokChip(t)),
            h("td", null, mono(`${vocabJsonKey(t.str)}: ${t.id}`)),
            h("td", null, h("b", null, t.id)),
            h("td", { class: "origin" }, idOrigin(t))))))),
    h("div", { class: "panel-title", style: { marginTop: "16px" } }, T.rangesTitle),
    h("div", { class: "id-ranges" }, T.ranges.map(([r, d], i) => h("div", { class: `rng rng${i}` }, h("b", { class: "mono" }, r), h("span", null, d)))),
  );
  if (toks.some((t) => t.partial)) box.append(h("div", { class: "callout" }, T.partialNote));
  if (state.truncated) box.append(h("div", { class: "callout warn" }, T.truncated(enc.tokens.length, MAX_TOKENS)));
}

// ═════════════ 2. Token embedding ═════════════
function renderEmbedding() {
  const wte = $("viz-wte");
  const sim = $("viz-sim");
  const { res } = state;
  renderEmbInit({ res, tokens: state.tokens, focus: state.focus, placeholder });
  if (!res) return placeholder(wte), placeholder(sim);
  const toks = state.tokens;
  const n = toks.length;
  const scale = robustScale(res.tokEmb);

  fill(wte,
    vizHead(T.wteTitle(n), legend(scale)),
    h("div", { class: "emb-layout" }, matrixSchematic(toks), h("div", null,
      heatRows(res.tokEmb, {
        labels: toks.map((t) => labelFor(t, `#${t.id}`)),
        scale,
        describe: (r, d, v) => T.wteTip(tokTip(toks[r]), int(toks[r].id), d, fmt(v, 4)),
      }),
      focusValues("e", res.tokEmb[state.focus], toks[state.focus], `W_E[${toks[state.focus].id}]`),
      h("p", { class: "sub" }, T.wteSub))),
  );

  const size = cellSize(n, 520);
  let best = null;
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) {
    if (toks[i].id === toks[j].id) continue;
    const c = cosine(res.tokEmb[i], res.tokEmb[j]);
    if (!best || c > best.c) best = { i, j, c };
  }
  fill(sim,
    vizHead(T.simTitle, legend(1, { digits: 0 })),
    matrixGrid({
      n, size,
      rowLabels: toks.map((t) => tokChip(t)),
      colLabels: toks.map((t) => tokChip(t)),
      cell: (i, j) => {
        const c = cosine(res.tokEmb[i], res.tokEmb[j]);
        return {
          bg: divergingRGB(c, 1),
          text: fmt(c, 2),
          tip: `cos(${tokTip(toks[i])}, ${tokTip(toks[j])}) = <b>${fmt(c, 3)}</b>${toks[i].id === toks[j].id ? `<br>${T.sameToken}` : ""}`,
        };
      },
    }),
    best ? h("p", { class: "sub" }, T.mostSimilar(tokChip(toks[best.i]), tokChip(toks[best.j]), fmt(best.c, 2))) : null,
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
  const svg = s("svg", { class: "matrix-svg", viewBox: `0 0 ${W0} ${H}`, role: "img", "aria-label": T.matAria });
  svg.append(
    s("text", { class: "caption", x: matX, y: 12 }, T.matCaption),
    s("text", { class: "dimtext", x: matX, y: 27 }, T.matCols),
    s("rect", { class: "mat", x: matX, y: top, width: matW, height: matH, rx: 3 }),
    s("text", { class: "dimtext", x: matX + matW + 4, y: top + 8 }, T.matRow0),
    s("text", { class: "dimtext", x: matX + matW + 4, y: top + matH }, T.matRowLast),
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
  if (prev > top + matH - 12) for (let i = lab.length - 1, lim = top + matH - 12; i >= 0; i--, lim -= 16) lab[i] = Math.min(lab[i], lim);
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
  return h("div", { class: "vals" }, tokChip(tok), " ", h("span", { class: "k" }, `${sym} = ${expr} = `), `[${first}, ${T.more(vec.length - 8)}]`);
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
  posCache[kind] = {
    rows, top,
    sim: rows.map((a) => rows.map((b) => cosine(a, b))),
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
  const P = positionTable(state.posView);
  const used = Math.min(n, POSITION_ROWS);
  const viewBtn = (key, label) =>
    h("button", { type: "button", "aria-pressed": String(state.posView === key), onclick: () => ((state.posView = key), renderPosition()) }, label);

  const simCanvas = heatImage((r) => P.sim[r], POSITION_ROWS, POSITION_ROWS, {
    scale: 1,
    rowPx: 2,
    describe: (r, c, v) => T.posSimTip(r, c, fmt(v, 3)),
  });
  simCanvas.style.maxWidth = "256px";

  fill(wpe,
    vizHead(learned ? T.wpeTitle(POSITION_ROWS) : T.sinTitle(POSITION_ROWS),
      [h("span", { class: "seg-toggle" }, viewBtn("learned", T.learned), viewBtn("sin", T.sinusoidal)), legend(P.scale)]),
    h("div", { class: "pos-image" },
      h("div", { class: "pos-axis" }, h("span", null, T.pos(0)), h("span", null, T.pos(POSITION_ROWS - 1))),
      heatImage((r) => P.rows[r], POSITION_ROWS, D, {
        scale: P.scale,
        rowPx: 2,
        bracket: [0, used],
        describe: (r, d, v) => T.wpeTip(r, d, fmt(v, 4), r < n ? tokTip(toks[r]) : null),
      }),
      h("span"), h("div", { class: "heat-axis" }, h("span", null, T.dim(0)), h("span", null, T.dim(D - 1)))),
    h("p", { class: "sub" }, learned ? T.wpeSub(used, P.smoothDims) : T.sinSub),
    h("div", { class: "viz-row", style: { marginTop: "16px" } },
      h("div", { class: "grow" },
        h("div", { class: "panel-title" }, learned ? T.topDims : T.someWaves),
        lineChart(P.top.map((d) => ({ label: T.dim(d), values: P.rows.map((r) => r[d]) })), { mark: used }),
        h("p", { class: "sub" }, learned ? T.curvesLearned : T.curvesSin)),
      h("div", { style: { flex: "0 1 280px" } },
        h("div", { class: "panel-title" }, T.posSimTitle),
        simCanvas,
        h("p", { class: "sub" }, T.posSimSub, learned ? T.row0(P.row0Ratio.toFixed(1)) : ""))),
    learned
      ? h("div", { style: { marginTop: "14px" } },
          h("div", { class: "panel-title" }, T.rowsUsed),
          heatRows(res.posEmb, {
            labels: toks.map((t, r) => labelFor(t, T.pos(r))),
            scale: robustScale(res.posEmb, 0.95),
            describe: (r, d, v) => T.posRowTip(r, tokTip(toks[r]), d, fmt(v, 4)),
          }))
      : null,
  );

  // 3.2 sum, for the selected token
  const trio = [res.tokEmb[i], res.posEmb[i], res.x[i]];
  const sc = robustScale(trio);
  const dup = findDuplicate(toks);
  const [lt, lp, lx] = T.sumLabels;
  fill(sum,
    vizHead([T.selectedToken, tokChip(toks[i]), T.atPosition(i)], legend(sc)),
    heatRows(trio, {
      labels: [h("span", null, lt, h("sub", null, "i")), h("span", null, lp, h("sub", null, "i")), h("b", null, lx, h("sub", null, "i"))],
      scale: sc,
      rowHeight: 22,
      describe: (_r, d) => `${tokTip(toks[i])} · ${T.dim(d)}<br>e ${fmt(trio[0][d], 4)} + p ${fmt(trio[1][d], 4)} = <b>x ${fmt(trio[2][d], 4)}</b>`,
    }),
    h("div", { class: "panel-title", style: { marginTop: "18px" } }, T.allX(n)),
    heatRows(res.x, {
      labels: toks.map((t, r) => labelFor(t, T.pos(r))),
      scale: robustScale(res.x),
      describe: (r, d, v) => `${tokTip(toks[r])} · ${T.dim(d)}<br>e ${fmt(res.tokEmb[r][d], 4)} + p ${fmt(res.posEmb[r][d], 4)} = <b>${fmt(v, 4)}</b>`,
    }),
    dup
      ? h("div", { class: "callout" }, T.dup(tokChip(toks[dup[0]]), dup[0], dup[1], h("b", null, fmt(cosine(res.x[dup[0]], res.x[dup[1]]), 3))))
      : h("div", { class: "callout" },
          T.tryRepeat(h("span", { class: "example-link", role: "button", tabindex: 0, onclick: () => setText(T.repeatExample) }, T.repeatExample))),
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
  if (!res) {
    $("attn-controls").replaceChildren();
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
      h("span", { class: "lbl" }, T.head),
      h("div", { class: "head-pick" },
        profiles.map((p, hd) =>
          h("button", { type: "button", "aria-pressed": String(hd === state.head), title: `${T.headN(hd)}: ${T.headDesc[p.label]}`, onclick: () => setHead(hd) },
            hd, h("small", null, T.headTag[p.label]))))),
    h("div", { class: "ctl tokens" },
      h("span", { class: "lbl" }, T.query),
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
    T.lnTip(tokTip(t), d, fmt(x[d], 3), fmt(L.mean, 3), fmt(L.std, 3), fmt(L.norm[d], 3), fmt(W.lnW[d], 3), fmt(W.lnB[d], 3), fmt(L.out[d], 3));
  const one = (vec, label) => {
    const sc = robustScale([vec]);
    return h("div", { style: { marginBottom: "10px" } },
      h("div", { class: "viz-head", style: { marginBottom: "4px" } }, h("span", { style: { fontSize: "14px" } }, label), legend(sc)),
      heatRows([vec], { scale: sc, rowHeight: 20, describe, axis: false }));
  };
  const range = (v) => `${T.range} ${fmt(Math.min(...v), 2)} … ${fmt(Math.max(...v), 2)}`;
  fill($("viz-ln"),
    vizHead([T.selectedToken, tokChip(t)]),
    one(x, h("span", null, "x", h("sub", null, "i"), ` · ${range(x)}`)),
    one(L.norm, h("span", null, "(x", h("sub", null, "i"), " − μ) / σ", ` · ${T.meanStd}`)),
    one(L.out, h("b", null, "x̂", h("sub", null, "i"), ` = γ ⊙ … + β · ${range(L.out)}`)),
    h("div", { class: "stats" },
      h("span", null, h("b", null, fmt(L.mean, 4)), T.mean),
      h("span", null, h("b", null, fmt(L.std, 4)), T.std),
      h("span", null, h("b", null, "768"), T.lnDims)),
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
        describe: (r, d, v) => T.qkvTip(key.toLowerCase(), tokTip(toks[r]), d, Math.floor(d / HD), d % HD, fmt(v, 3)),
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
      shape("", T.heads12, T.eachShape(n))),
    h("p", { class: "sub", style: { marginTop: 0 } }, T.qkvSub(hd, seg[0], seg[1] - 1)),
    group("Q", T.queries, "q", res.q),
    group("K", T.keys, "k", res.k),
    group("V", T.values, "v", res.v),
  );
}

function scoreGrid(n, size, cell) {
  const toks = state.tokens;
  return matrixGrid({
    n, size,
    corner: h("span", null, T.corner[0], h("br"), T.corner[1]),
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
      tip: T.scoreTip(tokTip(toks[i]), tokTip(toks[j]), fmt(v, 3), j > i),
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
    vizHead(T.scoresTitle(state.head, n), legend(sc, { digits: 1 })),
    h("div", { class: "viz-row" }, h("div", null, grid, h("p", { class: "axis-note" }, T.fadedNote)), inspector),
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
  const top = [...prod.keys()].sort((a, b) => Math.abs(prod[b]) - Math.abs(prod[a])).slice(0, 3);
  const tip = (d) => T.prodTip(d, fmt(q[d], 3), fmt(k[d], 3), fmt(prod[d], 3));
  fill(box,
    h("div", { class: "panel-title" }, T.inspector),
    h("div", { class: "eq" }, T.qOf, tokChip(toks[i]), T.kOf, tokChip(toks[j]), " = ", h("b", null, fmt(sum, 3))),
    h("div", { class: "bars-block" }, h("div", { class: "lbl" }, h("span", { class: "q-ink" }, T.qBars(i, state.head))), bars(q, { scale: qkScale, describe: tip })),
    h("div", { class: "bars-block" }, h("div", { class: "lbl" }, h("span", { class: "k-ink" }, T.kBars(j, state.head))), bars(k, { scale: qkScale, describe: tip })),
    h("div", { class: "bars-block" },
      h("div", { class: "lbl" }, h("span", null, `${T.products} q`, h("sub", null, "d"), " × k", h("sub", null, "d")), h("span", null, T.sumIs(fmt(sum, 3)))),
      bars(prod, { scale: robustScale([prod], 1), describe: tip })),
    h("div", { class: "contrib" }, T.largest, top.map((d) => `${T.dim(d)} (${fmt(prod[d], 2)})`).join(", "), T.largestTail(fmt(sum, 2), fmt(sum / 8, 3))),
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
    if (state.mask && j > i) return { cls: "masked", text: "−∞", tip: T.maskedTip(tokTip(toks[i]), tokTip(toks[j])) };
    const v = hd.scaled[i][j];
    return { bg: divergingRGB(v, sc), text: fmt(v, 2), cls: i === state.focus ? "rowsel" : "", tip: `${fmt(hd.scores[i][j], 3)} ÷ 8 = <b>${fmt(v, 3)}</b>` };
  });
  // Why scale: compare the top weight in the selected row with and without ÷√64.
  const i = state.focus;
  const raw = hd.scores[i];
  const upto = state.mask ? i : n - 1;
  const softTop = (row, div) => {
    let m = -Infinity;
    for (let j = 0; j <= upto; j++) m = Math.max(m, row[j] / div);
    let total = 0;
    for (let j = 0; j <= upto; j++) total += Math.exp(row[j] / div - m);
    return 1 / total;
  };
  const toggle = h("label", { class: "toggle" }, h("input", { type: "checkbox", checked: state.mask }), T.causalMask);
  toggle.querySelector("input").addEventListener("change", (e) => {
    state.mask = e.target.checked;
    renderMask();
    renderSoftmax();
    renderWeighted();
  });
  fill($("viz-mask"),
    vizHead(T.maskTitle(state.head), [toggle, legend(sc, { digits: 1 })]),
    grid,
    !state.mask ? h("div", { class: "callout warn" }, T.maskOffWarn) : null,
    n > 1 ? h("div", { class: "callout" }, T.whyScale(tokChip(toks[i]), h("b", null, pct(softTop(raw, 8))), h("b", null, pct(softTop(raw, 1))))) : null,
  );
}

const probsOf = (hd) => (state.mask ? hd.probs : hd.probsNoMask);
const maskOffTag = () => (state.mask ? null : h("span", { class: "maskoff" }, T.maskOffTag));

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
    if (state.mask && j > r) return { cls: "masked", text: "0", tip: T.maskedWeight };
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
    h("thead", null, h("tr", null, T.smCols.map((c) => h("th", null, c)))),
    h("tbody", null, ex.map((e, j) =>
      h("tr", null,
        h("td", null, tokChip(toks[j])),
        h("td", null, fmt(hd.scaled[i][j], 3)),
        h("td", null, fmtE(e)),
        h("td", null, pct(e / total, 1)),
        h("td", { class: "barcell" }, h("div", { class: "wbar", style: { width: `${(e / total) * 100}%` } }))))),
    h("tfoot", null, h("tr", null, h("td", null, T.sum), h("td", null, ""), h("td", null, fmtE(total)), h("td", null, "100%"), h("td", null, ""))),
  );
  fill($("viz-softmax"),
    vizHead([T.softmaxTitle(state.head), maskOffTag()], seqLegend()),
    h("div", { class: "viz-row" },
      h("div", null, grid, h("p", { class: "axis-note" }, T.rowsSum)),
      h("div", { class: "grow", style: { maxWidth: "440px" } },
        h("div", { class: "panel-title" }, T.whoListens),
        attentionLines(toks, P, i, { onFocus: setFocus }))),
    h("div", { style: { marginTop: "16px" } },
      h("div", { class: "panel-title" }, T.softmaxRow, tokChip(toks[i])),
      h("div", { class: "scroll-x" }, table),
      h("p", { class: "sub" }, T.softmaxSub)),
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
      return T.wTip(pct(P[i][j], 1), tokTip(toks[j]), d, fmt(P[i][j], 3), fmt(hd.v[j][d], 3), fmt(weighted[r][d], 3));
    }
    const terms = js.map((j, r2) => ({ j, v: weighted[r2][d] })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 4);
    return T.zTip(d, fmt(z[d], 3), terms.map((t) => `${tokTip(toks[t.j])} ${fmt(t.v, 3)}`).join(", "));
  };
  fill($("viz-weighted"),
    vizHead([T.outputFor(state.head), tokChip(toks[i]), maskOffTag()], legend(sc)),
    heatRows([...weighted, z], {
      labels: [
        ...js.map((j) => h("span", { class: "tok-label" }, h("span", { class: "mono" }, pct(P[i][j], 1)), "×", tokChip(toks[j]))),
        h("b", null, "= z", h("sub", null, i), " (64)"),
      ],
      scale: sc,
      rowHeight: js.length > 16 ? 10 : 16,
      describe,
    }),
    h("p", { class: "sub" }, T.weightedSub),
  );
}

function renderHeads(profiles) {
  const { res } = state;
  const toks = state.tokens;
  const n = toks.length;
  const i = state.focus;
  const maskedBg = getComputedStyle(document.documentElement).getPropertyValue("--surface-2").trim();
  const cards = res.heads.map((hd, k) => {
    const cv = h("canvas", { width: n, height: n });
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(n, n);
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < n; j++) {
        const c = j > r ? null : sequentialRGB(hd.probs[r][j]);
        img.data.set(c ? [c[0], c[1], c[2], 255] : [0, 0, 0, 0], (r * n + j) * 4);
      }
    }
    ctx.putImageData(img, 0, 0);
    cv.style.background = maskedBg;
    const p = profiles[k];
    const tip = T.headTip(k, T.headDesc[p.label], { self: pct(p.self, 0), prev: pct(p.prev, 0), first: pct(p.first, 0), other: pct(p.other, 0) });
    const card = h("button", { class: "head-card", type: "button", "aria-pressed": String(k === state.head), onclick: () => setHead(k) },
      h("div", { class: "ttl" }, T.headN(k), h("span", null, T.headTag[p.label])),
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
      h("span", null, h("i", { class: "p-self" }), T.legendSelf),
      h("span", null, h("i", { class: "p-prev" }), T.legendPrev),
      h("span", null, h("i", { class: "p-first" }), T.legendFirst),
      h("span", null, h("i", { class: "p-other" }), T.legendOther),
      h("span", { style: { color: "var(--muted)" } }, T.legendNote)),
    h("div", { class: "heads-grid" }, cards),
    h("div", { style: { marginTop: "18px" } },
      vizHead([T.concatFor, tokChip(toks[i])], legend(robustScale([z]))),
      heatRows([z], {
        labels: [h("span", null, "z", h("sub", null, i), " (768)")],
        scale: robustScale([z]),
        rowHeight: 22,
        boundaries: HD,
        segment: [hdSel * HD, (hdSel + 1) * HD],
        describe: (_r, d, v) => T.concatTip(Math.floor(d / HD), d % HD, fmt(v, 3)),
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
  const [lx, la, lh] = T.projLabels;
  fill($("viz-proj"),
    vizHead([T.selectedToken, tokChip(toks[i])], legend(sc)),
    heatRows(trio, {
      labels: [
        h("span", null, lx, h("sub", null, i)),
        h("span", null, la, h("sub", null, i), "W", h("sub", null, "O"), " + b"),
        h("b", null, lh, h("sub", null, i)),
      ],
      scale: sc,
      rowHeight: 22,
      describe: (_r, d) => T.projTip(d, fmt(trio[0][d], 3), fmt(trio[1][d], 3), fmt(trio[2][d], 3)),
    }),
    h("div", { class: "stats" },
      h("span", null, h("b", null, fmt(norm(trio[0]), 2)), "‖x‖"),
      h("span", null, h("b", null, fmt(norm(trio[1]), 2)), T.attnNorm),
      h("span", null, h("b", null, fmt(norm(trio[2]), 2)), "‖h‖")),
    h("p", { class: "sub" }, T.outlierNote),
    h("div", { class: "panel-title", style: { marginTop: "16px" } }, T.allH(n)),
    heatRows(res.resid, {
      labels: toks.map((t) => tokChip(t)),
      scale: robustScale(res.resid),
      describe: (r, d, v) => T.residTip(tokTip(toks[r]), d, fmt(res.x[r][d], 3), fmt(res.attnOut[r][d], 3), fmt(v, 3)),
    }),
  );
}
