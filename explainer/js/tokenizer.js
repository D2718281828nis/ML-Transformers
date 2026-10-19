// GPT-2 byte-level BPE tokenizer.
// A line-by-line port of src/transformers/models/gpt2/tokenization_gpt2.py
// (bytes_to_unicode, the pre-tokenization regex and GPT2Tokenizer.bpe) that also
// records every merge step so the page can replay it.

// Same pattern as GPT2Tokenizer.pat, split into its 12 alternatives. Each one is wrapped
// in a capture group so we can tell which alternative produced a chunk; groups don't
// change what matches. JS needs the `u` flag for \p{...}.
export const PRETOKENIZE_ALTERNATIVES = [
  "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
  " ?\\p{L}+", " ?\\p{N}+", " ?[^\\s\\p{L}\\p{N}]+", "\\s+(?!\\S)", "\\s+",
];
export const PRETOKENIZE_PATTERN = PRETOKENIZE_ALTERNATIVES.join("|");
const PRETOKENIZE_GROUPED = PRETOKENIZE_ALTERNATIVES.map((a) => `(${a})`).join("|");

// GPT-2 maps every byte 0..255 to a printable unicode character so the vocabulary
// can be stored as text. Printable bytes map to themselves; the rest (space,
// control bytes, ...) are shifted to 256+n. That is why a space shows up as "Ġ".
export function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = new Map();
  bs.forEach((b, i) => map.set(b, String.fromCodePoint(cs[i])));
  return map;
}

// Merge every non-overlapping occurrence of the pair (a, b), left to right.
// `positions` are the indices in the result where a merge happened.
export function mergeAll(word, a, b) {
  const merged = [];
  const positions = [];
  for (let i = 0; i < word.length; i++) {
    if (i < word.length - 1 && word[i] === a && word[i + 1] === b) {
      positions.push(merged.length);
      merged.push(a + b);
      i++;
    } else {
      merged.push(word[i]);
    }
  }
  return { merged, positions };
}

// Indices where the pair (a, b) would be merged.
export function pairPositions(word, [a, b]) {
  const pos = [];
  for (let i = 0; i < word.length - 1; i++) {
    if (word[i] === a && word[i + 1] === b) {
      pos.push(i);
      i++;
    }
  }
  return pos;
}

// Apply BPE rules to a list of symbols. rankOf("a b") gives the rule number of a pair
// (undefined if there is no such rule). At each step the neighbouring pair with the
// lowest rule number (= learned earliest) is merged everywhere, until no pair has a rule.
// Returns the final pieces and every step, so the page can replay them.
export function bpeMerge(symbols, rankOf) {
  let word = symbols;
  const steps = [];
  while (word.length > 1) {
    let bestRank = Infinity;
    let best = -1;
    for (let i = 0; i < word.length - 1; i++) {
      const r = rankOf(word[i] + " " + word[i + 1]);
      if (r !== undefined && r < bestRank) {
        bestRank = r;
        best = i;
      }
    }
    if (best < 0) break;
    const pair = [word[best], word[best + 1]];
    const { merged, positions } = mergeAll(word, ...pair);
    steps.push({ before: word, pair, rank: bestRank, after: merged, positions });
    word = merged;
  }
  return { pieces: word, steps };
}

// A key as it appears in vocab.json, which Python's json.dump wrote with non-ASCII
// characters escaped as \uXXXX (so "Ġneed" is stored as "\u0120need").
export function vocabJsonKey(str) {
  const body = Array.from(str, (ch) => {
    const c = ch.codePointAt(0);
    if (ch === '"' || ch === "\\") return "\\" + ch;
    return c < 0x20 || c > 0x7e ? `\\u${c.toString(16).padStart(4, "0")}` : ch;
  }).join("");
  return `"${body}"`;
}

const utf8 = new TextEncoder();
const strictDecoder = new TextDecoder("utf-8", { fatal: true });

export class GPT2Tokenizer {
  constructor(vocab, mergesText) {
    this.encoder = vocab; // token string -> id
    this.decoder = new Map(Object.entries(vocab).map(([k, v]) => [v, k]));
    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = new Map([...this.byteEncoder].map(([b, c]) => [c, b]));
    this.ranks = new Map(); // "a b" -> rank (0 = first rule in merges.txt)
    this.merges = []; // rank -> "a b"
    for (const line of mergesText.split("\n")) {
      if (!line || line.startsWith("#version")) continue;
      this.ranks.set(line, this.merges.length);
      this.merges.push(line);
    }
    this.vocabSize = Object.keys(vocab).length;
    this.numMerges = this.merges.length;
    this.cache = new Map();
  }

  static async load(baseUrl) {
    const [vocab, merges] = await Promise.all([
      fetch(`${baseUrl}/vocab.json`).then((r) => {
        if (!r.ok) throw new Error(`vocab.json: HTTP ${r.status}`);
        return r.json();
      }),
      fetch(`${baseUrl}/merges.txt`).then((r) => {
        if (!r.ok) throw new Error(`merges.txt: HTTP ${r.status}`);
        return r.text();
      }),
    ]);
    return new GPT2Tokenizer(vocab, merges);
  }

  // Byte-level string for a chunk of text: UTF-8 bytes -> printable characters.
  toByteString(text) {
    return Array.from(utf8.encode(text), (b) => this.byteEncoder.get(b)).join("");
  }

  // Bytes behind a byte-level token string.
  tokenBytes(tokenStr) {
    return Array.from(tokenStr, (ch) => this.byteDecoder.get(ch));
  }

  // Human-readable form of one token. Tokens that hold only part of a multi-byte
  // UTF-8 character cannot be decoded on their own, so they are shown as hex.
  displayToken(tokenStr) {
    const bytes = this.tokenBytes(tokenStr);
    try {
      return { text: strictDecoder.decode(new Uint8Array(bytes)), partial: false, bytes };
    } catch {
      // Thin space between bytes: a normal space would be drawn as the "␣" token marker.
      const hex = bytes.map((b) => b.toString(16).toUpperCase().padStart(2, "0")).join(" ");
      return { text: `‹${hex}›`, partial: true, bytes };
    }
  }

  // BPE on one byte-level chunk with GPT-2's 50,000 rules (cached).
  bpe(chunk) {
    const cached = this.cache.get(chunk);
    if (cached) return cached;
    const result = bpeMerge(Array.from(chunk), (pair) => this.ranks.get(pair));
    this.cache.set(chunk, result);
    return result;
  }

  // The same with only the first `n` rules, i.e. the tokenizer you would get by
  // stopping training after n rounds.
  bpeFirst(chunk, n) {
    return bpeMerge(Array.from(chunk), (pair) => {
      const r = this.ranks.get(pair);
      return r !== undefined && r < n ? r : undefined;
    }).pieces;
  }

  // Split text into chunks; `alt` is the 1-based index of the alternative that matched.
  pretokenize(text) {
    const re = new RegExp(PRETOKENIZE_GROUPED, "gu");
    return Array.from(text.matchAll(re), (m) => ({
      text: m[0],
      start: m.index,
      alt: m.findIndex((g, i) => i > 0 && g !== undefined),
    }));
  }

  // Full encode with every intermediate kept for the visualisation.
  encode(text) {
    const chunks = [];
    const tokens = [];
    for (const { text: piece, start, alt } of this.pretokenize(text)) {
      const byteStr = this.toByteString(piece);
      const { pieces, steps } = this.bpe(byteStr);
      const chunk = { index: chunks.length, text: piece, start, alt, byteStr, steps, tokens: [] };
      for (const piece of pieces) {
        const id = this.encoder[piece];
        if (id === undefined) throw new Error(`BPE produced a piece missing from the vocabulary: ${piece}`);
        const tok = { index: tokens.length, str: piece, id, chunk: chunk.index, ...this.displayToken(piece) };
        chunk.tokens.push(tok);
        tokens.push(tok);
      }
      chunks.push(chunk);
    }
    return { text, chunks, tokens, ids: tokens.map((t) => t.id) };
  }
}
