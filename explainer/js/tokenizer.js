// GPT-2 byte-level BPE tokenizer.
// A line-by-line port of src/transformers/models/gpt2/tokenization_gpt2.py
// (bytes_to_unicode, the pre-tokenization regex and GPT2Tokenizer.bpe) that also
// records every merge step so the page can replay it.

// Same pattern as GPT2Tokenizer.pat. JS needs the `u` flag for \p{...}.
export const PRETOKENIZE_PATTERN =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

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

const utf8 = new TextEncoder();
const strictDecoder = new TextDecoder("utf-8", { fatal: true });

export class GPT2Tokenizer {
  constructor(vocab, mergesText) {
    this.encoder = vocab; // token string -> id
    this.decoder = new Map(Object.entries(vocab).map(([k, v]) => [v, k]));
    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = new Map([...this.byteEncoder].map(([b, c]) => [c, b]));
    this.ranks = new Map();
    let rank = 0;
    for (const line of mergesText.split("\n")) {
      if (!line || line.startsWith("#version")) continue;
      this.ranks.set(line, rank++);
    }
    this.vocabSize = Object.keys(vocab).length;
    this.numMerges = rank;
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

  // BPE on one byte-level chunk. Returns the final pieces and every merge step:
  // at each step the adjacent pair with the lowest rank (= learned earliest) is
  // merged everywhere in the word, until no adjacent pair is in merges.txt.
  bpe(chunk) {
    const cached = this.cache.get(chunk);
    if (cached) return cached;
    let word = Array.from(chunk);
    const steps = [];
    while (word.length > 1) {
      let bestRank = Infinity;
      let best = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const first = word[best];
      const second = word[best + 1];
      const merged = [];
      const positions = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          positions.push(merged.length);
          merged.push(first + second);
          i++;
        } else {
          merged.push(word[i]);
        }
      }
      steps.push({ before: word, pair: [first, second], rank: bestRank, after: merged, positions });
      word = merged;
    }
    const result = { pieces: word, steps };
    this.cache.set(chunk, result);
    return result;
  }

  // Full encode with every intermediate kept for the visualisation.
  encode(text) {
    const re = new RegExp(PRETOKENIZE_PATTERN, "gu");
    const chunks = [];
    const tokens = [];
    for (const m of text.matchAll(re)) {
      const byteStr = this.toByteString(m[0]);
      const { pieces, steps } = this.bpe(byteStr);
      const chunk = { index: chunks.length, text: m[0], start: m.index, byteStr, steps, tokens: [] };
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
