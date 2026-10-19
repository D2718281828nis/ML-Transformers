// Loads only the GPT-2 tensors this page needs, straight from the Hugging Face Hub.
//
// model.safetensors is 548 MB, but it is a plain file: 8 bytes of header length,
// a JSON header with each tensor's byte offsets, then raw little-endian float32
// data. HTTP Range requests let us download just the rows we use: one 3 KB row of
// the token-embedding table per token, the first 128 positional rows, and the
// layer-0 attention weights (~10 MB in total).

export const MODEL_REPO = "openai-community/gpt2";
export const MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e";
export const WEIGHTS_URL = `https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/model.safetensors`;

const CACHE_NAME = `gpt2-explainer-${MODEL_REVISION.slice(0, 8)}`;
const HEADER_PROBE = 32 * 1024;

export class RemoteSafetensors {
  constructor(url = WEIGHTS_URL, { onProgress } = {}) {
    this.url = url;
    this.resolvedUrl = null; // signed CDN URL after the Hub redirect; reused to skip the 302
    this.header = null;
    this.dataStart = 0;
    this.onProgress = onProgress || (() => {});
    this.loaded = 0;
    this.expected = 0;
  }

  async #cacheGet(key) {
    try {
      if (typeof caches === "undefined") return null;
      const cache = await caches.open(CACHE_NAME);
      const hit = await cache.match(this.#cacheKey(key));
      return hit ? await hit.arrayBuffer() : null;
    } catch {
      return null;
    }
  }

  async #cachePut(key, buf) {
    try {
      if (typeof caches === "undefined") return;
      const cache = await caches.open(CACHE_NAME);
      await cache.put(this.#cacheKey(key), new Response(buf.slice(0)));
    } catch {
      /* storage unavailable (private mode, quota): just refetch next time */
    }
  }

  #cacheKey(key) {
    return new Request(new URL(`./__gpt2-cache/${encodeURIComponent(key)}`, location.href).href);
  }

  async #fetchRange(start, endExclusive, track = true) {
    const attempt = async (url) => {
      const res = await fetch(url, { headers: { Range: `bytes=${start}-${endExclusive - 1}` } });
      if (res.status !== 206) {
        // A 200 here would mean the server ignored Range and is sending 548 MB.
        res.body?.cancel();
        throw new Error(`Range request failed: HTTP ${res.status}`);
      }
      if (res.url && res.url !== url) this.resolvedUrl = res.url;
      if (!track || !res.body) return res.arrayBuffer();
      const reader = res.body.getReader();
      const out = new Uint8Array(endExclusive - start);
      let offset = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        out.set(value, offset);
        offset += value.length;
        this.loaded += value.length;
        this.onProgress(this.loaded, this.expected);
      }
      return out.buffer;
    };
    if (this.resolvedUrl) {
      try {
        return await attempt(this.resolvedUrl);
      } catch {
        this.resolvedUrl = null; // signed URL expired: go through the Hub again
      }
    }
    return attempt(this.url);
  }

  async init() {
    if (this.header) return;
    const cached = await this.#cacheGet("header");
    let headerJson;
    if (cached) {
      headerJson = new TextDecoder().decode(cached);
    } else {
      const probe = new Uint8Array(await this.#fetchRange(0, HEADER_PROBE, false));
      const n = Number(new DataView(probe.buffer).getBigUint64(0, true));
      const bytes = 8 + n <= probe.length ? probe.subarray(8, 8 + n) : new Uint8Array(await this.#fetchRange(8, 8 + n, false));
      headerJson = new TextDecoder().decode(bytes);
      await this.#cachePut("header", new TextEncoder().encode(headerJson).buffer);
    }
    this.header = JSON.parse(headerJson);
    this.dataStart = 8 + new TextEncoder().encode(headerJson).length;
  }

  info(name) {
    const t = this.header[name];
    if (!t) throw new Error(`Tensor ${name} not found`);
    if (t.dtype !== "F32") throw new Error(`Tensor ${name} is ${t.dtype}, expected F32`);
    return t;
  }

  // Bytes that rows(...) / tensor(...) calls will download, for the progress bar.
  expect(bytes) {
    this.expected += bytes;
    this.onProgress(this.loaded, this.expected);
  }

  // `count` consecutive rows of a 2-D tensor starting at `row`, as one Float32Array.
  async rows(name, row, count) {
    const t = this.info(name);
    const cols = t.shape.length > 1 ? t.shape.slice(1).reduce((a, b) => a * b, 1) : 1;
    const rowBytes = cols * 4;
    const start = this.dataStart + t.data_offsets[0] + row * rowBytes;
    const end = start + count * rowBytes;
    const key = `${name}:${row}:${count}`;
    let buf = await this.#cacheGet(key);
    if (buf) {
      this.loaded += buf.byteLength;
      this.onProgress(this.loaded, this.expected);
    } else {
      buf = await this.#fetchRange(start, end);
      await this.#cachePut(key, buf);
    }
    return new Float32Array(buf);
  }

  async tensor(name) {
    const t = this.info(name);
    return this.rows(name, 0, t.shape[0]);
  }

  tensorBytes(name) {
    const t = this.info(name);
    return t.data_offsets[1] - t.data_offsets[0];
  }
}

// Layer-0 attention weights + the start of the position table.
export const POSITION_ROWS = 128;
export const LAYER0_TENSORS = [
  "h.0.ln_1.weight",
  "h.0.ln_1.bias",
  "h.0.attn.c_attn.weight",
  "h.0.attn.c_attn.bias",
  "h.0.attn.c_proj.weight",
  "h.0.attn.c_proj.bias",
];

export async function loadStaticWeights(st) {
  await st.init();
  const wpeBytes = POSITION_ROWS * st.info("wpe.weight").shape[1] * 4;
  st.expect(LAYER0_TENSORS.reduce((s, n) => s + st.tensorBytes(n), 0) + wpeBytes);
  const [lnW, lnB, attnW, attnB, projW, projB, wpe] = await Promise.all([
    ...LAYER0_TENSORS.map((n) => st.tensor(n)),
    st.rows("wpe.weight", 0, POSITION_ROWS),
  ]);
  return { lnW, lnB, attnW, attnB, projW, projB, wpe };
}

// Token-embedding rows, fetched on demand and kept for the session.
export class EmbeddingRows {
  constructor(st) {
    this.st = st;
    this.rows = new Map();
    this.pending = new Map();
  }

  async get(ids) {
    const d = this.st.info("wte.weight").shape[1];
    const missing = [...new Set(ids)].filter((id) => !this.rows.has(id) && !this.pending.has(id));
    if (missing.length) this.st.expect(missing.length * d * 4);
    for (const id of missing) {
      const p = this.st.rows("wte.weight", id, 1).then((row) => {
        this.rows.set(id, row);
        this.pending.delete(id);
      });
      p.catch(() => this.pending.delete(id));
      this.pending.set(id, p);
    }
    await Promise.all(ids.map((id) => this.pending.get(id)).filter(Boolean));
    return ids.map((id) => this.rows.get(id));
  }
}
