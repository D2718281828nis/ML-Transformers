# Embedding & Attention Explainer

An interactive page that follows a sentence through the first part of GPT-2 (small): tokenization (pre-tokenization regex, byte-pair-encoding merges, token IDs), token embeddings, positional encoding, and the first self-attention layer (LayerNorm, Q/K/V, multi-head split, scores, scaling, causal mask, softmax, weighted values, output projection, residual).

It zooms into the part of [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) that the original page covers only briefly.

**Live:** https://d2718281828nis.github.io/ML-Transformers/

## How it works

- **Real tokenizer.** `js/tokenizer.js` is a port of `src/transformers/models/gpt2/tokenization_gpt2.py` that also records every BPE merge step. `assets/gpt2/vocab.json` and `merges.txt` come from [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) (MIT).
- **Real weights, fetched in pieces.** `js/weights.js` reads the safetensors header of `model.safetensors` (548 MB) on the Hugging Face Hub. It then downloads only what the page needs with HTTP Range requests: one 3 KB embedding row per token, the first 128 position rows and the layer-0 attention weights. That is about 10 MB, kept in the browser's Cache Storage for later visits. The model revision is pinned.
- **Plain math, no framework.** `js/model.js` computes the embeddings and the attention sub-layer and keeps every intermediate result for the visualizations. `js/viz.js` holds the drawing helpers, and `js/app.js` renders each section. There is no build step. Formulas use KaTeX from a CDN.

## Run locally

```bash
cd explainer
python3 -m http.server 8000   # then open http://localhost:8000
```

The page has to be served over HTTP (ES modules and `fetch` don't work from `file://`). It needs network access to `huggingface.co` for the weights. Tokenization works without it.

## Verify against `transformers`

```bash
python explainer/tools/verify.py   # needs torch, requests, regex and Node >= 22
```

This runs the page's JavaScript in Node and compares token IDs, input embeddings, all 12 attention maps and the attention output with `GPT2Model` from this repository's `src/transformers`. Attention weights agree to within 1e-6.

## Deploy

`.github/workflows/pages.yml` publishes this folder to GitHub Pages on every push to `main` that touches `explainer/`. One-time setup: **Settings → Pages → Source: GitHub Actions**.
