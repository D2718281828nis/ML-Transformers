"""Check the explainer's JavaScript against this repository's GPT-2 implementation.

Runs js/tokenizer.js + js/model.js in Node (tools/dump_js.mjs) and compares token ids,
input embeddings, all 12 layer-0 attention maps and the attention output with
GPT2Tokenizer / GPT2Model from src/transformers (eager attention).

Only the tensors that are needed are downloaded, with HTTP Range requests, so the
548 MB checkpoint is never fetched in full.

    pip install torch requests regex   # plus the usual transformers deps
    python explainer/tools/verify.py   # needs Node >= 22 on PATH
"""

import json
import pathlib
import struct
import subprocess
import sys

import numpy as np
import requests
import torch


EXPLAINER = pathlib.Path(__file__).resolve().parents[1]
REPO = EXPLAINER.parent
sys.path.insert(0, str(REPO / "src"))

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer  # noqa: E402


URL = "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"
TOKENIZER_CASES = [
    "Attention is all you need",
    "The cat sat on the mat because it was tired.",
    "Кошка сидит на ковре",
    "I'll be there, won't you?  Two  spaces\tand tabs\n\nnewlines",
    "Emoji 🤗 and café naïve 12345 3.14159",
    "   leading spaces and trailing   ",
    "日本語のテキスト",
    "It's they're we've I'm you'll he'd",
    "x = f(y) + 2**10 // 3;",
    "",
]
MODEL_CASES = ["Attention is all you need", "I saw the cat and the dog", "Кошка сидит на ковре"]
TOLERANCE = 1e-4

session = requests.Session()


def fetch(start, end):
    r = session.get(URL, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=60)
    r.raise_for_status()
    assert r.status_code == 206, r.status_code
    return r.content


def main():
    header_len = struct.unpack("<Q", fetch(0, 8))[0]
    header = json.loads(fetch(8, 8 + header_len))
    base = 8 + header_len

    def rows(name, r0, r1):
        info = header[name]
        cols = int(np.prod(info["shape"][1:]))
        start = base + info["data_offsets"][0]
        raw = fetch(start + r0 * cols * 4, start + r1 * cols * 4)
        return torch.from_numpy(np.frombuffer(raw, dtype="<f4").reshape(r1 - r0, cols).copy())

    def tensor(name):
        info = header[name]
        return rows(name, 0, info["shape"][0]).reshape(info["shape"])

    tok = GPT2Tokenizer(str(EXPLAINER / "assets/gpt2/vocab.json"), str(EXPLAINER / "assets/gpt2/merges.txt"))
    model_ids = [tok.encode(t) for t in MODEL_CASES]

    model = GPT2Model(GPT2Config(n_layer=1, attn_implementation="eager")).eval()
    state = {k: tensor(k) for k in header if k.startswith("h.0.") and k != "h.0.attn.bias"}
    state["wpe.weight"] = torch.zeros(1024, 768)
    state["wpe.weight"][:128] = rows("wpe.weight", 0, 128)
    state["wte.weight"] = torch.zeros(50257, 768)
    for i in sorted({i for ids in model_ids for i in ids}):
        state["wte.weight"][i] = rows("wte.weight", i, i + 1)[0]
    model.load_state_dict(state, strict=False)
    captured = {}
    model.h[0].attn.register_forward_hook(lambda m, i, o: captured.__setitem__("attn", o[0]))

    js = json.loads(
        subprocess.run(
            ["node", str(EXPLAINER / "tools/dump_js.mjs"), json.dumps({"tok": TOKENIZER_CASES, "model": MODEL_CASES})],
            check=True, capture_output=True, text=True,
        ).stdout
    )

    ok = True
    same = sum(tok.encode(t) == ids for t, ids in zip(TOKENIZER_CASES, js["tok"]))
    ok &= same == len(TOKENIZER_CASES)
    print(f"tokenizer: {same}/{len(TOKENIZER_CASES)} strings give identical ids")
    for text, ids, got in zip(MODEL_CASES, model_ids, js["model"]):
        with torch.no_grad():
            out = model(torch.tensor([ids]), output_attentions=True, output_hidden_states=True)
        ref = {"x": out.hidden_states[0][0], "probs": out.attentions[0][0], "attnOut": captured["attn"][0]}
        errs = {k: float((torch.tensor(got[k]) - v).abs().max()) for k, v in ref.items()}
        ok &= got["ids"] == ids and max(errs.values()) < TOLERANCE
        print(f"{text!r:32} n={len(ids):2}  " + "  ".join(f"max|Δ {k}|={v:.1e}" for k, v in errs.items()))
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
