"""Train the small models shown on the Autoencoder Explainer page and export them to ../assets/.

The models and training recipes are the ones in notebooks/autoencoders/:
  * digits: a convolutional AE and four VAEs with a 2-number bottleneck, trained on MNIST resized to 64 x 64
    (notebook 01 builds the same ideas step by step on scikit-learn's 8 x 8 digits);
  * ETTh1:  NLinear, forecasting autoencoders with k = 1..32 and the conditional VAE (notebook 02, sections 3, 4, 8).

Weights are stored as base64-encoded little-endian float32 arrays. The script also writes
assets/checks.json with reference outputs that tools/check.mjs compares against the page's JavaScript.

    python explainer/autoencoders/tools/build_assets.py            # everything; needs torch, numpy, pandas
    python explainer/autoencoders/tools/build_assets.py digits     # only the digit models (or: forecast)

The digit models train on Apple's GPU (MPS) or CUDA when available: ~4 min on an M-series Mac, much longer on a CPU.
GPU training is not bit-for-bit reproducible, so a rebuild gives slightly different (equally good) models.
"""

import base64
import gzip
import json
import math
import pathlib
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CACHE = ROOT / "tools" / ".cache"
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
torch.set_num_threads(4)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def tensor(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return {"shape": list(a.shape), "data": b64(a)}


def rounded(x, nd=5):
    return round(float(x), nd)


# ───────────────────────────── digits: MNIST at 64 x 64 ─────────────────────────────
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
SIDE = 64         # the models see 64 x 64 images: MNIST's 28 x 28, resized bilinearly (js/digits.js resizes the same way)
N_SHOWN = 2000    # test images shipped to the page
C, H = 8, 64      # conv channels (C, 2C, 2C, 2C) and the dense layer between the 4 x 4 feature map and the code
BETAS = [0.1, 1.0, 16.0, 128.0]  # 128 is past the point of posterior collapse
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def fetch(url, path):
    try:
        urllib.request.urlretrieve(url, path)
    except urllib.error.URLError:
        # python.org builds on macOS ship without CA certificates; certifi provides them.
        import ssl
        import certifi
        with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where())) as r:
            path.write_bytes(r.read())


def load_mnist(split):
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = []
    for kind, offset in (("images-idx3", 16), ("labels-idx1", 8)):
        name = f"{split}-{kind}-ubyte.gz"
        if not (CACHE / name).exists():
            fetch(MNIST_URL + name, CACHE / name)
        arrays.append(np.frombuffer(gzip.decompress((CACHE / name).read_bytes()), np.uint8, offset=offset))
    return arrays[0].reshape(-1, 28, 28), arrays[1]


def resize(x28):
    """[n, 28, 28] in 0..1 -> [n, 1, 64, 64]"""
    return F.interpolate(torch.as_tensor(x28, dtype=torch.float32).unsqueeze(1), size=SIDE, mode="bilinear", align_corners=False).clamp(0, 1)


def encoder(out):
    # 64 -> 32 -> 16 -> 8 -> 4 pixels, then dense layers down to the code
    return nn.Sequential(
        nn.Conv2d(1, C, 4, 2, 1), nn.ReLU(), nn.Conv2d(C, 2 * C, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(2 * C, 2 * C, 4, 2, 1), nn.ReLU(), nn.Conv2d(2 * C, 2 * C, 4, 2, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(2 * C * 16, H), nn.ReLU(), nn.Linear(H, out))


def decoder(k=2):
    # the mirror image: dense layers up to a 4 x 4 feature map, then transposed convolutions back to 64 x 64 logits
    return nn.Sequential(
        nn.Linear(k, H), nn.ReLU(), nn.Linear(H, 2 * C * 16), nn.ReLU(), nn.Unflatten(1, (2 * C, 4, 4)),
        nn.ConvTranspose2d(2 * C, 2 * C, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(2 * C, 2 * C, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(2 * C, C, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(C, 1, 4, 2, 1))


class AE(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.encoder, self.decoder = encoder(k), decoder(k)

    def encode(self, x):
        return self.encoder(x), None

    def loss(self, x):
        return bce(self.decoder(self.encoder(x)), x), torch.zeros(())


class VAE(nn.Module):
    def __init__(self, k=2, beta=1.0):
        super().__init__()
        self.beta = beta
        self.encoder, self.decoder = encoder(2 * k), decoder(k)

    def encode(self, x):
        return self.encoder(x).chunk(2, dim=-1)

    def loss(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(-1).mean()
        return bce(self.decoder(z), x), kl


def bce(logits, x):
    # summed over the 4096 pixels, averaged over images: nats per image
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum((1, 2, 3)).mean()


def fit_digits(model, Xtr, epochs=15, lr=2e-3, bs=256, seed=0):
    torch.manual_seed(seed)
    model.to(DEVICE).train()
    Xtr = Xtr.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    beta = getattr(model, "beta", 0.0)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(perm), bs):
            rec, kl = model.loss(Xtr[perm[i:i + bs]])
            opt.zero_grad()
            (rec + beta * kl).backward()
            opt.step()
        sched.step()
    return model.cpu().eval()


def seq_layers(seq):
    out = []
    for m in seq:
        if isinstance(m, nn.Linear):
            out.append({"type": "linear", "w": tensor(m.weight), "b": tensor(m.bias)})
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            out.append({"type": "conv" if isinstance(m, nn.Conv2d) else "convT", "w": tensor(m.weight), "b": tensor(m.bias),
                        "stride": m.stride[0], "pad": m.padding[0]})
        elif isinstance(m, nn.ReLU):
            out.append({"type": "relu"})
        elif isinstance(m, nn.Flatten):
            out.append({"type": "flatten"})
        elif isinstance(m, nn.Unflatten):
            out.append({"type": "unflatten", "shape": list(m.unflattened_size)})
        else:
            raise TypeError(m)
    return out


def build_digits(checks):
    x_tr, _ = load_mnist("train")
    x_te, y_te = load_mnist("t10k")
    Xtr, Xte = resize(x_tr / 255.0), resize(x_te / 255.0)
    # The page gets the first N_SHOWN test images at 28 x 28 in 17 levels (one character each) and resizes them itself,
    # so the codes shipped with each model are computed from exactly those images.
    q = np.rint(x_te[:N_SHOWN] / 255.0 * 16).astype(int)
    Xshown = resize(q / 16.0)

    candidates = [("ae", AE())] + [(f"vae-{b:g}", VAE(beta=b)) for b in BETAS]
    models = {}
    checks["digits"] = {}
    for key, model in candidates:
        model = fit_digits(model, Xtr)
        with torch.no_grad():
            torch.manual_seed(0)
            parts = [(*model.loss(x), F.mse_loss(torch.sigmoid(model.decoder(model.encode(x)[0])), x)) for x in Xte.split(1000)]
            rec, kl, mse = (sum(p[j] for p in parts) / len(parts) for j in range(3))
            mu, logvar = model.encode(Xshown)
            probe = torch.tensor([[0.5, -1.0], [-1.5, 1.2]])
            models[key] = {"kind": "vae" if logvar is not None else "ae", "beta": getattr(model, "beta", 0),
                           "encoder": seq_layers(model.encoder), "decoder": seq_layers(model.decoder),
                           "codes": tensor(mu), "logvar": tensor(logvar) if logvar is not None else None,
                           "metrics": {"rec": rounded(rec), "kl": rounded(kl), "mse": rounded(mse)}}
            checks["digits"][key] = {"mu": mu[:5].tolist(), "logvar": logvar[:5].tolist() if logvar is not None else None,
                                     "decode": torch.sigmoid(model.decoder(probe)).flatten(1).tolist()}
        print(f"digits {key}: rec {rec:.1f} KL {kl:.2f} mse {mse:.4f}")

    alphabet = "0123456789abcdefg"
    (ASSETS / "digits.json").write_text(json.dumps({
        "source": "MNIST test set (LeCun, Cortes & Burges), first 2000 images, 28x28 in 17 levels (0..g); the page resizes them to 64x64",
        "n": N_SHOWN, "side": 28, "images": "".join(alphabet[v] for v in q.ravel()), "labels": "".join(map(str, y_te[:N_SHOWN])),
    }, separators=(",", ":")))
    (ASSETS / "digit-models.json").write_text(json.dumps({"side": SIDE, "latent": 2, "models": models}, separators=(",", ":")))


# ───────────────────────────── ETTh1 (notebook 02) ─────────────────────────────
MONTH = 30 * 24
L, T = 336, 96


def load_series():
    CACHE.mkdir(parents=True, exist_ok=True)
    csv = CACHE / "ETTh1.csv"
    if not csv.exists():
        fetch(ETT_URL, csv)
    df = pd.read_csv(csv, parse_dates=["date"])
    raw = df.iloc[:, 1:].to_numpy(np.float32)
    mean, std = raw[:12 * MONTH].mean(0), raw[:12 * MONTH].std(0)
    return df, list(df.columns[1:]), torch.from_numpy((raw - mean) / std), mean, std


DF, VARS, SERIES, MEAN, STD = load_series()
BORDERS = {"train": (0, 12 * MONTH), "val": (12 * MONTH, 16 * MONTH), "test": (16 * MONTH, 20 * MONTH)}


def windows(L, T, split):
    a, b = BORDERS[split]
    a = a if split == "train" else a - L
    seg = SERIES[a:b]
    n = len(seg) - L - T + 1
    idx = torch.arange(n)[:, None] + torch.arange(L + T)[None]
    w = seg[idx].permute(0, 2, 1).reshape(-1, L + T)
    return w[:, :L].contiguous(), w[:, L:].contiguous()


class NLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(L, T)

    def forward(self, x):
        last = x[:, -1:]
        return self.lin(x - last) + last


class ForecastAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.E, self.D = nn.Linear(L, k), nn.Linear(k, T)

    def forward(self, x):
        last = x[:, -1:]
        return self.D(self.E(x - last)) + last


class ForecastCVAE(nn.Module):
    def __init__(self, k=16, kz=8, hidden=64):
        super().__init__()
        self.E, self.D = nn.Linear(L, k), nn.Linear(k, T)
        self.h = nn.Linear(L, hidden)
        self.prior = nn.Linear(hidden, 2 * kz)
        self.posterior = nn.Sequential(nn.Linear(hidden + T, hidden), nn.Tanh(), nn.Linear(hidden, 2 * kz))
        self.U = nn.Linear(kz, T, bias=False)
        self.log_sigma = nn.Parameter(torch.full((T,), -0.5))

    def _base(self, x):
        last = x[:, -1:]
        xn = x - last
        return self.D(self.E(xn)) + last, torch.tanh(self.h(xn))

    def elbo_terms(self, x, y):
        base, h = self._base(x)
        mu_p, logvar_p = self.prior(h).chunk(2, -1)
        mu_q, logvar_q = self.posterior(torch.cat([h, y - base], -1)).chunk(2, -1)
        z = mu_q + torch.randn_like(mu_q) * torch.exp(0.5 * logvar_q)
        mean = base + self.U(z)
        var = torch.exp(2 * self.log_sigma)
        nll = (0.5 * (y - mean) ** 2 / var + self.log_sigma + 0.5 * math.log(2 * math.pi)).sum(-1)
        kl = 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1).sum(-1)
        return nll, kl

    def forward(self, x):
        base, h = self._base(x)
        return base + self.U(self.prior(h).chunk(2, -1)[0])

    @torch.no_grad()
    def sample(self, x, n):
        base, h = self._base(x)
        mu_p, logvar_p = self.prior(h).chunk(2, -1)
        z = mu_p + torch.randn(n, *mu_p.shape) * torch.exp(0.5 * logvar_p)
        return base + self.U(z) + torch.randn(n, *base.shape) * self.log_sigma.exp()


def mse_loss(model, x, y):
    return F.mse_loss(model(x), y)


def cvae_loss(model, x, y):
    if not model.training:
        torch.manual_seed(1)
    nll, kl = model.elbo_terms(x, y)
    return (nll + kl).mean() / y.shape[1]


def fit(model, loss_fn=mse_loss, epochs=10, lr=0.005, bs=32, seed=0):
    x, y = windows(L, T, "train")
    xv, yv = windows(L, T, "val")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        for group in opt.param_groups:
            group["lr"] = lr * 0.5 ** epoch
        model.train()
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), bs):
            b = perm[i:i + bs]
            loss = loss_fn(model, x[b], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model, xv, yv).item()
        if val < best - 1e-7:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def scores(model, transform=None):
    x, y = windows(L, T, "test")
    if transform is not None:
        x = transform(x)
    p = model(x)
    return {"mse": rounded(F.mse_loss(p, y)), "mae": rounded((p - y).abs().mean())}


def shuffle(x):
    return x[:, torch.randperm(x.shape[1], generator=torch.Generator().manual_seed(0))]


def half_exchange(x):
    h = x.shape[1] // 2
    return torch.cat([x[:, h:], x[:, :h]], dim=1)


@torch.no_grad()
def prob_scores(sampler, x, y, n=100, chunk=2500):
    torch.manual_seed(0)
    inside = width = crps = 0.0
    for i in range(0, len(x), chunk):
        s = sampler(x[i:i + chunk], n)
        yy = y[i:i + chunk]
        lo, hi = s.quantile(0.1, 0), s.quantile(0.9, 0)
        inside += ((yy >= lo) & (yy <= hi)).float().sum().item()
        width += (hi - lo).sum().item()
        ss, _ = s.sort(0)
        w = (2 * torch.arange(1, n + 1) - n - 1).float().view(-1, 1, 1)
        crps += ((s - yy[None]).abs().mean(0) - (w * ss).sum(0) / n ** 2).sum().item()
    total = y.numel()
    return {"coverage80": rounded(inside / total, 4), "width80": rounded(width / total, 4), "crps": rounded(crps / total, 4)}


def build_forecast(checks):
    x_te, y_te = windows(L, T, "test")
    out = {"L": L, "T": T, "vars": VARS}

    torch.manual_seed(0)
    nl = fit(NLinear())
    out["nlinear"] = {"W": tensor(nl.lin.weight), "b": tensor(nl.lin.bias), "test": scores(nl),
                      "shuffled": scores(nl, shuffle), "halfExchanged": scores(nl, half_exchange)}
    out["repeat"] = {"test": scores(lambda x: x[:, -1:].expand(-1, T))}
    print("NLinear", out["nlinear"]["test"])

    out["ae"] = {}
    for k in [1, 2, 4, 8, 16, 32]:
        torch.manual_seed(0)
        m = fit(ForecastAE(k))
        entry = {"E": tensor(m.E.weight), "bE": tensor(m.E.bias), "D": tensor(m.D.weight), "bD": tensor(m.D.bias),
                 "test": scores(m)}
        if k == 16:
            entry["shuffled"], entry["halfExchanged"] = scores(m, shuffle), scores(m, half_exchange)
            ae16 = m
        out["ae"][str(k)] = entry
        print(f"AE k={k}", entry["test"])

    torch.manual_seed(0)
    cvae = fit(ForecastCVAE(), cvae_loss, lr=0.003, bs=64)
    xv, yv = windows(L, T, "val")
    with torch.no_grad():
        resid_sd = (ae16(xv) - yv).std(0)
    baseline = prob_scores(lambda x, n: ae16(x)[None] + torch.randn(n, len(x), T) * resid_sd, x_te, y_te)
    prob = prob_scores(cvae.sample, x_te, y_te)
    out["cvae"] = {
        "E": tensor(cvae.E.weight), "bE": tensor(cvae.E.bias), "D": tensor(cvae.D.weight), "bD": tensor(cvae.D.bias),
        "H": tensor(cvae.h.weight), "bH": tensor(cvae.h.bias), "P": tensor(cvae.prior.weight), "bP": tensor(cvae.prior.bias),
        "U": tensor(cvae.U.weight), "logSigma": tensor(cvae.log_sigma),
        "test": {**scores(cvae), **prob}, "baseline": {**scores(ae16), **baseline, "residSd": tensor(resid_sd)},
    }
    print("CVAE", out["cvae"]["test"], "baseline", {k: v for k, v in baseline.items()})

    # the series the page needs: L hours before the test split + the 4 test months, all 7 variables
    a, b = BORDERS["test"][0] - L, BORDERS["test"][1]
    out["series"] = {"start": str(DF.date.iloc[a]), "testOffset": L, "n": b - a, "data": tensor(SERIES[a:b].T.contiguous()),
                     "mean": [rounded(v, 4) for v in MEAN], "std": [rounded(v, 4) for v in STD]}

    with torch.no_grad():
        probe = x_te[[0, 777, 5000]]
        checks["forecast"] = {
            "windows": [[0, 0], [111, 0], [714, 2]],          # [window index, variable] of probe rows (row = window * 7 + variable)
            "nlinear": nl(probe).tolist(),
            "ae8": out_check(ForecastAE, out["ae"]["8"], probe),
            "cvaeMean": cvae(probe).tolist(),
        }
    (ASSETS / "etth1.json").write_text(json.dumps(out, separators=(",", ":")))


def out_check(cls, entry, probe):
    k = entry["E"]["shape"][0]
    m = cls(k)
    dec = lambda t: torch.from_numpy(np.frombuffer(base64.b64decode(t["data"]), dtype="<f4").reshape(t["shape"]).copy())
    m.E.weight.data, m.E.bias.data, m.D.weight.data, m.D.bias.data = dec(entry["E"]), dec(entry["bE"]), dec(entry["D"]), dec(entry["bD"])
    return m(probe).tolist()


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["digits", "forecast"]
    ASSETS.mkdir(exist_ok=True)
    checks_file = ASSETS / "checks.json"
    checks = json.loads(checks_file.read_text()) if checks_file.exists() else {}
    if "digits" in parts:
        build_digits(checks)
    if "forecast" in parts:
        build_forecast(checks)
    checks_file.write_text(json.dumps(checks))
    for p in sorted(ASSETS.glob("*.json")):
        print(f"{p.name:22s} {p.stat().st_size / 1024:7.1f} KB")
