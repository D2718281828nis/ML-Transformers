"""Train the small models shown on the Autoencoder Explainer page and export them to ../assets/.

The models and training recipes are the ones in notebooks/autoencoders/:
  * digits: an AE and four VAEs (beta = 0.1, 1, 2, 4) with a 2-number bottleneck (notebook 01, sections 3-6);
  * ETTh1:  NLinear, forecasting autoencoders with k = 1..32 and the conditional VAE (notebook 02, sections 3, 4, 8).

Weights are stored as base64-encoded little-endian float32 arrays. The script also writes
assets/checks.json with reference outputs that tools/check.mjs compares against the page's JavaScript.

    python explainer/autoencoders/tools/build_assets.py      # needs torch, numpy, pandas, scikit-learn; ~2 min on a CPU
"""

import base64
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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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


# ───────────────────────────── digits (notebook 01) ─────────────────────────────
H = 128


class AE(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, H), nn.ReLU(), nn.Linear(H, k))
        self.decoder = nn.Sequential(nn.Linear(k, H), nn.ReLU(), nn.Linear(H, 64))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class VAE(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, H), nn.ReLU(), nn.Linear(H, 2 * k))
        self.decoder = nn.Sequential(nn.Linear(k, H), nn.ReLU(), nn.Linear(H, 64))

    def encode(self, x):
        return self.encoder(x).chunk(2, dim=-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


def bce(logits, x):
    return F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)


def vae_terms(model, x):
    logits, mu, logvar = model(x)
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(-1)
    return rec.mean(), kl.mean()


def fit_digits(model, loss_fn, Xtr, Xte, epochs=300, lr=2e-3, bs=128, seed=0):
    # Same loop as notebook 01, including the per-epoch evaluation (it draws VAE noise too).
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(perm), bs):
            loss = loss_fn(model, Xtr[perm[i:i + bs]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            loss_fn(model, Xtr), loss_fn(model, Xte)
    return model.eval()


def mlp_layers(seq):
    lin = [m for m in seq if isinstance(m, nn.Linear)]
    return [{"w": tensor(m.weight), "b": tensor(m.bias)} for m in lin]


def build_digits(checks):
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)
    y = digits.target
    idx = np.arange(len(X))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    Xtr, Xte, Xall = torch.from_numpy(X[i_tr]), torch.from_numpy(X[i_te]), torch.from_numpy(X)
    is_test = np.zeros(len(X), dtype=int)
    is_test[i_te] = 1

    models = {}
    torch.manual_seed(0)
    ae = fit_digits(AE(2), lambda m, x: bce(m(x)[0], x), Xtr, Xte)
    with torch.no_grad():
        logits, _ = ae(Xte)
        models["ae"] = {"kind": "ae", "beta": 0, "encoder": mlp_layers(ae.encoder), "decoder": mlp_layers(ae.decoder),
                        "metrics": {"rec": rounded(bce(logits, Xte)), "kl": 0,
                                    "mse": rounded(F.mse_loss(torch.sigmoid(logits), Xte))}}
        checks["digits"] = {"ae": {"z": ae.encoder(Xall[:5]).tolist(),
                                   "decode": torch.sigmoid(ae.decoder(torch.tensor([[1.5, -2.0], [8.0, 3.0]]))).tolist()}}
    for beta in [0.1, 1.0, 2.0, 4.0]:
        torch.manual_seed(0)
        vae = fit_digits(VAE(2), lambda m, x, b=beta: sum(t * w for t, w in zip(vae_terms(m, x), (1.0, b))), Xtr, Xte)
        with torch.no_grad():
            torch.manual_seed(0)
            rec, kl = vae_terms(vae, Xte)
            mu, logvar = vae.encode(Xte)
            mse = F.mse_loss(torch.sigmoid(vae.decoder(mu)), Xte)
            key = f"vae-{beta:g}"
            models[key] = {"kind": "vae", "beta": beta, "encoder": mlp_layers(vae.encoder), "decoder": mlp_layers(vae.decoder),
                           "metrics": {"rec": rounded(rec), "kl": rounded(kl), "mse": rounded(mse)}}
            mu5, lv5 = vae.encode(Xall[:5])
            checks["digits"][key] = {"mu": mu5.tolist(), "logvar": lv5.tolist(),
                                     "decode": torch.sigmoid(vae.decoder(torch.tensor([[0.5, -1.0], [-1.5, 1.2]]))).tolist()}
        print(f"digits {key}: rec {rec:.2f} KL {kl:.2f} mse {mse:.4f}")

    # 64 pixels per image in 0..16, one character each (0-9, a-g)
    alphabet = "0123456789abcdefg"
    images = "".join(alphabet[v] for v in digits.data.astype(int).ravel())
    (ASSETS / "digits.json").write_text(json.dumps({
        "source": "scikit-learn load_digits (UCI optical recognition of handwritten digits), 8x8, values 0..16",
        "n": int(len(X)), "images": images, "labels": "".join(map(str, y)), "test": "".join(map(str, is_test)),
    }, separators=(",", ":")))
    (ASSETS / "digit-models.json").write_text(json.dumps({"hidden": H, "latent": 2, "models": models}, separators=(",", ":")))


# ───────────────────────────── ETTh1 (notebook 02) ─────────────────────────────
MONTH = 30 * 24
L, T = 336, 96


def load_series():
    CACHE.mkdir(parents=True, exist_ok=True)
    csv = CACHE / "ETTh1.csv"
    if not csv.exists():
        try:
            urllib.request.urlretrieve(ETT_URL, csv)
        except urllib.error.URLError:
            # python.org builds on macOS ship without CA certificates; certifi provides them.
            import ssl
            import certifi
            with urllib.request.urlopen(ETT_URL, context=ssl.create_default_context(cafile=certifi.where())) as r:
                csv.write_bytes(r.read())
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
    ASSETS.mkdir(exist_ok=True)
    checks = {}
    build_digits(checks)
    build_forecast(checks)
    (ASSETS / "checks.json").write_text(json.dumps(checks))
    for p in sorted(ASSETS.glob("*.json")):
        print(f"{p.name:22s} {p.stat().st_size / 1024:7.1f} KB")
