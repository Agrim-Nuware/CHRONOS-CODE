import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import write_notebook, md, code

cells = []

cells.append(md("""\
# S&P 500 Zero-Shot Forecasting — Moirai Family

Runs **Moirai-1.0-R-large**, **Moirai-MoE-1.0-R-base**, and **Moirai-2.0-R-small** zero-shot
(no fine-tuning) on a rolling backtest of the S&P 500, across 3 targets (price level,
simple return, log return) and 2 horizons (5-day, 21-day).

Moirai-1.0 and Moirai-MoE also get a **covariate variant** using VIX and the 10-year
Treasury yield (TNX) as past-only dynamic covariates (`past_feat_dynamic_real`).
Moirai-2.0 is univariate-only — the uni2ts authors' own code comments flag its
multivariate/covariate path as untested, and the official example notebook only
demonstrates covariates for Moirai-1.0/MoE, so we don't rely on it here.

**Before running:** Runtime -> Change runtime type -> GPU.

Output: `moirai_results.csv`, one row per (version, variant, target, horizon, window).
"""))

cells.append(md("""\
**Dependency list below is uni2ts's actual `pyproject.toml` `dependencies`**, not a
guess -- deliberately install everything it declares **except `torch` and `numpy`**:
uni2ts pins `torch>=2.1,<2.5` and `numpy~=1.26.0`, and letting pip "satisfy" those on
Colab would downgrade its preinstalled GPU-enabled torch build (and numpy, which half
of Colab's preinstalled stack is compiled against) -- a much worse problem than
leaving those two slightly out of uni2ts's preferred range. `--no-deps` on the
`pip install -e .` step keeps uni2ts's own install from re-triggering that same
resolution.
"""))

cells.append(code("""\
!pip install -q "lightning>=2.0" "gluonts~=0.14.3" "scipy~=1.11.3" "einops==0.7.*" \\
    "jaxtyping~=0.2.24" "python-dotenv==1.0.0" "hydra-core==1.3" orjson tensorboard \\
    multiprocess "huggingface_hub>=0.23.0" safetensors "datasets~=2.17.1" "jax[cpu]" \\
    yfinance

# gluonts pins pandas<2.2.0, so the install above silently downgrades pandas on disk
# (Colab ships something newer). Pin it back up explicitly -- see the note below on
# why this alone isn't enough.
!pip install -q -U "pandas>=2.2"

# Always clone fresh: if an earlier attempt in this session left a partial/broken
# `uni2ts` directory behind, a skip-if-exists check would silently keep reusing it.
!rm -rf uni2ts
!git clone https://github.com/SalesforceAIResearch/uni2ts.git
!ls uni2ts/src/uni2ts/model   # sanity check -- should list moirai, moirai_moe, moirai2, ...
!cd uni2ts && pip install -e . --no-deps
"""))

cells.append(md("""\
## This cell force-restarts the runtime -- expected, not a bug

Colab's kernel has pandas already loaded **before this notebook's first cell even
runs** (it preloads pandas for its own dataframe-rendering UI) -- both the pure-Python
layer and pandas' compiled C extensions (`pandas._libs.*`). `gluonts`'s install two
cells up downgrades pandas' files on disk to satisfy its `pandas<2.2.0` pin, and we
pin it back up right after -- but that only changes files on disk, not what's already
loaded in this process's memory. Once a compiled extension has been loaded, deleting it
from `sys.modules` and re-importing does **not** force it to reload -- that's a
pure-Python-only trick, and pandas' `_libs.algos`/`_libs.hashtable` etc. are compiled.
The only reliable fix is a real process restart, so this cell forces one on purpose.

**What happens next:** running the cell below will show something like "Your session
crashed" -- that's the restart, not a failure. Everything installed/cloned above lives
on Colab's VM disk and survives the restart; only in-memory Python state resets. After
it restarts, **continue by running the cells below this one** (select the next cell and
use Runtime -> "Run after", or just click through) -- don't re-run the install cell
above, and don't use "Run all" again from the top, since that would re-trigger the same
gluonts pandas downgrade before the next restart-worthy step.
"""))

cells.append(code("""\
import os
os.kill(os.getpid(), 9)
"""))

cells.append(code("""\
import os
import sys

sys.path.insert(0, os.path.abspath("uni2ts/src"))

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from einops import rearrange

print("pandas:", pd.__version__)
assert not pd.__version__.startswith("2.1"), (
    "pandas is still on the gluonts-downgraded version -- the restart cell above "
    "didn't actually restart the kernel. Runtime -> Restart session manually, then "
    "re-run from this cell (not from the top)."
)

import uni2ts
print("uni2ts loaded from:", uni2ts.__file__)
assert os.path.isdir(os.path.join(os.path.dirname(uni2ts.__file__), "model")), (
    "uni2ts.model missing -- the git clone likely failed silently; check the install cell's output"
)

print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""))

cells.append(md("## Config (identical across all 4 model-family notebooks)"))

cells.append(code("""\
TICKERS = {"price": "^GSPC", "vix": "^VIX", "tnx": "^TNX"}
DATA_START = "2005-01-01"
TARGETS = ["price", "simple_return", "log_return"]
CONTEXT_LENGTH = 512
HORIZONS = [5, 21]
NUM_WINDOWS = 60
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
"""))

cells.append(md("## Data: fetch + align S&P 500 / VIX / TNX"))

cells.append(code("""\
def fetch_raw(start=DATA_START, tickers=TICKERS):
    frames = {}
    for name, symbol in tickers.items():
        raw = yf.download(symbol, start=start, progress=False, auto_adjust=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {symbol}")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # MultiIndex columns on newer yfinance
            close = close.iloc[:, 0]
        frames[name] = close.rename(name)
    combined = pd.concat(frames.values(), axis=1, join="inner")
    combined.columns = list(frames.keys())
    return combined.sort_index().dropna()


def build_targets(raw):
    out = raw.copy()
    out["simple_return"] = out["price"].pct_change()
    out["log_return"] = np.log(out["price"] / out["price"].shift(1))
    return out.dropna()


df = build_targets(fetch_raw())
print(f"Loaded {len(df)} rows: {df.index.min().date()} -> {df.index.max().date()}")
df.tail()
"""))

cells.append(md("## Rolling-origin backtest windows (same origins reused across both horizons)"))

cells.append(code("""\
def make_origins(n_rows, context_length, max_horizon, num_windows):
    lo = context_length - 1
    hi = n_rows - max_horizon - 1
    if hi <= lo:
        raise ValueError("Not enough rows for the requested context/horizon.")
    return np.unique(np.linspace(lo, hi, num=num_windows, dtype=int)).tolist()


def build_window(frame, origin_idx, context_length, horizon):
    ctx_start = origin_idx - context_length + 1
    ctx_end = origin_idx + 1
    fut_end = ctx_end + horizon
    return frame.iloc[ctx_start:ctx_end], frame.iloc[ctx_end:fut_end]


ORIGINS = make_origins(len(df), CONTEXT_LENGTH, max(HORIZONS), NUM_WINDOWS)
print(f"{len(ORIGINS)} windows, origins from {df.index[ORIGINS[0]].date()} to {df.index[ORIGINS[-1]].date()}")
"""))

cells.append(md("## Metrics: MASE, sMAPE, WQL (plain numpy, no gluonts.ev dependency)"))

cells.append(code("""\
def mase(actual, point_forecast, context, seasonality=1):
    actual, point_forecast, context = (np.asarray(x, dtype=float) for x in (actual, point_forecast, context))
    scale = np.mean(np.abs(context[seasonality:] - context[:-seasonality]))
    if scale == 0 or not np.isfinite(scale):
        return np.nan
    return float(np.mean(np.abs(actual - point_forecast)) / scale)


def smape(actual, point_forecast, eps=1e-8):
    actual, point_forecast = np.asarray(actual, dtype=float), np.asarray(point_forecast, dtype=float)
    denom = np.abs(actual) + np.abs(point_forecast) + eps
    return float(100.0 * np.mean(2.0 * np.abs(actual - point_forecast) / denom))


def wql(actual, quantile_forecasts):
    actual = np.asarray(actual, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    level_losses = []
    for level, pred in quantile_forecasts.items():
        pred = np.asarray(pred, dtype=float)
        diff = actual - pred
        pinball = np.where(diff >= 0, level * diff, (level - 1.0) * diff)
        level_losses.append(2.0 * np.sum(pinball) / denom)
    return float(np.mean(level_losses))


def compute_window_metrics(actual, point_forecast, quantile_forecasts, context):
    return {
        "mase": mase(actual, point_forecast, context),
        "smape": smape(actual, point_forecast),
        "wql": wql(actual, quantile_forecasts),
    }
"""))

cells.append(md("## Moirai model registry + cached loading"))

cells.append(code("""\
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

MOIRAI_REGISTRY = [
    {"version": "1.0", "checkpoint": "Salesforce/moirai-1.0-R-large", "cls": "moirai", "patch_size": 32, "variants": ["uni", "cov"]},
    {"version": "moe", "checkpoint": "Salesforce/moirai-moe-1.0-R-base", "cls": "moirai-moe", "patch_size": 16, "variants": ["uni", "cov"]},
    {"version": "2.0", "checkpoint": "Salesforce/moirai-2.0-R-small", "cls": "moirai2", "patch_size": None, "variants": ["uni"]},
]

_module_cache = {}


def get_module(cls_name, checkpoint):
    key = (cls_name, checkpoint)
    if key not in _module_cache:
        if cls_name == "moirai":
            _module_cache[key] = MoiraiModule.from_pretrained(checkpoint)
        elif cls_name == "moirai-moe":
            _module_cache[key] = MoiraiMoEModule.from_pretrained(checkpoint)
        elif cls_name == "moirai2":
            _module_cache[key] = Moirai2Module.from_pretrained(checkpoint)
        else:
            raise ValueError(cls_name)
    return _module_cache[key]


_wrapper_cache = {}


def get_wrapper(cls_name, checkpoint, patch_size, horizon, past_feat_dim):
    key = (cls_name, checkpoint, horizon, past_feat_dim)
    if key in _wrapper_cache:
        return _wrapper_cache[key]
    module = get_module(cls_name, checkpoint)
    common = dict(
        prediction_length=horizon,
        context_length=CONTEXT_LENGTH,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=past_feat_dim,
    )
    if cls_name == "moirai":
        wrapper = MoiraiForecast(module=module, patch_size=patch_size, num_samples=100, **common)
    elif cls_name == "moirai-moe":
        wrapper = MoiraiMoEForecast(module=module, patch_size=patch_size, num_samples=100, **common)
    elif cls_name == "moirai2":
        common.pop("past_feat_dynamic_real_dim")
        wrapper = Moirai2Forecast(module=module, past_feat_dynamic_real_dim=0, **common)
    wrapper = wrapper.to(DEVICE)
    wrapper.eval()
    _wrapper_cache[key] = wrapper
    return wrapper
"""))

cells.append(md("## Forecast call (direct forward/predict, bypassing GluonTS dataset plumbing since we use custom windows)"))

cells.append(code("""\
def moirai_forecast(cls_name, checkpoint, patch_size, horizon, target_values, cov_values=None):
    past_feat_dim = 0 if cov_values is None else cov_values.shape[1]
    wrapper = get_wrapper(cls_name, checkpoint, patch_size, horizon, past_feat_dim)

    if cls_name == "moirai2":
        kwargs = {}
        preds = wrapper.predict([target_values.astype(np.float32)], **kwargs)  # (1, num_quantiles, horizon)
        preds = preds[0]
        levels = list(wrapper.module.quantile_levels)
        quantile_forecasts = {q: preds[i] for i, q in enumerate(levels)}
        point_forecast = quantile_forecasts[0.5]
        return point_forecast, quantile_forecasts

    past_target = rearrange(torch.as_tensor(target_values, dtype=torch.float32, device=DEVICE), "t -> 1 t 1")
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
    kwargs = {}
    if cov_values is not None:
        cov_tensor = torch.as_tensor(cov_values, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        kwargs["past_feat_dynamic_real"] = cov_tensor
        kwargs["past_observed_feat_dynamic_real"] = torch.ones_like(cov_tensor, dtype=torch.bool)

    with torch.no_grad():
        samples = wrapper(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            **kwargs,
        )
    samples = samples[0].detach().cpu().numpy()  # (num_samples, horizon)
    point_forecast = np.median(samples, axis=0)
    quantile_forecasts = {q: np.quantile(samples, q, axis=0) for q in QUANTILE_LEVELS}
    return point_forecast, quantile_forecasts
"""))

cells.append(md("## Main sweep: every (version, variant, target, horizon, window)"))

cells.append(code("""\
rows = []
for spec in MOIRAI_REGISTRY:
    for variant in spec["variants"]:
        for horizon in HORIZONS:
            for target in TARGETS:
                for window_id, origin in enumerate(ORIGINS):
                    context_df, future_df = build_window(df, origin, CONTEXT_LENGTH, horizon)
                    target_values = context_df[target].to_numpy()
                    actual = future_df[target].to_numpy()
                    cov_values = context_df[["vix", "tnx"]].to_numpy() if variant == "cov" else None

                    point_forecast, quantile_forecasts = moirai_forecast(
                        spec["cls"], spec["checkpoint"], spec["patch_size"], horizon, target_values, cov_values
                    )
                    m = compute_window_metrics(actual, point_forecast, quantile_forecasts, target_values)
                    rows.append({
                        "family": "moirai",
                        "version": spec["version"],
                        "checkpoint": spec["checkpoint"],
                        "variant": variant,
                        "target": target,
                        "horizon": horizon,
                        "window_id": window_id,
                        "origin_date": str(df.index[origin].date()),
                        **m,
                    })
        print(f"done: {spec['checkpoint']} variant sweep")

results_df = pd.DataFrame(rows)
results_df.to_csv("moirai_results.csv", index=False)
print(f"\\nWrote moirai_results.csv ({len(results_df)} rows)")
results_df.groupby(["version", "variant", "target", "horizon"])[["mase", "smape", "wql"]].mean()
"""))

cells.append(md("## Download results"))

cells.append(code("""\
from google.colab import files
files.download("moirai_results.csv")
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_moirai_family.ipynb"))
