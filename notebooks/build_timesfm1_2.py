import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import write_notebook, md, code

cells = []

cells.append(md("""\
# S&P 500 Zero-Shot Forecasting — TimesFM 1.0 & 2.0

Runs **timesfm-1.0-200m** and **timesfm-2.0-500m** zero-shot (no fine-tuning) on a
rolling backtest of the S&P 500, across 3 targets (price level, simple return, log
return) and 2 horizons (5-day, 21-day). Univariate only.

**Why TimesFM is split into two notebooks:** TimesFM 1.0/2.0 and 2.5 ship as
*different, incompatible pip package versions* under the same `timesfm` import name
(1.0/2.0 need `timesfm==1.3.0`'s older `TimesFm`/`TimesFmHparams` API; 2.5 needs the
latest package's `TimesFM_2p5_200M_torch` class). Running both in one environment
would require an uninstall/reinstall between them, so this notebook covers 1.0/2.0
and a separate `03b_timesfm_25.ipynb` covers 2.5.

**Caveat carried into the results:** per Google's own docs, "[1.0/2.0] focus on point
forecasts. We experimentally offer quantile heads but they have not been calibrated
after pretraining." Read the WQL numbers for these two checkpoints with that in mind --
TimesFM-2.5's quantile head is the first one Google describes as actually calibrated.

**Before running:** Runtime -> Change runtime type -> GPU.

Output: `timesfm_1_2_results.csv`, one row per (version, target, horizon, window).
"""))

cells.append(code("""\
!pip install -q "timesfm[torch]==1.3.0" yfinance
"""))

cells.append(code("""\
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import timesfm

print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
BACKEND = "gpu" if torch.cuda.is_available() else "cpu"
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
        if isinstance(close, pd.DataFrame):
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

cells.append(md("## TimesFM 1.0/2.0 registry + cached loading"))

cells.append(code("""\
TIMESFM_REGISTRY = [
    {"version": "1.0", "checkpoint": "google/timesfm-1.0-200m-pytorch", "num_layers": None},
    {"version": "2.0", "checkpoint": "google/timesfm-2.0-500m-pytorch", "num_layers": 50},
]

_tfm_cache = {}


def get_tfm(checkpoint, num_layers, horizon):
    key = (checkpoint, horizon)
    if key in _tfm_cache:
        return _tfm_cache[key]
    hparams_kwargs = dict(
        backend=BACKEND,
        per_core_batch_size=32,
        horizon_len=horizon,
        context_len=CONTEXT_LENGTH,
        use_positional_embedding=False,
    )
    if num_layers is not None:
        hparams_kwargs["num_layers"] = num_layers
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(**hparams_kwargs),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
    )
    _tfm_cache[key] = tfm
    return tfm
"""))

cells.append(md("""\
## Forecast call

**Sanity check first:** we're not certain of the exact column/axis convention of
`experimental_quantile_forecast` for this older package version (the 2.5 docs describe
"mean, then 10th to 90th quantiles" on axis -1 of size 10 -- we're assuming the same
layout here). Run the probe cell below once and confirm the shape is `(1, horizon, 10)`
before trusting the quantile indexing in the sweep.
"""))

cells.append(code("""\
_ctx0, _ = build_window(df, ORIGINS[0], CONTEXT_LENGTH, HORIZONS[0])
_tfm0 = get_tfm(TIMESFM_REGISTRY[0]["checkpoint"], TIMESFM_REGISTRY[0]["num_layers"], HORIZONS[0])
_point, _quantile = _tfm0.forecast([_ctx0["log_return"].to_numpy()], freq=[0])
print("point_forecast shape:", np.asarray(_point).shape)
print("quantile_forecast shape:", np.asarray(_quantile).shape)
"""))

cells.append(code("""\
def timesfm_forecast(checkpoint, num_layers, horizon, target_values):
    tfm = get_tfm(checkpoint, num_layers, horizon)
    point_forecast, quantile_forecast = tfm.forecast([target_values], freq=[0])
    point_forecast = np.asarray(point_forecast)[0]  # (horizon,)
    quantile_forecast = np.asarray(quantile_forecast)[0]  # (horizon, 10): mean, q0.1..q0.9
    quantile_forecasts = {q: quantile_forecast[:, i + 1] for i, q in enumerate(QUANTILE_LEVELS)}
    return point_forecast, quantile_forecasts
"""))

cells.append(md("## Main sweep: every (version, target, horizon, window)"))

cells.append(code("""\
rows = []
for spec in TIMESFM_REGISTRY:
    for horizon in HORIZONS:
        for target in TARGETS:
            for window_id, origin in enumerate(ORIGINS):
                context_df, future_df = build_window(df, origin, CONTEXT_LENGTH, horizon)
                target_values = context_df[target].to_numpy()
                actual = future_df[target].to_numpy()

                point_forecast, quantile_forecasts = timesfm_forecast(
                    spec["checkpoint"], spec["num_layers"], horizon, target_values
                )
                m = compute_window_metrics(actual, point_forecast, quantile_forecasts, target_values)
                rows.append({
                    "family": "timesfm",
                    "version": spec["version"],
                    "checkpoint": spec["checkpoint"],
                    "variant": "uni",
                    "target": target,
                    "horizon": horizon,
                    "window_id": window_id,
                    "origin_date": str(df.index[origin].date()),
                    **m,
                })
    print(f"done: {spec['checkpoint']}")

results_df = pd.DataFrame(rows)
results_df.to_csv("timesfm_1_2_results.csv", index=False)
print(f"\\nWrote timesfm_1_2_results.csv ({len(results_df)} rows)")
results_df.groupby(["version", "target", "horizon"])[["mase", "smape", "wql"]].mean()
"""))

cells.append(md("## Download results"))

cells.append(code("""\
from google.colab import files
files.download("timesfm_1_2_results.csv")
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "03a_timesfm_1_2.ipynb"))
