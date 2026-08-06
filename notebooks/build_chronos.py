import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import write_notebook, md, code

cells = []

cells.append(md("""\
# S&P 500 Zero-Shot Forecasting — Chronos Family

Runs **chronos-t5-large** (original), **chronos-bolt-base**, and **chronos-2** zero-shot
(no fine-tuning) on a rolling backtest of the S&P 500, across 3 targets (price level,
simple return, log return) and 2 horizons (5-day, 21-day).

Only **Chronos-2** gets a covariate variant (VIX + 10yr Treasury yield as past-only
context, no future-known values passed) -- the original Chronos and Chronos-Bolt
architectures are purely univariate by design.

Original Chronos does autoregressive sampling, which is meaningfully slower per call
than Bolt or Chronos-2's direct quantile heads -- expect it to be the slowest section.

**Before running:** Runtime -> Change runtime type -> GPU.

Output: `chronos_results.csv`, one row per (version, variant, target, horizon, window).
"""))

cells.append(code("""\
!pip install -q "chronos-forecasting>=1.5" yfinance
"""))

cells.append(code("""\
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from chronos import BaseChronosPipeline, Chronos2Pipeline

print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
DEVICE_MAP = "cuda" if torch.cuda.is_available() else "cpu"
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

cells.append(md("## Chronos model registry + cached pipeline loading"))

cells.append(code("""\
CHRONOS_REGISTRY = [
    {"version": "original", "checkpoint": "amazon/chronos-t5-large", "cls": "base", "variants": ["uni"]},
    {"version": "bolt", "checkpoint": "amazon/chronos-bolt-base", "cls": "base", "variants": ["uni"]},
    {"version": "2", "checkpoint": "amazon/chronos-2", "cls": "chronos2", "variants": ["uni", "cov"]},
]

_pipeline_cache = {}


def get_pipeline(cls_name, checkpoint):
    if checkpoint not in _pipeline_cache:
        if cls_name == "base":
            _pipeline_cache[checkpoint] = BaseChronosPipeline.from_pretrained(
                checkpoint, device_map=DEVICE_MAP, torch_dtype=torch.bfloat16
            )
        elif cls_name == "chronos2":
            _pipeline_cache[checkpoint] = Chronos2Pipeline.from_pretrained(
                checkpoint, device_map=DEVICE_MAP
            )
        else:
            raise ValueError(cls_name)
    return _pipeline_cache[checkpoint]
"""))

cells.append(md("## Forecast call"))

cells.append(code("""\
def base_chronos_forecast(checkpoint, horizon, target_values):
    pipeline = get_pipeline("base", checkpoint)
    quantiles, _mean = pipeline.predict_quantiles(
        context=torch.tensor(target_values, dtype=torch.float32),
        prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
    )
    quantiles = quantiles[0].detach().cpu().numpy()  # (horizon, num_quantiles)
    quantile_forecasts = {q: quantiles[:, i] for i, q in enumerate(QUANTILE_LEVELS)}
    point_forecast = quantile_forecasts[0.5]
    return point_forecast, quantile_forecasts


def chronos2_forecast(checkpoint, horizon, target, context_df, cov=False):
    pipeline = get_pipeline("chronos2", checkpoint)
    context_df = context_df.reset_index()
    context_df = context_df.rename(columns={context_df.columns[0]: "timestamp"})
    context_df["id"] = "SPX"
    cols = ["id", "timestamp", target] + (["vix", "tnx"] if cov else [])
    pred_df = pipeline.predict_df(
        context_df[cols],
        future_df=None,
        prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
        id_column="id",
        timestamp_column="timestamp",
        target=target,
        # Real trading-day data has irregular holiday gaps that break pandas'
        # frequency auto-detection (pd.infer_freq wants a perfectly regular step) --
        # "B" (business day) is the standard convention for market data and avoids
        # relying on inference at all. Only affects the timestamps predict_df
        # generates internally, not the forecast values we actually score.
        freq="B",
    )
    quantile_forecasts = {q: pred_df[str(q)].to_numpy() for q in QUANTILE_LEVELS}
    point_forecast = quantile_forecasts[0.5]
    return point_forecast, quantile_forecasts
"""))

cells.append(md("""\
**Note on the `chronos2_forecast` quantile column names:** `predict_df` is expected to
name quantile columns after the requested levels (e.g. `"0.1"`, `"0.5"`, ...). If Chronos-2's
actual output columns differ (e.g. a `"mean"` column, or float-formatted names), the cell
below will raise a `KeyError` -- print `pred_df.columns` once and adjust `str(q)` accordingly
before running the full sweep. We could not execute this against the real checkpoint ourselves,
so treat this cell as the one most likely to need a small fix on first run.
"""))

cells.append(code("""\
# Sanity check: run once on the first window before the full sweep, so a column-name
# mismatch fails fast instead of after an hour of compute.
_ctx0, _ = build_window(df, ORIGINS[0], CONTEXT_LENGTH, HORIZONS[0])
_pipeline = get_pipeline("chronos2", "amazon/chronos-2")
_probe_df = _ctx0.reset_index()
_probe_df = _probe_df.rename(columns={_probe_df.columns[0]: "timestamp"})
_probe_df["id"] = "SPX"
_probe_pred = _pipeline.predict_df(
    _probe_df[["id", "timestamp", "log_return"]],
    future_df=None,
    prediction_length=HORIZONS[0],
    quantile_levels=QUANTILE_LEVELS,
    id_column="id",
    timestamp_column="timestamp",
    target="log_return",
    freq="B",
)
print(_probe_pred.columns.tolist())
_probe_pred.head()
"""))

cells.append(md("## Main sweep: every (version, variant, target, horizon, window)"))

cells.append(code("""\
rows = []
for spec in CHRONOS_REGISTRY:
    for variant in spec["variants"]:
        for horizon in HORIZONS:
            for target in TARGETS:
                for window_id, origin in enumerate(ORIGINS):
                    context_df, future_df = build_window(df, origin, CONTEXT_LENGTH, horizon)
                    target_values = context_df[target].to_numpy()
                    actual = future_df[target].to_numpy()

                    if spec["cls"] == "base":
                        point_forecast, quantile_forecasts = base_chronos_forecast(
                            spec["checkpoint"], horizon, target_values
                        )
                    else:
                        point_forecast, quantile_forecasts = chronos2_forecast(
                            spec["checkpoint"], horizon, target, context_df, cov=(variant == "cov")
                        )

                    m = compute_window_metrics(actual, point_forecast, quantile_forecasts, target_values)
                    rows.append({
                        "family": "chronos",
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
results_df.to_csv("chronos_results.csv", index=False)
print(f"\\nWrote chronos_results.csv ({len(results_df)} rows)")
results_df.groupby(["version", "variant", "target", "horizon"])[["mase", "smape", "wql"]].mean()
"""))

cells.append(md("## Download results"))

cells.append(code("""\
from google.colab import files
files.download("chronos_results.csv")
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_chronos_family.ipynb"))
