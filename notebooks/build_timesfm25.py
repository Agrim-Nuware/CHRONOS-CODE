import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import write_notebook, md, code

cells = []

cells.append(md("""\
# S&P 500 Zero-Shot Forecasting — TimesFM 2.5

Runs **timesfm-2.5-200m** zero-shot (no fine-tuning) on a rolling backtest of the
S&P 500, across 3 targets (price level, simple return, log return) and 2 horizons
(5-day, 21-day). Univariate only.

Separate notebook from TimesFM 1.0/2.0 because 2.5 ships as a different, incompatible
pip package version with a different API (`TimesFM_2p5_200M_torch` + `ForecastConfig`
vs. the older `TimesFm`/`TimesFmHparams`) -- see `03a_timesfm_1_2.ipynb` for those two.

**Before running:** Runtime -> Change runtime type -> GPU.

Output: `timesfm_25_results.csv`, one row per (target, horizon, window).
"""))

cells.append(code("""\
!pip install -q "timesfm[torch]" yfinance
"""))

cells.append(code("""\
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import timesfm

torch.set_float32_matmul_precision("high")
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
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

cells.append(md("""\
## Load TimesFM 2.5

One model instance covers both horizons: `max_horizon` just sets the compiled upper
bound, `.forecast(horizon=...)` can request anything up to it.
"""))

cells.append(code("""\
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=CONTEXT_LENGTH,
        max_horizon=max(HORIZONS),
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
"""))

cells.append(md("""\
**Sanity check first:** confirm the quantile axis convention before the full sweep --
docs say `quantile_forecast.shape == (batch, horizon, 10)`: mean, then deciles 0.1-0.9.
"""))

cells.append(code("""\
_ctx0, _ = build_window(df, ORIGINS[0], CONTEXT_LENGTH, HORIZONS[0])
_point, _quantile = model.forecast(horizon=HORIZONS[0], inputs=[_ctx0["log_return"].to_numpy()])
print("point_forecast shape:", np.asarray(_point).shape)
print("quantile_forecast shape:", np.asarray(_quantile).shape)
"""))

cells.append(code("""\
def timesfm25_forecast(horizon, target_values):
    point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[target_values])
    point_forecast = np.asarray(point_forecast)[0]  # (horizon,)
    quantile_forecast = np.asarray(quantile_forecast)[0]  # (horizon, 10): mean, q0.1..q0.9
    quantile_forecasts = {q: quantile_forecast[:, i + 1] for i, q in enumerate(QUANTILE_LEVELS)}
    return point_forecast, quantile_forecasts
"""))

cells.append(md("## Main sweep: every (target, horizon, window)"))

cells.append(code("""\
rows = []
for horizon in HORIZONS:
    for target in TARGETS:
        for window_id, origin in enumerate(ORIGINS):
            context_df, future_df = build_window(df, origin, CONTEXT_LENGTH, horizon)
            target_values = context_df[target].to_numpy()
            actual = future_df[target].to_numpy()

            point_forecast, quantile_forecasts = timesfm25_forecast(horizon, target_values)
            m = compute_window_metrics(actual, point_forecast, quantile_forecasts, target_values)
            rows.append({
                "family": "timesfm",
                "version": "2.5",
                "checkpoint": "google/timesfm-2.5-200m-pytorch",
                "variant": "uni",
                "target": target,
                "horizon": horizon,
                "window_id": window_id,
                "origin_date": str(df.index[origin].date()),
                **m,
            })
    print(f"done: horizon={horizon}")

results_df = pd.DataFrame(rows)
results_df.to_csv("timesfm_25_results.csv", index=False)
print(f"\\nWrote timesfm_25_results.csv ({len(results_df)} rows)")
results_df.groupby(["target", "horizon"])[["mase", "smape", "wql"]].mean()
"""))

cells.append(md("## Download results"))

cells.append(code("""\
from google.colab import files
files.download("timesfm_25_results.csv")
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "03b_timesfm_25.ipynb"))
