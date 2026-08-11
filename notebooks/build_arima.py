import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import write_notebook, md, code

cells = []

cells.append(md("""\
# S&P 500 Forecasting — ARIMA Baseline

Runs a classical **ARIMA** model on the same rolling backtest of the S&P 500 as the
9 foundation-model checkpoints, across 3 targets (price level, simple return, log
return) and 2 horizons (5-day, 21-day). Univariate only.

This is not a foundation model -- it's the classical statistical baseline the other
9 checkpoints are implicitly being compared against. Like them, it uses a **fixed
order with no per-window tuning** (no `auto_arima` search), so it's a fair,
zero-shot-equivalent comparison rather than a hand-tuned classical model.

**No GPU needed** -- ARIMA is CPU-only. You can leave the Colab runtime on its
default (no accelerator) for this one.

Output: `arima_results.csv`, one row per (target, horizon, window).
"""))

cells.append(code("""\
!pip install -q statsmodels yfinance
"""))

cells.append(code("""\
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
print("statsmodels ready")
"""))

cells.append(md("## Config (identical across all model-family notebooks)"))

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
## ARIMA order per target

`price` is a non-stationary, trending series -- it gets the `d=1` differencing term.
`simple_return`/`log_return` are already return-like (roughly stationary), so they
use `d=0` (a plain ARMA model) rather than differencing an already-differenced series.
The order itself (`p=2, q=2`) is a fixed, modest choice applied identically to every
window -- no per-window/per-series tuning (no `auto_arima`), matching the zero-shot
spirit of the other 9 checkpoints.
"""))

cells.append(code("""\
ARIMA_ORDER = {
    "price": (2, 1, 2),
    "simple_return": (2, 0, 2),
    "log_return": (2, 0, 2),
}


def fit_and_forecast(target_values, order, horizon):
    \"\"\"Fit ARIMA on target_values, forecast `horizon` steps ahead.
    Falls back to a flat naive forecast if the fit fails to converge --
    rare, but a hard crash on window 47 of 60 would waste the whole run.\"\"\"
    try:
        result = ARIMA(target_values, order=order).fit()
        forecast_obj = result.get_forecast(steps=horizon)
        mean = np.asarray(forecast_obj.predicted_mean)
        quantile_forecasts = {}
        for q in QUANTILE_LEVELS:
            if q == 0.5:
                quantile_forecasts[q] = mean
                continue
            if q < 0.5:
                alpha = 2 * q
                bound_col = 0  # lower bound
            else:
                alpha = 2 * (1 - q)
                bound_col = 1  # upper bound
            ci = np.asarray(forecast_obj.conf_int(alpha=alpha))
            quantile_forecasts[q] = ci[:, bound_col]
        return mean, quantile_forecasts
    except Exception as e:
        print(f"  ARIMA fit failed ({e}) -- falling back to naive for this window")
        naive = np.full(horizon, target_values[-1])
        return naive, {q: naive for q in QUANTILE_LEVELS}
"""))

cells.append(md("""\
**Sanity check first:** confirm `conf_int()` really returns (lower, upper) in that
column order before trusting the quantile indexing above -- this is documented
statsmodels behavior, but wasn't executed against a real environment while writing
this notebook (no local GPU/Python to test against), so it's called out explicitly
per this project's own convention for unverified assumptions.
"""))

cells.append(code("""\
_ctx0, _ = build_window(df, ORIGINS[0], CONTEXT_LENGTH, HORIZONS[0])
_probe_result = ARIMA(_ctx0["log_return"].to_numpy(), order=ARIMA_ORDER["log_return"]).fit()
_probe_forecast = _probe_result.get_forecast(steps=HORIZONS[0])
_probe_ci = _probe_forecast.conf_int(alpha=0.2)
print("conf_int columns (expect lower < upper on every row):")
print(np.asarray(_probe_ci))
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

            point_forecast, quantile_forecasts = fit_and_forecast(target_values, ARIMA_ORDER[target], horizon)
            m = compute_window_metrics(actual, point_forecast, quantile_forecasts, target_values)
            rows.append({
                "family": "arima",
                "version": "auto",
                "checkpoint": "statsmodels.tsa.arima.model.ARIMA",
                "variant": "uni",
                "target": target,
                "horizon": horizon,
                "window_id": window_id,
                "origin_date": str(df.index[origin].date()),
                **m,
            })
        print(f"done: target={target}, horizon={horizon}")

results_df = pd.DataFrame(rows)
results_df.to_csv("arima_results.csv", index=False)
print(f"\\nWrote arima_results.csv ({len(results_df)} rows)")
results_df.groupby(["target", "horizon"])[["mase", "smape", "wql"]].mean()
"""))

cells.append(md("""\
## Example-window trajectory (for illustrative charts)

The sweep above only kept metrics per window, not the actual forecast values. This
captures the full history/actual/naive/forecast trajectory for just **one** window
(the most recent one, `ORIGINS[-1]`), matching the schema the other 9 checkpoints'
notebooks write so it drops straight into `04_compare_results.ipynb`.
"""))

cells.append(code("""\
EXAMPLE_ORIGIN = ORIGINS[-1]
EXAMPLE_HORIZON = 21

example_rows = []
for target in TARGETS:
    context_df, future_df = build_window(df, EXAMPLE_ORIGIN, CONTEXT_LENGTH, EXAMPLE_HORIZON)
    target_values = context_df[target].to_numpy()
    actual = future_df[target].to_numpy()
    point_forecast, _ = fit_and_forecast(target_values, ARIMA_ORDER[target], EXAMPLE_HORIZON)
    naive = np.full(EXAMPLE_HORIZON, target_values[-1])

    series = [("history", context_df.index, target_values), ("actual", future_df.index, actual),
              ("naive", future_df.index, naive), ("forecast", future_df.index, point_forecast)]
    for series_name, dates, values in series:
        for date, value in zip(dates, values):
            example_rows.append({
                "family": "arima", "version": "auto", "checkpoint": "statsmodels.tsa.arima.model.ARIMA",
                "target": target, "series": series_name, "date": str(date.date()), "value": float(value),
            })
print("example window done: arima")

example_df = pd.DataFrame(example_rows)
example_df.to_csv("arima_example_windows.csv", index=False)
print(f"Wrote arima_example_windows.csv ({len(example_df)} rows)")
"""))

cells.append(md("## Download results"))

cells.append(code("""\
from google.colab import files
files.download("arima_results.csv")
files.download("arima_example_windows.csv")
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_arima_baseline.ipynb"))
