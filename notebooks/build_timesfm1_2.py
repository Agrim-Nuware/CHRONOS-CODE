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

cells.append(md("""\
**Why this clones the repo instead of `pip install timesfm==1.3.0`:** every PyPI
release of the old API (up to and including 1.3.0) declares `requires-python <3.12`,
and Colab runs Python 3.12 -- pip refuses to install any of them here at all, full
stop, no version pin fixes that. But that's a packaging-metadata restriction, not a
real code incompatibility: checking the actual source (`v1/src/timesfm/timesfm_torch.py`
and `timesfm_base.py`), the torch backend only genuinely imports numpy, pandas, torch,
`huggingface_hub`, and `utilsforecast` -- nothing that's actually Python-3.12-incompatible.
Cloning the same source PyPI would have installed and adding it to `sys.path` directly
sidesteps pip's version gate entirely.
"""))

cells.append(code("""\
!rm -rf timesfm_v1_repo
!git clone --depth 1 https://github.com/google-research/timesfm.git timesfm_v1_repo
!ls timesfm_v1_repo/v1/src/timesfm   # sanity check -- should list timesfm_base.py, timesfm_torch.py, ...
!pip install -q "huggingface_hub[cli]>=0.23.0" "utilsforecast>=0.1.10" yfinance
"""))

cells.append(md("""\
**Why `TimesFmTorch` is imported directly instead of `timesfm.TimesFm`:** the
package's `__init__.py` tries the JAX backend first and only falls back to torch if
JAX isn't importable -- and Colab often ships JAX preinstalled by default, which would
silently pick the wrong backend for our PyTorch checkpoints. Importing the torch class
by name sidesteps that guesswork regardless of what else happens to be on this machine.
"""))

cells.append(code("""\
import os
import sys

sys.path.insert(0, os.path.abspath("timesfm_v1_repo/v1/src"))

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from timesfm.timesfm_base import TimesFmCheckpoint, TimesFmHparams
from timesfm.timesfm_torch import TimesFmTorch as TimesFm
import timesfm as _timesfm_pkg
print("timesfm loaded from:", _timesfm_pkg.__file__)

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
# num_layers and use_positional_embedding are architecture-specific, not just size
# knobs -- 1.0 (200M) and 2.0 (500M) are genuinely different architectures under the
# same old API. TimesFmHparams' own defaults (num_layers=20, use_positional_embedding=
# True) already match 1.0; the official example only overrides both for 2.0. An
# earlier version of this notebook applied 2.0's overrides to both checkpoints, which
# doesn't crash -- it just silently feeds 1.0 the wrong position-encoding scheme and
# produces badly degraded (but not obviously broken-looking) forecasts.
TIMESFM_REGISTRY = [
    {"version": "1.0", "checkpoint": "google/timesfm-1.0-200m-pytorch", "num_layers": None, "use_positional_embedding": True},
    {"version": "2.0", "checkpoint": "google/timesfm-2.0-500m-pytorch", "num_layers": 50, "use_positional_embedding": False},
]

_tfm_cache = {}


def get_tfm(checkpoint, num_layers, use_positional_embedding, horizon):
    key = (checkpoint, horizon)
    if key in _tfm_cache:
        return _tfm_cache[key]
    hparams_kwargs = dict(
        backend=BACKEND,
        per_core_batch_size=32,
        horizon_len=horizon,
        context_len=CONTEXT_LENGTH,
        use_positional_embedding=use_positional_embedding,
    )
    if num_layers is not None:
        hparams_kwargs["num_layers"] = num_layers
    tfm = TimesFm(
        hparams=TimesFmHparams(**hparams_kwargs),
        checkpoint=TimesFmCheckpoint(huggingface_repo_id=checkpoint),
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
_spec0 = TIMESFM_REGISTRY[0]
_ctx0, _ = build_window(df, ORIGINS[0], CONTEXT_LENGTH, HORIZONS[0])
_tfm0 = get_tfm(_spec0["checkpoint"], _spec0["num_layers"], _spec0["use_positional_embedding"], HORIZONS[0])
_point, _quantile = _tfm0.forecast([_ctx0["log_return"].to_numpy()], freq=[0])
print("point_forecast shape:", np.asarray(_point).shape)
print("quantile_forecast shape:", np.asarray(_quantile).shape)
"""))

cells.append(code("""\
def timesfm_forecast(checkpoint, num_layers, use_positional_embedding, horizon, target_values):
    tfm = get_tfm(checkpoint, num_layers, use_positional_embedding, horizon)
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
                    spec["checkpoint"], spec["num_layers"], spec["use_positional_embedding"],
                    horizon, target_values,
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

cells.append(md("""\
## Example-window trajectories (for illustrative charts)

The sweep above only kept metrics per window, not the actual forecast values --
too much data to carry for 60 windows x 3 targets x 2 checkpoints. This captures the
full history/actual/naive/forecast trajectory for just **one** window (the most
recent one, `ORIGINS[-1]`), reusing the already-loaded models from the sweep above
(fast -- no re-downloading).
"""))

cells.append(code("""\
EXAMPLE_ORIGIN = ORIGINS[-1]
EXAMPLE_HORIZON = 21

example_rows = []
for spec in TIMESFM_REGISTRY:
    for target in TARGETS:
        context_df, future_df = build_window(df, EXAMPLE_ORIGIN, CONTEXT_LENGTH, EXAMPLE_HORIZON)
        target_values = context_df[target].to_numpy()
        actual = future_df[target].to_numpy()
        point_forecast, _ = timesfm_forecast(
            spec["checkpoint"], spec["num_layers"], spec["use_positional_embedding"],
            EXAMPLE_HORIZON, target_values,
        )
        naive = np.full(EXAMPLE_HORIZON, target_values[-1])

        series = [("history", context_df.index, target_values), ("actual", future_df.index, actual),
                  ("naive", future_df.index, naive), ("forecast", future_df.index, point_forecast)]
        for series_name, dates, values in series:
            for date, value in zip(dates, values):
                example_rows.append({
                    "family": "timesfm", "version": spec["version"], "checkpoint": spec["checkpoint"],
                    "target": target, "series": series_name, "date": str(date.date()), "value": float(value),
                })
    print(f"example window done: {spec['checkpoint']}")

example_df = pd.DataFrame(example_rows)
example_df.to_csv("timesfm_1_2_example_windows.csv", index=False)
print(f"Wrote timesfm_1_2_example_windows.csv ({len(example_df)} rows)")
"""))

cells.append(md("## Download results"))

cells.append(code("""\
from google.colab import files
files.download("timesfm_1_2_results.csv")
files.download("timesfm_1_2_example_windows.csv")
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "03a_timesfm_1_2.ipynb"))
