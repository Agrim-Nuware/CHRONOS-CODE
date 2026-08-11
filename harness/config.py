"""Central config shared by every model-family notebook.

Keep this identical across the Moirai / Chronos / TimesFM notebooks so the
rolling windows and metrics are computed on exactly the same basis, and the
results CSVs line up in the final comparison step.
"""

TICKERS = {
    "price": "^GSPC",   # S&P 500 index level
    "vix": "^VIX",       # CBOE Volatility Index (covariate)
    "tnx": "^TNX",        # 10-year Treasury yield (covariate)
}

DATA_START = "2005-01-01"
# DATA_END left as None -> fetch through "today" at run time.
DATA_END = None

TARGETS = ["price", "simple_return", "log_return"]

CONTEXT_LENGTH = 512      # ~2 trading years
HORIZONS = [5, 21]        # 1 trading week, ~1 trading month
NUM_WINDOWS = 60          # rolling-origin backtest windows, evenly spaced

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# One row per (family, version, checkpoint, variant) to run.
# variant is "uni" (univariate) or "cov" (VIX+TNX as past-only dynamic covariates).
MODEL_REGISTRY = [
    {"family": "moirai", "version": "1.0", "checkpoint": "Salesforce/moirai-1.0-R-large", "variants": ["uni", "cov"]},
    {"family": "moirai", "version": "moe", "checkpoint": "Salesforce/moirai-moe-1.0-R-base", "variants": ["uni", "cov"]},
    {"family": "moirai", "version": "2.0", "checkpoint": "Salesforce/moirai-2.0-R-small", "variants": ["uni"]},
    {"family": "chronos", "version": "original", "checkpoint": "amazon/chronos-t5-large", "variants": ["uni"]},
    {"family": "chronos", "version": "bolt", "checkpoint": "amazon/chronos-bolt-base", "variants": ["uni"]},
    {"family": "chronos", "version": "2", "checkpoint": "amazon/chronos-2", "variants": ["uni", "cov"]},
    # TimesFM's XReg covariate mechanism requires covariate values across the full
    # forecast horizon (not just history) -- VIX/TNX aren't knowable in advance,
    # so a "cov" variant here would leak future information. Univariate only.
    {"family": "timesfm", "version": "1.0", "checkpoint": "google/timesfm-1.0-200m-pytorch", "variants": ["uni"]},
    {"family": "timesfm", "version": "2.0", "checkpoint": "google/timesfm-2.0-500m-pytorch", "variants": ["uni"]},
    {"family": "timesfm", "version": "2.5", "checkpoint": "google/timesfm-2.5-200m-pytorch", "variants": ["uni"]},
    # Classical statistical baseline, not a foundation model -- fixed order, no
    # per-window tuning, so it's directly comparable in spirit to the zero-shot
    # checkpoints above (nobody hand-tunes a model per window here either).
    {"family": "arima", "version": "auto", "checkpoint": "statsmodels.tsa.arima.model.ARIMA", "variants": ["uni"]},
]

RESULTS_COLUMNS = [
    "family", "version", "checkpoint", "variant", "target", "horizon",
    "window_id", "origin_date", "mase", "smape", "wql",
]
