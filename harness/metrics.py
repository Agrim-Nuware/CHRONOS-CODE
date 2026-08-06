"""MASE / sMAPE / WQL implemented in plain numpy.

Deliberately has zero dependency on gluonts so this file can be pasted
unmodified into any of the three Colab notebooks (Moirai/Chronos/TimesFM),
whose environments otherwise have nothing in common.
"""

import numpy as np


def mase(actual, point_forecast, context, seasonality=1):
    """Mean Absolute Scaled Error. `context` is the in-sample series used to
    compute the naive-forecast scale (mean abs first difference)."""
    actual = np.asarray(actual, dtype=float)
    point_forecast = np.asarray(point_forecast, dtype=float)
    context = np.asarray(context, dtype=float)

    scale = np.mean(np.abs(context[seasonality:] - context[:-seasonality]))
    if scale == 0 or not np.isfinite(scale):
        return np.nan
    return float(np.mean(np.abs(actual - point_forecast)) / scale)


def smape(actual, point_forecast, eps=1e-8):
    """Symmetric MAPE, as a percentage. Guarded with eps since return-based
    targets can sit at/near zero -- treat this metric with caution for those."""
    actual = np.asarray(actual, dtype=float)
    point_forecast = np.asarray(point_forecast, dtype=float)
    denom = np.abs(actual) + np.abs(point_forecast) + eps
    return float(100.0 * np.mean(2.0 * np.abs(actual - point_forecast) / denom))


def wql(actual, quantile_forecasts):
    """Mean weighted quantile loss (a.k.a. mean scaled pinball loss), averaged
    across quantile levels. `quantile_forecasts` is {level: array-like[horizon]}.
    """
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


def compute_window_metrics(actual, point_forecast, quantile_forecasts, context, seasonality=1):
    """Convenience wrapper returning the standard (mase, smape, wql) triple for one window."""
    return {
        "mase": mase(actual, point_forecast, context, seasonality=seasonality),
        "smape": smape(actual, point_forecast),
        "wql": wql(actual, quantile_forecasts),
    }
