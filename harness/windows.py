"""Rolling-origin backtest window generation.

Same window origins are reused across both horizons (5-day, 21-day) so every
model is scored on literally the same forecast-origin dates regardless of
horizon -- only the widest horizon constrains how late an origin can be.
"""

import numpy as np


def make_origins(n_rows, context_length, max_horizon, num_windows):
    """Return `num_windows` evenly spaced valid origin indices (0-based, inclusive
    index of the last context timestep) across the full series.

    Valid range for an origin i: i >= context_length - 1 and i <= n_rows - max_horizon - 1.
    """
    lo = context_length - 1
    hi = n_rows - max_horizon - 1
    if hi <= lo:
        raise ValueError(
            f"Not enough rows ({n_rows}) for context_length={context_length} "
            f"and max_horizon={max_horizon}"
        )
    origins = np.linspace(lo, hi, num=num_windows, dtype=int)
    origins = np.unique(origins)  # linspace can repeat indices if num_windows > range
    return origins.tolist()


def build_window(df, origin_idx, context_length, horizon, target_col):
    """Slice one rolling window out of `df` for a given origin index.

    Returns (context_df, future_df) where context_df has `context_length` rows
    ending at origin_idx (inclusive) and future_df has `horizon` rows starting
    right after it. Both are full-column slices so covariate columns (vix, tnx)
    ride along for free.
    """
    ctx_start = origin_idx - context_length + 1
    ctx_end = origin_idx + 1  # exclusive
    fut_end = ctx_end + horizon
    context_df = df.iloc[ctx_start:ctx_end]
    future_df = df.iloc[ctx_end:fut_end]
    assert len(context_df) == context_length
    assert len(future_df) == horizon
    return context_df, future_df
