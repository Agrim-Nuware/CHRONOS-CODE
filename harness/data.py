"""Fetch and align S&P 500 / VIX / TNX daily data.

Self-contained on purpose (only needs yfinance + pandas + numpy) so this
same code can be pasted as-is into any of the three Colab notebooks without
pulling in torch/jax as a side effect.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_raw(start=None, end=None, tickers=None):
    """Download raw daily Close prices for price/vix/tnx, aligned on shared trading days."""
    from .config import TICKERS, DATA_START, DATA_END

    start = start or DATA_START
    end = end or DATA_END
    tickers = tickers or TICKERS

    frames = {}
    for name, symbol in tickers.items():
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {symbol}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # MultiIndex columns even for a single ticker on newer yfinance
            close = close.iloc[:, 0]
        frames[name] = close.rename(name)

    combined = pd.concat(frames.values(), axis=1, join="inner")
    combined.columns = list(frames.keys())
    combined = combined.sort_index().dropna()
    return combined


def build_targets(raw):
    """Add simple_return and log_return columns derived from price. First row (no prior day) is dropped."""
    df = raw.copy()
    df["simple_return"] = df["price"].pct_change()
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df = df.dropna()
    return df


def load_dataset(start=None, end=None):
    """One-call entry point: returns a DataFrame indexed by date with columns
    [price, vix, tnx, simple_return, log_return]."""
    raw = fetch_raw(start=start, end=end)
    return build_targets(raw)
