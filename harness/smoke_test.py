import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import data, windows, metrics
from harness.config import CONTEXT_LENGTH, HORIZONS, NUM_WINDOWS

print("Fetching data...")
df = data.load_dataset()
print(f"Rows: {len(df)}  Range: {df.index.min().date()} -> {df.index.max().date()}")
print(df.head())
print(df.tail())

max_h = max(HORIZONS)
origins = windows.make_origins(len(df), CONTEXT_LENGTH, max_h, NUM_WINDOWS)
print(f"\nGenerated {len(origins)} window origins (first 5): {origins[:5]}")
print(f"First origin date: {df.index[origins[0]].date()}  Last origin date: {df.index[origins[-1]].date()}")

ctx_df, fut_df = windows.build_window(df, origins[0], CONTEXT_LENGTH, HORIZONS[0], "log_return")
print(f"\nContext shape: {ctx_df.shape}  Future shape: {fut_df.shape}")

import numpy as np

actual = fut_df["log_return"].values
naive_point_forecast = np.full_like(actual, ctx_df["log_return"].values[-1])
fake_quantiles = {q: naive_point_forecast for q in [0.1, 0.5, 0.9]}

m = metrics.compute_window_metrics(
    actual, naive_point_forecast, fake_quantiles, ctx_df["log_return"].values
)
print(f"\nSanity metrics vs naive-last-value forecast: {m}")
print("\nSMOKE TEST PASSED")
