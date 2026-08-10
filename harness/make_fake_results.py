"""Generate synthetic result CSVs matching the real schema, purely to smoke-test
04_compare_results.ipynb's aggregation/plotting code before real Colab runs exist."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from harness import data, windows
from harness.config import CONTEXT_LENGTH, HORIZONS, NUM_WINDOWS, TARGETS

rng = np.random.default_rng(0)

df = data.load_dataset()
origins = windows.make_origins(len(df), CONTEXT_LENGTH, max(HORIZONS), NUM_WINDOWS)

REGISTRY = [
    {"family": "moirai", "version": "1.0", "variants": ["uni", "cov"]},
    {"family": "moirai", "version": "moe", "variants": ["uni", "cov"]},
    {"family": "moirai", "version": "2.0", "variants": ["uni"]},
    {"family": "chronos", "version": "original", "variants": ["uni"]},
    {"family": "chronos", "version": "bolt", "variants": ["uni"]},
    {"family": "chronos", "version": "2", "variants": ["uni", "cov"]},
    {"family": "timesfm", "version": "1.0", "variants": ["uni"]},
    {"family": "timesfm", "version": "2.0", "variants": ["uni"]},
    {"family": "timesfm", "version": "2.5", "variants": ["uni"]},
]

rows = []
for spec in REGISTRY:
    for variant in spec["variants"]:
        for horizon in HORIZONS:
            for target in TARGETS:
                for window_id, origin in enumerate(origins):
                    rows.append({
                        "family": spec["family"],
                        "version": spec["version"],
                        "checkpoint": f"fake/{spec['family']}-{spec['version']}",
                        "variant": variant,
                        "target": target,
                        "horizon": horizon,
                        "window_id": window_id,
                        "origin_date": str(df.index[origin].date()),
                        "mase": float(rng.gamma(2.0, 0.5)),
                        "smape": float(rng.gamma(2.0, 10.0)),
                        "wql": float(rng.gamma(2.0, 0.05)),
                    })

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(results_dir, exist_ok=True)

out = pd.DataFrame(rows)
for family, fname in [
    ("moirai", "moirai_results.csv"),
    ("chronos", "chronos_results.csv"),
]:
    out[out["family"] == family].to_csv(os.path.join(results_dir, fname), index=False)

timesfm_all = out[out["family"] == "timesfm"]
timesfm_all[timesfm_all["version"].isin(["1.0", "2.0"])].to_csv(
    os.path.join(results_dir, "timesfm_1_2_results.csv"), index=False
)
timesfm_all[timesfm_all["version"] == "2.5"].to_csv(
    os.path.join(results_dir, "timesfm_25_results.csv"), index=False
)
print("wrote fake result CSVs to", results_dir)

# --- Fake example-window trajectories, matching the schema each real notebook writes
# (family, version, checkpoint, target, series, date, value), so 04_compare_results.ipynb's
# 27-graph (3 targets x 9 checkpoints) example-window plotting can be smoke-tested locally
# before any real Colab run exists.
EXAMPLE_ORIGIN = origins[-1]
EXAMPLE_HORIZON = 21
CHECKPOINTS = [(spec["family"], spec["version"]) for spec in REGISTRY]

example_rows = []
for family, version in CHECKPOINTS:
    for target in TARGETS:
        context_df, future_df = windows.build_window(df, EXAMPLE_ORIGIN, CONTEXT_LENGTH, EXAMPLE_HORIZON, target)
        target_values = context_df[target].to_numpy()
        actual = future_df[target].to_numpy()
        naive = np.full(EXAMPLE_HORIZON, target_values[-1])
        noise_scale = np.std(np.diff(target_values))
        forecast = actual + rng.normal(0.0, noise_scale, size=EXAMPLE_HORIZON)

        series = [("history", context_df.index, target_values), ("actual", future_df.index, actual),
                  ("naive", future_df.index, naive), ("forecast", future_df.index, forecast)]
        for series_name, dates, values in series:
            for date, value in zip(dates, values):
                example_rows.append({
                    "family": family, "version": version, "checkpoint": f"fake/{family}-{version}",
                    "target": target, "series": series_name, "date": str(date.date()), "value": float(value),
                })

example_out = pd.DataFrame(example_rows)
for family, fname in [
    ("moirai", "moirai_example_windows.csv"),
    ("chronos", "chronos_example_windows.csv"),
]:
    example_out[example_out["family"] == family].to_csv(os.path.join(results_dir, fname), index=False)

timesfm_examples = example_out[example_out["family"] == "timesfm"]
timesfm_examples[timesfm_examples["version"].isin(["1.0", "2.0"])].to_csv(
    os.path.join(results_dir, "timesfm_1_2_example_windows.csv"), index=False
)
timesfm_examples[timesfm_examples["version"] == "2.5"].to_csv(
    os.path.join(results_dir, "timesfm_25_example_windows.csv"), index=False
)
print("wrote fake example-window CSVs to", results_dir)
