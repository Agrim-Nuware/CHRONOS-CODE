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
