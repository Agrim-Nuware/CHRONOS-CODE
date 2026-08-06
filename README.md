# S&P 500 Zero-Shot Forecasting: Moirai vs. Chronos vs. TimesFM

Zero-shot (no fine-tuning) accuracy comparison of 9 checkpoints across the Moirai,
Chronos, and TimesFM foundation-model families, on a rolling backtest of the S&P 500.

## What's being compared

- **9 checkpoints**: Moirai-1.0-R-large, Moirai-MoE-1.0-R-base, Moirai-2.0-R-small,
  chronos-t5-large, chronos-bolt-base, chronos-2, timesfm-1.0-200m, timesfm-2.0-500m,
  timesfm-2.5-200m.
- **3 targets**: S&P 500 close price level, simple return, log return.
- **2 horizons**: 5 trading days, 21 trading days.
- **60 rolling-origin backtest windows** spanning 2005-2026 (same origins reused
  across both horizons), 512-timestep (~2yr) context.
- **Covariates** (VIX + 10yr Treasury yield, past-only): only on Moirai-1.0,
  Moirai-MoE, and Chronos-2 -- the only checkpoints that support genuinely
  zero-shot-honest covariates with these specific variables. See "Decisions and
  caveats" below for why the other 6 don't get a covariate variant.
- **Metrics**: MASE, sMAPE, WQL (mean weighted quantile loss), computed per window
  then averaged.

## Why Colab, and why 4 notebooks instead of 1

This machine has no GPU (integrated Intel graphics only), so the actual model
inference runs on **Google Colab** with a GPU runtime. Everything data/metrics-related
was built and tested locally first (see `harness/`); the Colab notebooks inline that
same logic so each one is self-contained and needs no upload besides itself.

Four Colab notebooks instead of one because the three families need incompatible
pip environments (different torch versions, and TimesFM 1.0/2.0 vs. 2.5 are
literally different, mutually-incompatible package versions under the same
`timesfm` import name):

| Notebook | Covers | Output |
|---|---|---|
| `01_moirai_family.ipynb` | Moirai-1.0, Moirai-MoE, Moirai-2.0 | `moirai_results.csv` |
| `02_chronos_family.ipynb` | Chronos (original), Chronos-Bolt, Chronos-2 | `chronos_results.csv` |
| `03a_timesfm_1_2.ipynb` | TimesFM-1.0, TimesFM-2.0 | `timesfm_1_2_results.csv` |
| `03b_timesfm_25.ipynb` | TimesFM-2.5 | `timesfm_25_results.csv` |

## How to run

1. Upload each of the 4 notebooks to [Google Colab](https://colab.research.google.com)
   (File -> Upload notebook), one at a time.
2. For each: **Runtime -> Change runtime type -> T4 GPU** (or better), then
   **Runtime -> Run all**.
3. Each notebook ends with a `files.download(...)` call that pushes its results CSV
   to your browser's downloads. Save each one into this project's `results/` folder,
   keeping the exact filename.
4. Once all 4 CSVs are in `results/`, open `notebooks/04_compare_results.ipynb`
   **locally** (this one needs no GPU, just pandas/matplotlib) and run it top to
   bottom. It writes `results/summary_table.csv` and PNG charts to
   `results/figures/` -- those are the mentor-facing deliverables.

Runtime expectations: Moirai and Chronos-Bolt/Chronos-2 should be quick per-window;
**original Chronos (`chronos-t5-large`) is the slowest** since it samples forecast
trajectories autoregressively rather than using a direct quantile head -- expect that
section of `02_chronos_family.ipynb` to dominate the notebook's total runtime.

## Two cells flagged as "verify before trusting"

I could not execute the Chronos-2 or TimesFM APIs myself (no local GPU, and the
checkpoints are large) -- everything was built from the official docs/source, but
two spots carry residual risk and are called out with a probe cell right before the
real sweep starts, specifically so a mismatch fails in seconds, not after an hour of
compute:

- **`02_chronos_family.ipynb`**: `predict_df`'s quantile output column naming
  (assumed to be named after the quantile level, e.g. `"0.5"`).
- **`03a_timesfm_1_2.ipynb`**: whether the older TimesFM package's
  `experimental_quantile_forecast` follows the same "(mean, then deciles 0.1-0.9)"
  layout that the 2.5 docs describe for the newer package.

If either probe cell's printed shape/columns don't match what the next cell assumes,
that's a one-line indexing fix, not a redesign.

## Decisions and caveats worth remembering when presenting this

- **Model selection**: one checkpoint per model *version* (not every size tier), the
  largest available for that version, e.g. `-large` where it exists, the only
  available size where it doesn't (Moirai-2.0, all of TimesFM). This means checkpoint
  sizes aren't matched across models -- the comparison is "best foot forward per
  version," not "controlled for parameter count."
- **Covariates dropped for TimesFM entirely**: its XReg mechanism requires covariate
  values across the *entire* forecast horizon, not just history. VIX/TNX aren't
  knowable in advance, so faking future values would leak information. Moirai-2.0 is
  also univariate-only, but for a different reason -- the uni2ts authors' own code
  comments flag the multivariate/covariate path as untested for that version, and the
  official example notebook doesn't demonstrate it.
- **sMAPE is weak on return-based targets**: values near zero make sMAPE unstable.
  MASE and WQL are the more trustworthy numbers for the `simple_return`/`log_return`
  targets; sMAPE is really only solid for the `price` target.
- **TimesFM 1.0/2.0's quantile heads are explicitly uncalibrated** per Google's own
  docs ("experimentally offer quantile heads but they have not been calibrated after
  pretraining") -- their WQL numbers should be read with that caveat. TimesFM-2.5's
  quantile head is the first one Google describes as actually calibrated.
- **Regime breakdown**: `04_compare_results.ipynb` splits windows into calm/high-vol
  by a median VIX-at-origin split -- useful for pre-empting "is this just because most
  windows were a calm bull market?" from a mentor.

## Project layout

```
harness/            local-only: data fetch, windowing, metrics (tested against real yfinance data)
notebooks/
  01-03b*.ipynb      upload these to Colab
  04_compare_results.ipynb   run this locally after downloading the 4 result CSVs
  build_*.py         regenerate the .ipynb files from these if you need to tweak them
results/            put the 4 downloaded CSVs here; summary_table.csv + figures/ land here after step 4
```
