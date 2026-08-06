import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import write_notebook, md, code

cells = []

cells.append(md("""\
# S&P 500 Zero-Shot Forecasting — Results Comparison

**Runs locally** (plain pandas/matplotlib, no GPU or model libraries needed). Combines
the four results CSVs produced by the Colab notebooks
(`01_moirai_family`, `02_chronos_family`, `03a_timesfm_1_2`, `03b_timesfm_25`) into the
tables and charts for the mentor presentation.

**Before running:** download all four `*_results.csv` files from Colab and place them
in `../results/` (same names, unchanged).
"""))

cells.append(code("""\
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.getcwd()))
from harness.data import load_dataset

RESULTS_DIR = os.path.join("..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
"""))

cells.append(md("## Palette (validated categorical set — see the `dataviz` skill's reference palette)"))

cells.append(code("""\
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

FAMILY_COLOR = {"moirai": "#2a78d6", "chronos": "#eb6834", "timesfm": "#1baf7a"}
VARIANT_COLOR = {"uni": "#2a78d6", "cov": "#eb6834"}
REGIME_COLOR = {"calm": "#2a78d6", "high_vol": "#eb6834"}
VERSION_MARKER = {
    "1.0": "o", "moe": "s", "2.0": "^", "2.5": "D",
    "original": "o", "bolt": "s", "2": "^",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRIDLINE,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def model_label(row):
    names = {
        ("moirai", "1.0"): "Moirai-1.0", ("moirai", "moe"): "Moirai-MoE", ("moirai", "2.0"): "Moirai-2.0",
        ("chronos", "original"): "Chronos", ("chronos", "bolt"): "Chronos-Bolt", ("chronos", "2"): "Chronos-2",
        ("timesfm", "1.0"): "TimesFM-1.0", ("timesfm", "2.0"): "TimesFM-2.0", ("timesfm", "2.5"): "TimesFM-2.5",
    }
    return names.get((row["family"], row["version"]), f"{row['family']}-{row['version']}")
"""))

cells.append(md("## Load + combine all four result files"))

cells.append(code("""\
FILES = ["moirai_results.csv", "chronos_results.csv", "timesfm_1_2_results.csv", "timesfm_25_results.csv"]

frames = []
for fname in FILES:
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"MISSING: {path} -- download it from the corresponding Colab notebook first.")
        continue
    # dtype=str on version: a file where every version happens to look numeric
    # (e.g. TimesFM's "1.0"/"2.0"/"2.5") would otherwise get silently parsed as
    # float64, breaking the string-keyed lookup in model_label() below.
    frames.append(pd.read_csv(path, dtype={"version": str}))

all_results = pd.concat(frames, ignore_index=True)
all_results["model"] = all_results.apply(model_label, axis=1)
print(f"{len(all_results)} rows across {all_results['model'].nunique()} checkpoints")
all_results.head()
"""))

cells.append(md("## Master summary table (mean MASE / sMAPE / WQL per model x variant x target x horizon)"))

cells.append(code("""\
summary = (
    all_results
    .groupby(["family", "version", "model", "variant", "target", "horizon"])[["mase", "smape", "wql"]]
    .mean()
    .round(4)
    .reset_index()
    .sort_values(["family", "model", "variant", "target", "horizon"])
)
summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)
print(f"{len(summary)} summary rows -> {RESULTS_DIR}/summary_table.csv")
summary
"""))

cells.append(md("""\
## Headline chart: MASE across all 9 checkpoints

One target/horizon slice at a time (log return, 21-day horizon by default -- change
`TARGET`/`HORIZON` below to look at others). Bars grouped and colored by family so a
9th color is never invented for the 9th checkpoint -- version is carried by the x-axis
label instead of by hue.
"""))

cells.append(code("""\
TARGET = "log_return"
HORIZON = 21


def plot_headline(metric, target=TARGET, horizon=HORIZON, variant="uni"):
    subset = summary[(summary["target"] == target) & (summary["horizon"] == horizon) & (summary["variant"] == variant)]
    subset = subset.sort_values(["family", "model"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(subset))
    colors = [FAMILY_COLOR[f] for f in subset["family"]]
    bars = ax.bar(x, subset[metric], width=0.62, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(subset["model"], rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by checkpoint -- target={target}, horizon={horizon}d, variant={variant}")
    ax.grid(axis="y", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    family_display = {"moirai": "Moirai", "chronos": "Chronos", "timesfm": "TimesFM"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    ax.legend(handles, [family_display[f] for f in FAMILY_COLOR], frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"headline_{metric}_{target}_{horizon}d_{variant}.png"), dpi=160)
    return fig


for metric in ["mase", "smape", "wql"]:
    plot_headline(metric)
    plt.show()
"""))

cells.append(md("""\
## Horizon degradation: does accuracy fall off from 5-day to 21-day?

Color = family, marker = version within family (composite encoding instead of a
10th/11th color) -- one line per checkpoint, faceted by target so the three very
different target scales (price level vs. the two return series) don't get forced
onto one axis.
"""))

cells.append(code("""\
def plot_horizon_degradation(metric="mase", variant="uni"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, target in zip(axes, ["price", "simple_return", "log_return"]):
        subset = summary[(summary["target"] == target) & (summary["variant"] == variant)]
        for (family, version), grp in subset.groupby(["family", "version"]):
            grp = grp.sort_values("horizon")
            label = grp["model"].iloc[0]
            ax.plot(
                grp["horizon"], grp[metric],
                color=FAMILY_COLOR[family], marker=VERSION_MARKER.get(version, "o"),
                markersize=7, linewidth=1.8, label=label,
            )
        ax.set_title(target)
        ax.set_xlabel("horizon (days)")
        ax.set_xticks(sorted(subset["horizon"].unique()))
        ax.grid(axis="y", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(metric.upper())
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.suptitle(f"{metric.upper()} vs. forecast horizon ({variant} variant)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"horizon_degradation_{metric}_{variant}.png"), dpi=160, bbox_inches="tight")
    return fig


plot_horizon_degradation("mase")
plt.show()
"""))

cells.append(md("""\
## Regime breakdown: calm vs. high-volatility windows

Splits the rolling-backtest windows by the VIX level *at the forecast origin* (median
split across all windows) and compares MASE within each regime -- answers "does this
model just look good because most windows were calm bull-market stretches?"
"""))

cells.append(code("""\
df = load_dataset()
origin_vix = df["vix"].rename("origin_vix")
origin_vix.index = origin_vix.index.strftime("%Y-%m-%d")

results_with_vix = all_results.merge(
    origin_vix, left_on="origin_date", right_index=True, how="left"
)
assert results_with_vix["origin_vix"].notna().all(), "some origin_date values didn't match the fetched date index"
vix_median = results_with_vix["origin_vix"].median()
results_with_vix["regime"] = np.where(results_with_vix["origin_vix"] >= vix_median, "high_vol", "calm")
print(f"VIX median split at {vix_median:.1f}")


def plot_regime(metric="mase", target=TARGET, horizon=HORIZON, variant="uni"):
    subset = results_with_vix[
        (results_with_vix["target"] == target)
        & (results_with_vix["horizon"] == horizon)
        & (results_with_vix["variant"] == variant)
    ]
    pivot = subset.groupby(["model", "regime"])[metric].mean().unstack("regime")
    pivot = pivot.reindex(columns=["calm", "high_vol"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(pivot))
    width = 0.36
    ax.bar(x - width / 2, pivot["calm"], width=width, color=REGIME_COLOR["calm"], label="Calm (VIX below median)")
    ax.bar(x + width / 2, pivot["high_vol"], width=width, color=REGIME_COLOR["high_vol"], label="High-vol (VIX above median)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by market regime -- target={target}, horizon={horizon}d")
    ax.grid(axis="y", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"regime_{metric}_{target}_{horizon}d.png"), dpi=160)
    return fig


plot_regime("mase")
plt.show()
"""))

cells.append(md("""\
## Covariate lift: does adding VIX/TNX actually help?

Only Moirai-1.0, Moirai-MoE, and Chronos-2 have both a `uni` and `cov` variant to
compare (see the covariate-support discussion earlier -- the rest either don't support
covariates or couldn't do so honestly with VIX/TNX specifically).
"""))

cells.append(code("""\
def plot_covariate_lift(metric="mase", target=TARGET, horizon=HORIZON):
    subset = summary[(summary["target"] == target) & (summary["horizon"] == horizon)]
    pivot = subset.pivot_table(index="model", columns="variant", values=metric)
    pivot = pivot.dropna(subset=["cov"])  # only checkpoints that actually ran a cov variant
    if pivot.empty:
        print("No checkpoints with both variants for this target/horizon.")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(pivot))
    width = 0.36
    ax.bar(x - width / 2, pivot["uni"], width=width, color=VARIANT_COLOR["uni"], label="Univariate")
    ax.bar(x + width / 2, pivot["cov"], width=width, color=VARIANT_COLOR["cov"], label="+ VIX/TNX covariates")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Covariate lift -- {metric.upper()}, target={target}, horizon={horizon}d")
    ax.grid(axis="y", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"covariate_lift_{metric}_{target}_{horizon}d.png"), dpi=160)
    return fig


plot_covariate_lift("mase")
plt.show()
"""))

cells.append(md("""\
## Example-window charts: one figure per model

For each of the 9 checkpoints, one figure with 3 panels (price / simple return / log
return) showing that model's zero-shot forecast against the actual outcome and a
naive persistence baseline, for the single most recent backtest window (21-day
horizon). Requires the 4 `*_example_windows.csv` files downloaded from the Colab
notebooks alongside the main results CSVs.
"""))

cells.append(code("""\
EXAMPLE_FILES = [
    "moirai_example_windows.csv", "chronos_example_windows.csv",
    "timesfm_1_2_example_windows.csv", "timesfm_25_example_windows.csv",
]

example_frames = []
for fname in EXAMPLE_FILES:
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"MISSING: {path} -- download it from the corresponding Colab notebook first.")
        continue
    example_frames.append(pd.read_csv(path, dtype={"version": str}))

example_all = pd.concat(example_frames, ignore_index=True)
example_all["date"] = pd.to_datetime(example_all["date"])
example_all["model"] = example_all.apply(model_label, axis=1)
print(f"{len(example_all)} rows across {example_all['model'].nunique()} checkpoints")
"""))

cells.append(code("""\
SERIES_STYLE = {
    "history": dict(color=INK, linestyle="-", linewidth=1.4, label="History (Close)"),
    "actual": dict(color=INK, linestyle="--", linewidth=1.8, label="Actual"),
    "naive": dict(color=INK_MUTED, linestyle="-.", linewidth=1.4, label="Naive (persistence)"),
}
TARGET_TITLE = {"price": "Price level", "simple_return": "Simple return", "log_return": "Log return"}

# Context is 512 days but the horizon is only 21 -- plotting the full history makes
# the forecast/actual/naive comparison a barely-visible sliver at the right edge.
# Showing only the recent tail keeps proportions readable, closer to a typical
# history:horizon ratio of ~3:1 rather than ~24:1.
HISTORY_TAIL_DAYS = 60


def plot_example_window(model_name, family, rows_for_model):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, target in zip(axes, ["price", "simple_return", "log_return"]):
        sub = rows_for_model[rows_for_model["target"] == target]
        for series_name, style in SERIES_STYLE.items():
            s = sub[sub["series"] == series_name].sort_values("date")
            if series_name == "history":
                s = s.tail(HISTORY_TAIL_DAYS)
            ax.plot(s["date"], s["value"], **style)
        fc = sub[sub["series"] == "forecast"].sort_values("date")
        ax.plot(fc["date"], fc["value"], color=FAMILY_COLOR[family], linewidth=2.2,
                 label=f"{model_name} (zero-shot)")
        ax.set_title(TARGET_TITLE[target])
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False)
    fig.suptitle(f"{model_name} -- example window (most recent, 21-day horizon)", y=1.2)
    fig.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join(FIGURES_DIR, f"example_window_{safe_name}.png"), dpi=160, bbox_inches="tight")
    return fig


for (family, model_name), rows_for_model in example_all.groupby(["family", "model"]):
    plot_example_window(model_name, family, rows_for_model)
    plt.show()
"""))

cells.append(md("""\
## All figures saved to `../results/figures/`

`summary_table.csv` in `../results/` has the full numeric table behind every chart --
useful for a slide appendix or if a mentor asks for a number not shown on a chart.
"""))

cells.append(code("""\
print("Saved figures:")
for fname in sorted(os.listdir(FIGURES_DIR)):
    print(" -", fname)
"""))

write_notebook(cells, os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_compare_results.ipynb"))
