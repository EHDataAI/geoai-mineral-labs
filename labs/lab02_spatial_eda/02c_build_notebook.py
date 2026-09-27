from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
)


NOTEBOOK_NAME = "02_spatial_eda.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    cells = []

    cells.append(
        new_markdown_cell(
            """# LAB 02 - Spatial Exploratory Data Analysis

## Sampling density is not spatial signal

This laboratory examines a common problem in geospatial analysis:

> A location with many observations is not necessarily the location with the
> strongest underlying spatial signal.

The experiment uses synthetic data so that the underlying spatial process is
known. This makes it possible to compare sampling density, observed signal,
and synthetic ground truth before predictive modelling.

The workflow follows:

**CRS validation -> sampling-density analysis -> spatial aggregation ->
resolution sensitivity -> interpretation**
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## Experimental design

Each synthetic realization contains:

- 500 broadly distributed regional observations;
- 300 observations from an intensive local campaign;
- 800 observations in total.

The intensive campaign and the strongest underlying signal are deliberately
located in different parts of the study region.

All spatial calculations use **EPSG:32719 - WGS 84 / UTM zone 19S**, so
distances and grid dimensions are expressed in metres.
"""
        )
    )

    cells.append(
        new_code_cell(
            """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


def find_repo_root(start: Path) -> Path:
    current = start.resolve()

    for candidate in (current, *current.parents):
        if (candidate / "src" / "geoai").exists():
            return candidate

    raise RuntimeError(
        "Repository root containing src/geoai was not found."
    )


REPO_ROOT = find_repo_root(Path.cwd())
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geoai.spatial import (
    assign_regular_grid,
    generate_synthetic_sampling,
    nearest_neighbor_distances,
    require_metric_projected_crs,
    summarize_grid,
)

PRIMARY_CELL_SIZE_M = 10000.0
GRID_ORIGIN = (350000.0, 7100000.0)

print("Primary grid resolution:", PRIMARY_CELL_SIZE_M, "m")
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 1. Generate the controlled spatial experiment

The default seed is fixed at `12345` for the baseline walkthrough.

The synthetic dataset retains both:

- `signal_true`: the known underlying spatial process;
- `signal`: the observed value after random noise is added.

This distinction would not normally be available with real exploration data,
but it is useful here because it lets us validate the behaviour of the
exploratory workflow.
"""
        )
    )

    cells.append(
        new_code_cell(
            """samples = generate_synthetic_sampling(seed=12345)

crs = require_metric_projected_crs(samples)

print("Observations:", len(samples))
print("CRS:", crs.to_string())
print()
print(samples["sampling_component"].value_counts())
print()
print(samples.head())
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 2. Visualize the sampling pattern

The first map shows **where observations were collected**, not the magnitude of
the spatial signal.

A dense cluster can therefore represent sampling effort rather than geology.
"""
        )
    )

    cells.append(
        new_code_cell(
            """regional = samples[
    samples["sampling_component"] == "regional"
].copy()

campaign = samples[
    samples["sampling_component"] == "campaign"
].copy()

fig, ax = plt.subplots(figsize=(8, 7))

regional.plot(
    ax=ax,
    marker="o",
    markersize=18,
    alpha=0.65,
    label="Regional",
)

campaign.plot(
    ax=ax,
    marker="^",
    markersize=22,
    alpha=0.65,
    label="Intensive campaign",
)

ax.set_title("Sampling pattern")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.legend()
ax.set_aspect("equal")

plt.show()
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """### Interpretation

The intensive campaign creates an obvious cluster of observations.

If sampling density were interpreted visually without further analysis, this
area could incorrectly appear to be the most important spatial zone.

We therefore quantify sampling density rather than relying only on the map.
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 3. Nearest-neighbour sampling density

Nearest-neighbour distance provides a simple descriptive measure of how tightly
observations are spatially packed.

Shorter distances indicate denser local sampling.
"""
        )
    )

    cells.append(
        new_code_cell(
            """regional_nn = nearest_neighbor_distances(regional)
campaign_nn = nearest_neighbor_distances(campaign)

regional_mean_nn = float(regional_nn.mean())
campaign_mean_nn = float(campaign_nn.mean())
nn_ratio = campaign_mean_nn / regional_mean_nn

nn_summary = pd.DataFrame(
    {
        "Sampling component": [
            "Regional",
            "Intensive campaign",
        ],
        "Mean NN distance (m)": [
            regional_mean_nn,
            campaign_mean_nn,
        ],
    }
)

display(nn_summary.round(2))

print(
    "Campaign / regional NN ratio:",
    round(nn_ratio, 3),
)
"""
        )
    )

    cells.append(
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(7, 5))

ax.bar(
    nn_summary["Sampling component"],
    nn_summary["Mean NN distance (m)"],
)

ax.set_title("Mean nearest-neighbour distance")
ax.set_ylabel("Distance (m)")
ax.set_xlabel("Sampling component")

plt.show()
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """### Baseline result

For seed `12345`, the intensive campaign has a much smaller mean
nearest-neighbour distance than the regional observations.

The campaign/regional ratio is approximately **0.278**.

This confirms quantitatively that the campaign area is much more densely
sampled.
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 4. Visualize the observed spatial signal

The next map changes the question.

Instead of showing **where sampling is dense**, it shows the magnitude of the
observed synthetic signal.
"""
        )
    )

    cells.append(
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(8, 7))

scatter = ax.scatter(
    samples.geometry.x,
    samples.geometry.y,
    c=samples["signal"],
    s=28,
    alpha=0.8,
)

ax.set_title("Observed synthetic spatial signal")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")

colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label("Observed signal")

plt.show()
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 5. Aggregate observations using the selected grid

Grid aggregation requires a spatial-resolution decision.

LAB 02 evaluates 5 km, 10 km, and 20 km grids using repeated Monte Carlo
realizations and multiple grid origins.

The sensitivity experiment selected **10 km x 10 km** as the primary
descriptive resolution for this synthetic experiment.

This is not a universal recommendation for mineral exploration.
"""
        )
    )

    cells.append(
        new_code_cell(
            """gridded = assign_regular_grid(
    samples,
    cell_size_m=PRIMARY_CELL_SIZE_M,
    origin=GRID_ORIGIN,
)

observed_summary = summarize_grid(
    gridded,
    signal_column="signal",
)

true_summary = summarize_grid(
    gridded,
    signal_column="signal_true",
)

print("Populated cells:", len(observed_summary))
print(
    "Observations preserved:",
    int(observed_summary["observation_count"].sum()),
)
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 6. Compare sampling density and spatial signal

For the selected 10 km grid we identify:

1. the cell with the greatest number of observations;
2. the cell with the highest observed mean signal;
3. the cell with the highest known true mean signal.
"""
        )
    )

    cells.append(
        new_code_cell(
            """densest_cell = observed_summary.loc[
    observed_summary["observation_count"].idxmax()
]

observed_hotspot = observed_summary.loc[
    observed_summary["mean_signal"].idxmax()
]

true_hotspot = true_summary.loc[
    true_summary["mean_signal"].idxmax()
]

comparison = pd.DataFrame(
    [
        {
            "Feature": "Densest sampling cell",
            "Grid ID": densest_cell["grid_id"],
            "Observations": int(
                densest_cell["observation_count"]
            ),
            "Mean signal": float(
                densest_cell["mean_signal"]
            ),
        },
        {
            "Feature": "Highest observed-signal cell",
            "Grid ID": observed_hotspot["grid_id"],
            "Observations": int(
                observed_hotspot["observation_count"]
            ),
            "Mean signal": float(
                observed_hotspot["mean_signal"]
            ),
        },
        {
            "Feature": "Highest true-signal cell",
            "Grid ID": true_hotspot["grid_id"],
            "Observations": int(
                true_hotspot["observation_count"]
            ),
            "Mean signal": float(
                true_hotspot["mean_signal"]
            ),
        },
    ]
)

display(comparison.round(2))
"""
        )
    )

    cells.append(
        new_code_cell(
            """print(
    "Densest cell differs from observed hotspot:",
    densest_cell["grid_id"] != observed_hotspot["grid_id"],
)

print(
    "Densest cell differs from true hotspot:",
    densest_cell["grid_id"] != true_hotspot["grid_id"],
)

print(
    "Observed hotspot agrees with true hotspot:",
    observed_hotspot["grid_id"] == true_hotspot["grid_id"],
)
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """### Baseline interpretation

The densest sampling cell is not the strongest spatial-signal cell.

For the baseline realization, the highest observed-signal cell also agrees with
the known true-signal cell.

This illustrates the main methodological point:

> Sampling effort and spatial signal must be analysed separately.
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 7. Resolution sensitivity

The primary resolution was not chosen from the baseline map.

A separate confirmatory experiment used:

- **400 independent spatial realizations**;
- 3 grid resolutions;
- 4 grid-origin configurations;
- **4,800 grid evaluations**.

Grid origin and resolution are repeated evaluations within each realization,
not independent observations.
"""
        )
    )

    cells.append(
        new_code_cell(
            """resolution_summary = pd.DataFrame(
    {
        "Grid (km)": [5, 10, 20],
        "Mean offset match rate (%)": [
            87.4,
            98.8,
            98.9,
        ],
        "All 4 offsets matching (%)": [
            63.0,
            95.8,
            95.8,
        ],
        "Median n at true hotspot": [
            2.0,
            7.5,
            31.5,
        ],
        "Peak retention (%)": [
            94.3,
            78.8,
            45.7,
        ],
    }
)

display(resolution_summary)
"""
        )
    )

    cells.append(
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    resolution_summary["Grid (km)"],
    resolution_summary[
        "All 4 offsets matching (%)"
    ],
    marker="o",
)

ax.set_title("Hotspot stability by grid resolution")
ax.set_xlabel("Grid resolution (km)")
ax.set_ylabel("Realizations with all offsets matching (%)")
ax.set_xticks(
    resolution_summary["Grid (km)"]
)

plt.show()
"""
        )
    )

    cells.append(
        new_code_cell(
            """fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    resolution_summary["Grid (km)"],
    resolution_summary["Peak retention (%)"],
    marker="o",
)

ax.set_title("Synthetic peak retention by grid resolution")
ax.set_xlabel("Grid resolution (km)")
ax.set_ylabel("Peak retention (%)")
ax.set_xticks(
    resolution_summary["Grid (km)"]
)

plt.show()
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 8. Why 10 km was selected

The tested resolutions show a clear trade-off.

### 5 km

Preserves local signal well, but the hotspot contains only about two
observations and interpretation is sensitive to grid origin.

### 10 km

Provides high hotspot stability, better local sample support, and retains
approximately 79% of the theoretical spatial peak.

### 20 km

Provides high stability and sample support, but retains only about 46% of the
spatial peak, indicating substantial smoothing.

Therefore, **10 km is used as the primary descriptive resolution for this
synthetic experiment**.

It is not proposed as a universal grid size.
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 9. GeoAI relevance

This experiment demonstrates a problem that becomes more important when
predictive models are introduced.

A model such as Random Forest, gradient boosting, XGBoost, or a neural network
may partially learn **where observations were collected most intensively**
instead of learning the underlying geological process.

This is one reason why ordinary random train/test splitting can be misleading
for spatial data.

The next stage of the roadmap therefore introduces **spatial validation and
spatial cross-validation** before predictive mineral-prospectivity modelling.
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## 10. Limitations

This laboratory deliberately isolates one methodological problem.

It does not yet include:

- real geological observations;
- lithological or structural controls;
- real geochemical assays;
- compositional-data analysis;
- anisotropy;
- fitted spatial autocorrelation models;
- censoring or analytical detection limits;
- preferential-sampling correction;
- predictive modelling;
- mineral-prospectivity ranking.

These elements are deferred intentionally so that sampling-density and
aggregation effects are understood first.
"""
        )
    )

    cells.append(
        new_markdown_cell(
            """## Conclusion

The controlled experiment supports the main conclusion of LAB 02:

> A spatially dense cluster of samples is not automatically a hotspot of the
> underlying spatial signal.

The workflow therefore separates:

**sampling density -> spatial signal -> aggregation scale -> validation**

before moving to predictive GeoAI.

This extends the lesson from LAB 01:

> Spatial analysis can be computationally correct while still being
> scientifically misleading if spatial assumptions are not validated.
"""
        )
    )

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (geoai-mineral)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
    )

    return notebook


def main() -> None:
    output_path = (
        Path(__file__).resolve().parent
        / NOTEBOOK_NAME
    )

    notebook = build_notebook()

    nbformat.write(
        notebook,
        output_path,
    )

    print("Notebook created:")
    print(output_path)
    print("Cells:", len(notebook.cells))


if __name__ == "__main__":
    main()
