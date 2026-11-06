"""Plotting helpers shared by the notebooks. Every plot uses the same style."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

from src.data_loader import project_path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PRIMARY = "#2a78d6"
REFERENCE = "#898781"

# Each segment keeps one color in every figure, for both algorithms.
SEGMENT_COLORS = {
    "Heavy Listeners": "#2a78d6",
    "Frequent Skippers": "#eb6834",
    "Playlist Builders": "#1baf7a",
    "Low-Rating Users": "#eda100",
    "Light Listeners": "#e87ba4",
    "Low-Skip Listeners": "#008300",
}

# Marker shapes back up color, so segments stay distinguishable in a single scatter plot.
SEGMENT_MARKERS = {
    "Heavy Listeners": "o",
    "Frequent Skippers": "s",
    "Playlist Builders": "^",
    "Low-Rating Users": "D",
    "Light Listeners": "v",
    "Low-Skip Listeners": "P",
}

# Diverging (blue to red through neutral gray) and sequential (one blue) color maps.
DIVERGING_CMAP = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#d03b3b"])
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("blue_ramp", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])

DISCRETE_VALUE_LIMIT = 60  # Columns with few distinct values are plotted value by value.

FEATURE_LABELS = {
    "avg_listening_hours_per_week": "Listening hours per week",
    "playlists_created": "Playlists created",
    "avg_skips_per_day": "Skips per day",
    "music_suggestion_rating_1_to_5": "Suggestion rating (1-5)",
    "age": "Age (years)",
    "months_inactive": "Months inactive",
    "tenure_months": "Tenure (months)",
}

METRIC_LABELS = {
    "silhouette": "Silhouette (higher is better)",
    "davies_bouldin": "Davies-Bouldin (lower is better)",
    "calinski_harabasz": "Calinski-Harabasz (higher is better)",
}


def apply_plot_style() -> None:
    """Apply one quiet, readable style to all figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def save_figure(fig: plt.Figure, name: str, config: dict) -> None:
    """Save a figure as a PNG in results/figures."""
    fig.savefig(project_path(config["paths"]["figures_dir"]) / f"{name}.png", bbox_inches="tight")


def feature_label(column: str) -> str:
    """Readable label for a column name."""
    return FEATURE_LABELS.get(column, column.replace("_", " ").capitalize())


def _panel_grid(n_panels: int, n_columns: int, panel_size: tuple[float, float]) -> tuple[plt.Figure, np.ndarray]:
    """Create a grid of panels, hiding any spare ones."""
    n_rows = int(np.ceil(n_panels / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(panel_size[0] * n_columns, panel_size[1] * n_rows), squeeze=False)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    return fig, axes.flat[:n_panels]


def plot_numeric_histograms(df: pd.DataFrame, columns: list[str], title: str) -> plt.Figure:
    """One histogram per numeric column. Discrete columns get one bar per value,
    because grouped bins can merge neighbouring integers and create false spikes."""
    fig, axes = _panel_grid(len(columns), 3, (4.0, 2.7))
    for ax, column in zip(axes, columns):
        values = df[column]
        if values.nunique() <= DISCRETE_VALUE_LIMIT:
            counts = values.value_counts().sort_index()
            ax.bar(counts.index, counts.values, width=0.8, color=PRIMARY)
        else:
            ax.hist(values, bins=40, color=PRIMARY, edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(f"{feature_label(column)} (skew {values.skew():.2f})")
        ax.set_xlabel(feature_label(column))
        ax.set_ylabel("Users")
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_boxplots(df: pd.DataFrame, columns: list[str], title: str) -> plt.Figure:
    """One horizontal boxplot per column, each on its own scale."""
    fig, axes = _panel_grid(len(columns), 3, (4.0, 1.5))
    for ax, column in zip(axes, columns):
        ax.boxplot(
            df[column],
            orientation="horizontal",
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": PRIMARY},
            medianprops={"color": "#104281", "linewidth": 2},
            whiskerprops={"color": PRIMARY},
            capprops={"color": PRIMARY},
            flierprops={"marker": "o", "markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.5},
        )
        ax.set_yticks([])
        ax.set_xlabel(feature_label(column))
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_category_frequencies(df: pd.DataFrame, columns: list[str], title: str) -> plt.Figure:
    """One ranked horizontal bar chart per categorical column, as percentages."""
    fig, axes = _panel_grid(len(columns), 3, (4.2, 3.2))
    for ax, column in zip(axes, columns):
        shares = 100 * df[column].value_counts(normalize=True).sort_values()
        ax.barh(shares.index.astype(str), shares.values, height=0.72, color=PRIMARY)
        ax.set_title(feature_label(column))
        ax.set_xlabel("Users (%)")
        ax.set_xlim(left=0)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_association_heatmap(matrix: pd.DataFrame, title: str, vmin: float = -1, vmax: float = 1, colorbar_label: str = "Pearson correlation") -> plt.Figure:
    """Annotated heatmap of a symmetric association matrix."""
    labels = [feature_label(column) for column in matrix.columns]
    cmap = DIVERGING_CMAP if vmin < 0 else SEQUENTIAL_CMAP
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 0.7 * len(labels) + 2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=1,
        linecolor=SURFACE,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40)
    plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def plot_relationship_panels(df: pd.DataFrame, seed: int, sample_size: int = 5000) -> plt.Figure:
    """Three example relationships. Each panel title reports the correlation on all users."""
    sample = df.sample(sample_size, random_state=seed)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    pairs = [("avg_listening_hours_per_week", "playlists_created"), ("avg_listening_hours_per_week", "avg_skips_per_day")]
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(sample[x], sample[y], s=6, alpha=0.2, color=PRIMARY, linewidths=0)
        ax.set_xlabel(feature_label(x))
        ax.set_ylabel(feature_label(y))
        ax.set_title(f"r = {df[x].corr(df[y]):.3f}")
    hours_by_rating = [group["avg_listening_hours_per_week"].to_numpy() for _, group in df.groupby("music_suggestion_rating_1_to_5")]
    axes[2].boxplot(
        hours_by_rating,
        tick_labels=sorted(df["music_suggestion_rating_1_to_5"].unique()),
        patch_artist=True,
        boxprops={"facecolor": "#cde2fb", "edgecolor": PRIMARY},
        medianprops={"color": "#104281", "linewidth": 2},
        whiskerprops={"color": PRIMARY},
        capprops={"color": PRIMARY},
        flierprops={"marker": "o", "markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.4},
    )
    correlation = df["music_suggestion_rating_1_to_5"].corr(df["avg_listening_hours_per_week"])
    axes[2].set_title(f"r = {correlation:.3f}")
    axes[2].set_xlabel(feature_label("music_suggestion_rating_1_to_5"))
    axes[2].set_ylabel(feature_label("avg_listening_hours_per_week"))
    fig.suptitle(f"Behavioural relationships (scatter panels show {sample_size:,} sampled users)", fontweight="bold")
    return fig


def plot_box_by_group(df: pd.DataFrame, group_column: str, value_columns: list[str], title: str) -> plt.Figure:
    """Boxplots of several numeric columns for each level of one group."""
    levels = sorted(df[group_column].unique())
    fig, axes = plt.subplots(1, len(value_columns), figsize=(3.0 * len(value_columns), 3.3))
    for ax, column in zip(np.atleast_1d(axes), value_columns):
        data = [df.loc[df[group_column] == level, column].to_numpy() for level in levels]
        ax.boxplot(
            data,
            tick_labels=levels,
            patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": PRIMARY},
            medianprops={"color": "#104281", "linewidth": 2},
            whiskerprops={"color": PRIMARY},
            capprops={"color": PRIMARY},
            flierprops={"marker": "o", "markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": "none", "alpha": 0.4},
        )
        ax.set_ylabel(feature_label(column))
        ax.set_xlabel(feature_label(group_column))
        ax.grid(axis="x", visible=False)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_elbow(scan: pd.DataFrame, selected_k: int, title: str) -> plt.Figure:
    """Inertia against k, marking the selected k."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(scan["k"], scan["inertia"], color=PRIMARY, linewidth=2, marker="o", markersize=6)
    chosen = scan.loc[scan["k"] == selected_k].iloc[0]
    ax.annotate(
        f"Selected k = {selected_k}",
        (chosen["k"], chosen["inertia"]),
        xytext=(10, 12),
        textcoords="offset points",
        color=INK_SECONDARY,
    )
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_xticks(scan["k"])
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    return fig


def plot_metrics_vs_reference(
    real: pd.DataFrame, reference_summary: pd.DataFrame, selected_k: int, title: str, real_label: str = "Real data"
) -> plt.Figure:
    """One panel per internal metric: real data against the permuted-reference range."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for ax, metric in zip(axes, METRIC_LABELS):
        band = reference_summary[metric]
        ax.fill_between(band.index, band["min"], band["max"], color=REFERENCE, alpha=0.3, linewidth=0, label="Permuted reference (range)")
        ax.plot(band.index, band["mean"], color=REFERENCE, linewidth=2, label="Permuted reference (mean)")
        ax.plot(real["k"], real[metric], color=PRIMARY, linewidth=2, marker="o", markersize=5, label=real_label)
        ax.axvline(selected_k, color=AXIS, linewidth=1, zorder=0)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(real["k"])
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{title} (vertical line: selected k = {selected_k})", fontweight="bold")
    return fig


def plot_stability(stability: pd.DataFrame, title: str) -> plt.Figure:
    """Stability against k for each algorithm, on real and permuted data."""
    algorithms = list(stability["algorithm"].unique())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5.2 * len(algorithms), 3.5), sharey=True, squeeze=False)
    styles = {"Real data": (PRIMARY, "o"), "Permuted reference": (REFERENCE, "s")}
    for ax, algorithm in zip(axes.flat, algorithms):
        subset = stability[stability["algorithm"] == algorithm]
        for dataset, (color, marker) in styles.items():
            rows = subset[subset["dataset"] == dataset]
            ax.plot(rows["k"], rows["mean_ari"], color=color, marker=marker, markersize=5, linewidth=2, label=dataset)
        ax.set_title(algorithm)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(sorted(subset["k"].unique()))
    axes.flat[0].set_ylabel("Mean pairwise ARI between subsamples")
    axes.flat[0].set_ylim(0, 1.05)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    return fig


def _draw_dendrogram(ax: plt.Axes, linkage_matrix: np.ndarray, title: str, last_merges: int = 30, cut: float | None = None) -> None:
    """Draw a truncated dendrogram in one neutral color."""
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=last_merges,
        no_labels=True,
        ax=ax,
        color_threshold=0,
        above_threshold_color=INK_SECONDARY,
    )
    if cut is not None:
        ax.axhline(cut, color=PRIMARY, linewidth=1.5, label="Cut giving the selected k")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title)
    ax.set_xlabel(f"Clusters remaining at the last {last_merges} merges")
    ax.set_ylabel("Merge distance")
    ax.grid(axis="x", visible=False)


def plot_linkage_dendrograms(linkages: dict[str, np.ndarray], title: str) -> plt.Figure:
    """Compare truncated dendrograms for several linkage methods."""
    fig, axes = plt.subplots(1, len(linkages), figsize=(4.6 * len(linkages), 3.6))
    for ax, (method, matrix) in zip(np.atleast_1d(axes), linkages.items()):
        _draw_dendrogram(ax, matrix, f"{method.capitalize()} linkage")
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_single_dendrogram(linkage_matrix: np.ndarray, title: str, cut: float) -> plt.Figure:
    """One truncated dendrogram with a horizontal line at the selected cut."""
    fig, ax = plt.subplots(figsize=(9, 4))
    _draw_dendrogram(ax, linkage_matrix, title, cut=cut)
    return fig


def plot_merge_heights(linkage_matrix: np.ndarray, selected_k: int, title: str, k_max: int = 15) -> plt.Figure:
    """Merge distance needed to go from k clusters to k-1, to look for a clear gap."""
    k_values = np.arange(2, k_max + 1)
    heights = linkage_matrix[-(k_values - 1), 2]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(k_values, heights, color=PRIMARY, linewidth=2, marker="o", markersize=5)
    ax.axvline(selected_k, color=AXIS, linewidth=1, zorder=0)
    ax.set_xlabel("Number of clusters before the merge (k)")
    ax.set_ylabel("Merge distance")
    ax.set_xticks(k_values)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    return fig


def plot_linkage_metrics(scans: pd.DataFrame, title: str) -> plt.Figure:
    """Silhouette and smallest-cluster share for each linkage, on the same sample."""
    colors = {"ward": "#2a78d6", "average": "#eb6834", "complete": "#1baf7a", "single": "#eda100"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for method, rows in scans.groupby("linkage"):
        axes[0].plot(rows["k"], rows["silhouette"], color=colors[method], linewidth=2, marker="o", markersize=5, label=method.capitalize())
        axes[1].plot(rows["k"], 100 * rows["smallest_cluster_share"], color=colors[method], linewidth=2, marker="o", markersize=5, label=method.capitalize())
    axes[0].set_title(METRIC_LABELS["silhouette"])
    axes[0].set_ylabel("Silhouette")
    axes[1].set_title("Smallest cluster as a share of the sample")
    axes[1].set_ylabel("Users in smallest cluster (%)")
    for ax in axes:
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(sorted(scans["k"].unique()))
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_pca_segments(coordinates: np.ndarray, segments: pd.Series, explained: np.ndarray, title: str, seed: int, sample_size: int = 6000) -> plt.Figure:
    """Small multiples of PC1 and PC2: each panel highlights one segment on gray."""
    rng = np.random.default_rng(seed)
    index = rng.choice(len(coordinates), size=sample_size, replace=False)
    points, labels = coordinates[index], segments.to_numpy()[index]
    names = [name for name in SEGMENT_COLORS if name in set(labels)]
    fig, axes = plt.subplots(1, len(names), figsize=(2.7 * len(names), 3.0), sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        mask = labels == name
        ax.scatter(points[~mask, 0], points[~mask, 1], s=3, color=GRID, linewidths=0)
        ax.scatter(points[mask, 0], points[mask, 1], s=4, color=SEGMENT_COLORS[name], alpha=0.7, linewidths=0)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}%)")
    np.atleast_1d(axes)[0].set_ylabel(f"PC2 ({100 * explained[1]:.1f}%)")
    fig.suptitle(title, fontweight="bold")
    return fig


def _spread_labels(positions: dict[str, np.ndarray], min_dx: float = 1.5, min_dy: float = 0.28) -> dict[str, np.ndarray]:
    """Move labels up or down just enough that their boxes do not cover each other."""
    placed: dict[str, np.ndarray] = {}
    for name, position in sorted(positions.items(), key=lambda item: item[1][1]):
        x, y = position
        for other in placed.values():
            if abs(other[0] - x) < min_dx and y - other[1] < min_dy:
                y = other[1] + min_dy
        placed[name] = np.array([x, y])
    return placed


def plot_pca_all_segments(coordinates: np.ndarray, segments: pd.Series, explained: np.ndarray, title: str, seed: int, sample_size: int = 6000) -> plt.Figure:
    """All segments in one PC1/PC2 scatter plot, with each centroid labelled directly.

    Points are drawn in random order so that no segment hides behind another.
    """
    rng = np.random.default_rng(seed)
    index = rng.permutation(rng.choice(len(coordinates), size=sample_size, replace=False))
    points, labels = coordinates[index], segments.to_numpy()[index]
    names = [name for name in SEGMENT_COLORS if name in set(labels)]

    fig, ax = plt.subplots(figsize=(9, 6))
    # Each small batch holds every segment, which keeps the drawing order mixed.
    for batch in np.array_split(np.arange(sample_size), 60):
        for name in names:
            mask = labels[batch] == name
            ax.scatter(
                points[batch][mask, 0],
                points[batch][mask, 1],
                s=11,
                marker=SEGMENT_MARKERS[name],
                color=SEGMENT_COLORS[name],
                alpha=0.75,
                linewidths=0,
            )
    centres = {name: coordinates[segments.to_numpy() == name].mean(axis=0) for name in names}
    for name, centre in _spread_labels(centres).items():
        ax.annotate(
            name,
            centre,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=INK,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": SURFACE, "edgecolor": AXIS, "linewidth": 0.8},
        )
    handles = [
        plt.Line2D([], [], linestyle="none", marker=SEGMENT_MARKERS[name], color=SEGMENT_COLORS[name], markersize=7, label=name)
        for name in names
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% of variance)")
    ax.set_title(title)
    return fig


def plot_segment_sizes(sizes: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Share of users per segment, one panel per algorithm."""
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 3.0), sharex=True)
    for ax, (algorithm, table) in zip(np.atleast_1d(axes), sizes.items()):
        ordered = table.sort_values("percent")
        bars = ax.barh(ordered.index, ordered["percent"], height=0.72, color=[SEGMENT_COLORS[name] for name in ordered.index])
        ax.bar_label(bars, labels=[f"{p:.1f}%" for p in ordered["percent"]], padding=3, fontsize=9, color=INK_SECONDARY)
        ax.set_title(algorithm)
        ax.set_xlabel("Users (%)")
        ax.set_xlim(0, ordered["percent"].max() * 1.2)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_profile_heatmaps(z_profiles: dict[str, pd.DataFrame], unit_profiles: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """Color shows z-score distance from the average user; text shows original-unit means."""
    fig, axes = plt.subplots(1, len(z_profiles), figsize=(6.2 * len(z_profiles), 3.3))
    for ax, (algorithm, z_values) in zip(np.atleast_1d(axes), z_profiles.items()):
        annotations = unit_profiles[algorithm].loc[z_values.index, z_values.columns]
        sns.heatmap(
            z_values,
            annot=annotations,
            fmt=".1f",
            cmap=DIVERGING_CMAP,
            vmin=-1.8,
            vmax=1.8,
            linewidths=1,
            linecolor=SURFACE,
            cbar_kws={"label": "Mean z-score"},
            xticklabels=[feature_label(column) for column in z_values.columns],
            ax=ax,
        )
        ax.set_title(algorithm)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.suptitle(title, fontweight="bold")
    return fig


def plot_contingency(table: pd.DataFrame, title: str) -> plt.Figure:
    """Heatmap of how users from one partition are distributed across another."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.heatmap(table, annot=True, fmt=",d", cmap=SEQUENTIAL_CMAP, linewidths=1, linecolor=SURFACE, cbar_kws={"label": "Users"}, ax=ax)
    ax.set_xlabel("Ward segment")
    ax.set_ylabel("K-Means segment")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    plt.setp(ax.get_xticklabels(), ha="right")
    return fig


def plot_external_rates(rates: pd.DataFrame, overall: pd.Series, title: str) -> plt.Figure:
    """Rate of each external variable per segment against its overall rate."""
    fig, axes = plt.subplots(1, rates.shape[1], figsize=(3.7 * rates.shape[1], 3.0), sharey=True)
    for ax, column in zip(np.atleast_1d(axes), rates.columns):
        ax.barh(rates.index, rates[column], height=0.72, color=[SEGMENT_COLORS[name] for name in rates.index])
        reference_line = ax.axvline(overall[column], color=INK, linewidth=1.5)
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("Users (%)")
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
    fig.legend([reference_line], ["All users"], loc="outside lower center", fontsize=9)
    fig.suptitle(title, fontweight="bold")
    return fig
