"""Draw the French charts for slides 6, 7, 8, 11 and 13 from the course-metric results.

Run from the project root, after scripts/course_metrics.py, course_shuffle_test.py and course_dunn.py:
.venv/bin/python scripts/make_course_figures.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from make_french_figures import SEGMENT_NAMES, save_in_french, to_french, viz
from src.clustering import name_segments
from src.data_loader import load_clean_data, load_config, project_path

SAMPLE_SIZE = 6000
LINKAGES = {"single": ("Saut minimal", "#eda100", "D"), "complete": ("Saut maximal", "#1baf7a", "^"), "ward": ("Ward", "#2a78d6", "o")}


def plot_kmeans_choice(scan: pd.DataFrame, selected_k: int):
    """The elbow and the silhouette of K-Means, each with the selected k marked."""
    rows = scan[scan["method"] == "kmeans"]
    fig, axes = viz.plt.subplots(1, 2, figsize=(9.2, 4.9))
    axes[0].plot(rows["k"], rows["inertia"], color=viz.PRIMARY, marker="o", markersize=7, linewidth=2)
    axes[0].set_title("Coude : l'inertie baisse moins après k = 5", fontsize=12)
    axes[0].set_ylabel("Inertie (somme des carrés intra-cluster)")
    axes[0].set_ylim(0, None)
    axes[1].plot(rows["k"], rows["silhouette_all_users"], color=viz.PRIMARY, marker="o", markersize=7, linewidth=2)
    axes[1].set_title("Silhouette : k = 2, puis k = 5", fontsize=12)
    axes[1].set_ylabel("Silhouette moyenne (50 000 utilisateurs)")
    axes[1].set_ylim(0.16, 0.20)
    for k, offset in [(2, (10, -3)), (selected_k, (0, 10))]:
        value = rows.loc[rows["k"] == k, "silhouette_all_users"].iloc[0]
        axes[1].annotate(f"{value:.3f}", (k, value), textcoords="offset points", xytext=offset, ha="left" if k == 2 else "center", fontsize=11, color=viz.INK)
    for ax in axes:
        ax.axvline(selected_k, color=viz.AXIS, linewidth=1, zorder=0)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(rows["k"])
    return fig


def plot_linkage_choice(scan: pd.DataFrame, selected_k: int):
    """Silhouette and largest cluster per k, for the three linkages on the slide."""
    fig, axes = viz.plt.subplots(1, 2, figsize=(12.5, 3.65))
    for method, (label, color, marker) in LINKAGES.items():
        rows = scan[scan["method"] == method]
        axes[0].plot(rows["k"], rows["silhouette_all_users"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
        axes[1].plot(rows["k"], 100 * rows["largest_cluster_share"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
    axes[0].set_title("Silhouette : le saut minimal semble le meilleur")
    axes[0].set_ylabel("Silhouette (50 000 utilisateurs)")
    axes[0].legend(fontsize=9, loc="center right", bbox_to_anchor=(1, 0.46))
    axes[1].set_title("Mais son plus grand cluster contient presque tout le monde")
    axes[1].set_ylabel("Plus grand cluster (% des utilisateurs)")
    axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.axvline(selected_k, color=viz.AXIS, linewidth=1, zorder=0)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(sorted(scan["k"].unique()))
    return fig


def plot_shuffle_test(scan: pd.DataFrame, shuffled: pd.DataFrame, jumps: pd.DataFrame, selected_k: int):
    """Real data against the range of the shuffled datasets, per k."""
    fig, axes = viz.plt.subplots(1, 3, figsize=(12.5, 3.6))
    panels = []
    for method, title in [("kmeans", "K-Means : silhouette"), ("ward", "Ward : silhouette")]:
        real = scan[scan["method"] == method].set_index("k")["silhouette_all_users"]
        band = shuffled[shuffled["method"] == method].groupby("k")["silhouette_all_users"].agg(["min", "max"])
        panels.append((title, "Silhouette (50 000 utilisateurs)", real, band))
    k_values = sorted(scan["k"].unique())
    real_jumps = jumps[jumps["dataset"] == "real"].set_index("k")["ratio"].loc[k_values]
    jump_band = jumps[jumps["dataset"] != "real"].groupby("k")["ratio"].agg(["min", "max"]).loc[k_values]
    panels.append(("Ward : saut d'inertie", "Saut de k à k − 1 / saut vers k", real_jumps, jump_band))
    for ax, (title, y_label, real, band) in zip(axes, panels):
        ax.fill_between(band.index, band["min"], band["max"], color=viz.REFERENCE, alpha=0.3, linewidth=0, label="Données mélangées (étendue)")
        ax.plot(real.index, real.values, color=viz.PRIMARY, marker="o", markersize=5, linewidth=2, label="Données réelles")
        ax.axvline(selected_k, color=viz.AXIS, linewidth=1, zorder=0)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(k_values)
    axes[0].legend(fontsize=9, loc="upper right")
    return fig


def plot_k2_against_k5(centroids: pd.DataFrame, dunn: pd.DataFrame, users: pd.DataFrame, selected_k: int):
    """K-Means centroids at k = 2 and the selected k, next to the Dunn index for each k."""
    columns = ["avg_listening_hours_per_week", "playlists_created", "avg_skips_per_day", "music_suggestion_rating_1_to_5"]
    two, chosen = centroids[centroids["k"] == 2], centroids[centroids["k"] == selected_k]
    names = name_segments(chosen.set_index("cluster")[columns])
    labels = [f"k = 2 · note {'1 à 3' if row[columns[3]] < 0 else '4 ou 5'} ({row['share']:.0%})" for _, row in two.iterrows()]
    labels += [f"k = {selected_k} · {SEGMENT_NAMES[names[row['cluster']]]} ({row['share']:.0%})" for _, row in chosen.iterrows()]
    z = pd.concat([two, chosen])[columns].to_numpy()
    means = z * users[columns].std(ddof=0).to_numpy() + users[columns].mean().to_numpy()

    fig, axes = viz.plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    viz.sns.heatmap(
        z,
        annot=means,
        fmt=".1f",
        cmap=viz.DIVERGING_CMAP,
        vmin=-1.8,
        vmax=1.8,
        linewidths=1,
        linecolor=viz.SURFACE,
        cbar_kws={"label": "Écart à l'utilisateur moyen (score z)"},
        xticklabels=["Heures / semaine", "Playlists", "Sauts / jour", "Note (1-5)"],
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].axhline(2, color=viz.INK, linewidth=2)
    axes[0].set_title("k = 2 : seule la note change. k = 5 : chaque variable compte", fontsize=12)
    axes[0].tick_params(axis="y", labelsize=9.5)

    kmeans = dunn[dunn["method"] == "kmeans"]
    axes[1].bar(kmeans["k"], kmeans["dunn"], color=viz.PRIMARY, width=0.6)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-4, 0.5)
    axes[1].set_yticks([1e-4, 1e-3, 1e-2, 1e-1], ["0.0001", "0.001", "0.01", "0.1"])
    axes[1].minorticks_off()
    for _, row in kmeans[kmeans["k"] <= 3].iterrows():
        axes[1].annotate(f"{row['dunn']:.3f}", (row["k"], row["dunn"]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, color=viz.INK_SECONDARY)
    axes[1].annotate("écart d'une note\n(entre 3 et 4)", (2, kmeans["dunn"].iloc[0]), textcoords="offset points", xytext=(28, 14), fontsize=9, color=viz.INK)
    axes[1].annotate("à partir de k = 4 : ≈ 0,0003\nles clusters se touchent", (7, 0.0011), ha="center", fontsize=9, color=viz.INK)
    axes[1].set_title("Indice de Dunn (K-Means), échelle log", fontsize=12)
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Dunn = min(inter) / max(intra)")
    axes[1].set_xticks(kmeans["k"])
    axes[1].grid(axis="x", visible=False)
    return fig


def plot_two_features(users: pd.DataFrame, segments: pd.Series, title: str, seed: int):
    """Hours against skips, for one sample of users, with each segment's mean labelled."""
    x_column, y_column = "avg_listening_hours_per_week", "avg_skips_per_day"
    rng = np.random.default_rng(seed)
    index = rng.permutation(rng.choice(len(users), size=SAMPLE_SIZE, replace=False))
    x, y, labels = users[x_column].to_numpy(), users[y_column].to_numpy(), segments.to_numpy()
    names = [name for name in viz.SEGMENT_COLORS if name in set(labels)]

    fig, ax = viz.plt.subplots(figsize=(9, 6))
    # Each small batch holds every segment, which keeps the drawing order mixed.
    for batch in np.array_split(index, 60):
        for name in names:
            chosen = batch[labels[batch] == name]
            ax.scatter(x[chosen], y[chosen], s=11, marker=viz.SEGMENT_MARKERS[name], color=viz.SEGMENT_COLORS[name], alpha=0.75, linewidths=0)
    centres = {name: np.array([x[labels == name].mean(), y[labels == name].mean()]) for name in names}
    for name, centre in viz._spread_labels(centres, min_dx=6, min_dy=1).items():
        ax.annotate(
            name,
            centre,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=viz.INK,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": viz.SURFACE, "edgecolor": viz.AXIS, "linewidth": 0.8},
        )
    handles = [viz.plt.Line2D([], [], linestyle="none", marker=viz.SEGMENT_MARKERS[name], color=viz.SEGMENT_COLORS[name], markersize=7, label=name) for name in names]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("Heures d'écoute par semaine")
    ax.set_ylabel("Sauts par jour")
    ax.set_title(title)
    return fig


def find_example_pair(users: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """Two users who coincide on hours and skips but differ on the two hidden features."""
    columns = ["avg_listening_hours_per_week", "avg_skips_per_day", "playlists_created", "music_suggestion_rating_1_to_5"]
    table = users[columns].assign(segment=segments.values)
    builders = table[(table["segment"] == "Playlist Builders") & (table["music_suggestion_rating_1_to_5"] == 4) & (table["playlists_created"] >= 16)]
    raters = table[(table["segment"] == "Low-Rating Users") & (table["music_suggestion_rating_1_to_5"] == 1) & (table["playlists_created"] <= 6)]
    merged = builders.reset_index().merge(raters.reset_index(), on=["avg_listening_hours_per_week", "avg_skips_per_day"], suffixes=("_a", "_b"))
    return merged.sort_values(["index_a", "index_b"]).head(1)


def main() -> None:
    config = load_config()
    selected_k = config["kmeans"]["selected_k"]
    seed = config["random_seed"]
    metrics_dir = project_path(config["paths"]["metrics_dir"])
    profiles_dir = project_path(config["paths"]["profiles_dir"])
    viz.apply_plot_style()

    scan = pd.read_csv(metrics_dir / "course_full_data_scan.csv")
    save_in_french(plot_kmeans_choice(scan, selected_k), "kmeans_coude_silhouette")
    save_in_french(plot_linkage_choice(scan, selected_k), "cah_linkage_choice_full")
    shuffled = pd.read_csv(metrics_dir / "course_shuffle_scan.csv")
    jumps = pd.read_csv(metrics_dir / "course_shuffle_jumps.csv")
    save_in_french(plot_shuffle_test(scan, shuffled, jumps, selected_k), "test_melange_cours")

    users = load_clean_data(config)
    centroids = pd.read_csv(metrics_dir / "course_kmeans_centroids_by_k.csv")
    dunn = pd.read_csv(metrics_dir / "course_dunn.csv")
    save_in_french(plot_k2_against_k5(centroids, dunn, users, selected_k), "kmeans_k2_contre_k5")
    assignments = pd.read_csv(profiles_dir / "final_segment_assignments.csv")
    assert (assignments["user_id"] == users["user_id"]).all()
    for name, column, file_name in [("K-Means", "kmeans_segment", "kmeans_heures_sauts"), ("Ward (CAH)", "ward_segment", "cah_heures_sauts")]:
        save_in_french(plot_two_features(users, assignments[column], f"{name} : 2 variables sur 4 ({SAMPLE_SIZE:,} utilisateurs tirés au hasard)".replace(",", " "), seed), file_name)

    print(to_french("Saved charts."))
    print(find_example_pair(users, assignments["kmeans_segment"]).T)


if __name__ == "__main__":
    main()
