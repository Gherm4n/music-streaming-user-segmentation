"""Redraw the presentation charts with French labels, from the saved project results.

Run from the project root, after the notebooks: .venv/bin/python scripts/make_french_figures.py
The charts are written to results/figures_fr/. Only the text changes, not the data.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import Formatter
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import visualization as viz
from src.clustering import cut_height, full_ward_linkage
from src.data_loader import load_clean_data, load_config, load_representation, project_path
from src.evaluation import summarise_reference

OUTPUT_DIR = project_path("results/figures_fr")

SEGMENT_NAMES = {
    "Heavy Listeners": "Gros auditeurs",
    "Light Listeners": "Petits auditeurs",
    "Playlist Builders": "Créateurs de playlists",
    "Frequent Skippers": "Zappeurs fréquents",
    "Low-Rating Users": "Utilisateurs à note faible",
    "Low-Skip Listeners": "Auditeurs peu zappeurs",
}

EXACT_TRANSLATIONS = {
    **SEGMENT_NAMES,
    "Users": "Utilisateurs",
    "Users (%)": "Utilisateurs (%)",
    "All users": "Tous les utilisateurs",
    "Mean z-score": "Score z moyen",
    "Number of clusters (k)": "Nombre de clusters (k)",
    "Inertia (within-cluster sum of squares)": "Inertie (somme des carrés intra-cluster)",
    "Merge distance": "Distance de fusion",
    "Clusters remaining at the last 30 merges": "Clusters restants aux 30 dernières fusions",
    "Cut giving the selected k": "Coupe donnant le k retenu",
    "Permuted reference (range)": "Référence mélangée (étendue)",
    "Permuted reference (mean)": "Référence mélangée (moyenne)",
}

PARTIAL_TRANSLATIONS = {
    "(vertical line: selected k = ": "(ligne verticale : k retenu = ",
    "Selected k = ": "k retenu = ",
    "(skew ": "(asymétrie ",
    " of variance": " de la variance",
    "PC1": "CP1",
    "PC2": "CP2",
}

viz.FEATURE_LABELS.update(
    {
        "avg_listening_hours_per_week": "Heures d'écoute par semaine",
        "playlists_created": "Playlists créées",
        "avg_skips_per_day": "Sauts par jour",
        "music_suggestion_rating_1_to_5": "Note des suggestions (1-5)",
        "age": "Âge (années)",
        "months_inactive": "Mois d'inactivité",
        "tenure_months": "Ancienneté (mois)",
        "inactive_3_months_flag": "Indicateur d'inactivité 3 mois",
    }
)
viz.METRIC_LABELS.update(
    {
        "silhouette": "Silhouette (plus élevé = mieux)",
        "davies_bouldin": "Davies-Bouldin (plus bas = mieux)",
        "calinski_harabasz": "Calinski-Harabasz (plus élevé = mieux)",
    }
)


def to_french(text: str) -> str:
    """Translate chart text and apply French number formatting."""
    if text in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[text]
    for english, french in PARTIAL_TRANSLATIONS.items():
        text = text.replace(english, french)
    text = re.sub(r"(\d)\.(\d)", r"\1,\2", text)
    return re.sub(r"(\d)%", "\\1\u00a0%", text)


class FrenchTickFormatter(Formatter):
    """Pass another formatter's tick labels through to_french, at draw time."""

    def __init__(self, base: Formatter):
        self.base = base

    def set_locs(self, locs):
        self.base.set_locs(locs)

    def __call__(self, x, pos=None):
        return to_french(self.base(x, pos))


def save_in_french(fig, name: str) -> None:
    """Translate every text element of a finished figure, then save it."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if not isinstance(formatter, FrenchTickFormatter):
                axis.set_major_formatter(FrenchTickFormatter(formatter))
    for text in fig.findobj(Text):
        text.set_text(to_french(text.get_text()))
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight")
    viz.plt.close(fig)


def plot_linkage_choice(comparison: pd.DataFrame, selected_k: int):
    """Silhouette and largest cluster per k, for the three linkages shown on the slide."""
    linkages = {"single": ("Saut minimal", "#eda100", "D"), "complete": ("Saut maximal", "#1baf7a", "^"), "ward": ("Ward", "#2a78d6", "o")}
    fig, axes = viz.plt.subplots(1, 2, figsize=(12.5, 3.65))
    for method, (label, color, marker) in linkages.items():
        rows = comparison[comparison["linkage"] == method]
        axes[0].plot(rows["k"], rows["silhouette"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
        axes[1].plot(rows["k"], 100 * rows["largest_cluster_share"], color=color, marker=marker, markersize=6, linewidth=2, label=label)
    axes[0].set_title("Silhouette : le saut minimal semble le meilleur")
    axes[0].set_ylabel("Silhouette")
    axes[1].set_title("Mais son plus grand cluster contient presque tout le monde")
    axes[1].set_ylabel("Plus grand cluster (% de l'échantillon)")
    axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.axvline(selected_k, color=viz.AXIS, linewidth=1, zorder=0)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(sorted(comparison["k"].unique()))
    axes[0].legend(fontsize=9)
    return fig


def main() -> None:
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    viz.apply_plot_style()

    seed = config["random_seed"]
    features = config["columns"]["representation_a_numeric"]
    selected_k = config["kmeans"]["selected_k"]
    metrics_dir = project_path(config["paths"]["metrics_dir"])
    profiles_dir = project_path(config["paths"]["profiles_dir"])

    users = load_clean_data(config)
    representation_a = load_representation(config, "a")
    X = representation_a.to_numpy()
    assignments = pd.read_csv(profiles_dir / "final_segment_assignments.csv")
    assert (assignments["user_id"] == users["user_id"]).all() and (representation_a.index == users["user_id"]).all()
    segments = {"K-Means": assignments["kmeans_segment"], "Ward (CAH)": assignments["ward_segment"]}

    numeric_columns = ["age", "months_inactive", "music_suggestion_rating_1_to_5", "avg_listening_hours_per_week", "playlists_created", "avg_skips_per_day"]
    save_in_french(
        viz.plot_numeric_histograms(users, numeric_columns, "Distribution des variables numériques (50 000 utilisateurs)"),
        "eda_numeric_distributions",
    )

    correlation = users[numeric_columns + ["inactive_3_months_flag", "tenure_months"]].corr()
    save_in_french(
        viz.plot_association_heatmap(correlation, "Corrélation de Pearson entre les variables numériques", colorbar_label="Corrélation de Pearson"),
        "eda_correlation_heatmap",
    )

    kmeans_scan = pd.read_csv(metrics_dir / "kmeans_k_scan.csv")
    save_in_french(viz.plot_elbow(kmeans_scan, selected_k, "Courbe du coude de K-Means"), "kmeans_elbow")

    linkage_matrix = full_ward_linkage(X)
    save_in_french(
        viz.plot_single_dendrogram(linkage_matrix, "Dendrogramme de Ward sur les 50 000 utilisateurs (30 dernières fusions)", cut_height(linkage_matrix, selected_k)),
        "cah_ward_dendrogram_full",
    )

    linkage_comparison = pd.read_csv(metrics_dir / "cah_linkage_comparison_sample.csv")
    save_in_french(plot_linkage_choice(linkage_comparison, selected_k), "cah_linkage_choice")

    z_profiles = {name: representation_a.reset_index(drop=True).groupby(labels.values).mean() for name, labels in segments.items()}
    unit_profiles = {name: users.groupby(labels.values)[features].mean() for name, labels in segments.items()}
    save_in_french(
        viz.plot_profile_heatmaps(z_profiles, unit_profiles, "Profils des segments : le texte donne la moyenne en unités d'origine, la couleur l'écart à l'utilisateur moyen"),
        "comparison_profile_heatmaps",
    )

    pca = PCA(random_state=seed).fit(X)
    coordinates = pca.transform(X)[:, :2]
    shown = pca.explained_variance_ratio_[:2].sum()
    for name, file_name in [("K-Means", "kmeans_pca_all_segments"), ("Ward (CAH)", "cah_pca_all_segments")]:
        title = f"{name} : tous les segments dans une vue ACP (6 000 utilisateurs ; {shown:.1%} de la variance)"
        save_in_french(viz.plot_pca_all_segments(coordinates, segments[name], pca.explained_variance_ratio_, title, seed), file_name)

    reference = summarise_reference(pd.read_csv(metrics_dir / "kmeans_reference_scan.csv"))
    save_in_french(
        viz.plot_metrics_vs_reference(kmeans_scan, reference, selected_k, "K-Means : données réelles et données mélangées", real_label="Données réelles"),
        "kmeans_metrics_vs_reference",
    )

    flags = pd.DataFrame(
        {
            "Statut inactif": users["subscription_status"] == "Inactive",
            "Inactifs depuis 3 mois ou plus": users["inactive_3_months_flag"] == 1,
            "Convertis par une publicité": users["ad_conversion_to_subscription"] == "Yes",
            "Abonnement payant": users["subscription_type"] != "Free",
        }
    )
    rates = 100 * flags.groupby(segments["K-Means"].values).mean()
    saved_rates = pd.read_csv(profiles_dir / "kmeans_external_rates.csv", index_col=0)
    assert np.allclose(rates.round(2).to_numpy(), saved_rates.loc[rates.index].to_numpy()), "Rates must match the saved results"
    save_in_french(
        viz.plot_external_rates(rates, 100 * flags.mean(), "K-Means : segments et ensemble des utilisateurs, sur des variables non utilisées"),
        "comparison_external_rates_kmeans",
    )

    print(f"Saved {len(list(OUTPUT_DIR.glob('*.png')))} charts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
