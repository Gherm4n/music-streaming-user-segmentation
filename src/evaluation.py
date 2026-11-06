"""Internal metrics, reference comparisons, stability and cluster profiling."""

import itertools
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from src.clustering import (
    assign_to_nearest_centroid,
    cut_tree,
    fit_kmeans,
    full_ward_linkage,
    permute_columns,
)


def make_evaluation_index(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Pick the fixed users on which Silhouette is calculated.

    Silhouette needs all pairwise distances among the users it scores. Using the
    same subset for every candidate keeps the comparison fair and affordable.
    """
    return np.random.default_rng(seed).choice(n_rows, size=size, replace=False)


def internal_metrics(X: np.ndarray, labels: np.ndarray, silhouette_index: np.ndarray | None = None) -> dict:
    """Silhouette, Davies-Bouldin and Calinski-Harabasz, plus cluster balance.

    Silhouette uses silhouette_index when given; the other two metrics are cheap
    and always use every row.
    """
    if silhouette_index is None:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = silhouette_score(X[silhouette_index], labels[silhouette_index])
    shares = np.bincount(labels) / len(labels)
    return {
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "smallest_cluster_share": shares.min(),
        "largest_cluster_share": shares.max(),
    }


def scan_kmeans(
    X: np.ndarray, k_values: list[int], seed: int, n_init: int, max_iter: int, silhouette_index: np.ndarray
) -> pd.DataFrame:
    """Fit K-Means for each k and collect inertia, metrics and fit time."""
    rows = []
    for k in k_values:
        start = time.perf_counter()
        model = fit_kmeans(X, k, seed, n_init, max_iter)
        fit_seconds = time.perf_counter() - start
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                **internal_metrics(X, model.labels_, silhouette_index),
                "fit_seconds": fit_seconds,
            }
        )
    return pd.DataFrame(rows)


def scan_hierarchy(
    X: np.ndarray, linkage_matrix: np.ndarray, k_values: list[int], silhouette_index: np.ndarray | None = None
) -> pd.DataFrame:
    """Cut one hierarchy at each k and collect metrics."""
    rows = [
        {"k": k, **internal_metrics(X, cut_tree(linkage_matrix, k), silhouette_index)} for k in k_values
    ]
    return pd.DataFrame(rows)


def reference_scan(
    X: np.ndarray,
    k_values: list[int],
    algorithm: str,
    n_datasets: int,
    seed: int,
    silhouette_index: np.ndarray,
    n_init: int = 20,
) -> pd.DataFrame:
    """Repeat the k scan on column-permuted data, which has no joint structure.

    algorithm is 'kmeans' or 'ward'. Returns one row per dataset and k.
    """
    results = []
    for dataset in range(n_datasets):
        X_reference = permute_columns(X, seed + dataset)
        if algorithm == "kmeans":
            scan = scan_kmeans(X_reference, k_values, seed + dataset, n_init, 300, silhouette_index)
        else:
            scan = scan_hierarchy(X_reference, full_ward_linkage(X_reference), k_values, silhouette_index)
        results.append(scan.assign(reference_dataset=dataset))
    return pd.concat(results, ignore_index=True)


def summarise_reference(reference: pd.DataFrame, metrics: tuple = ("silhouette", "davies_bouldin", "calinski_harabasz")) -> pd.DataFrame:
    """Mean, minimum and maximum of each metric across reference datasets, per k."""
    return reference.groupby("k")[list(metrics)].agg(["mean", "min", "max"])


def compare_to_reference(real: pd.DataFrame, reference_summary: pd.DataFrame) -> pd.DataFrame:
    """Place real metrics next to the reference range for each k."""
    rows = []
    for _, row in real.iterrows():
        k = int(row["k"])
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            band = reference_summary.loc[k, metric]
            rows.append(
                {
                    "k": k,
                    "metric": metric,
                    "real": row[metric],
                    "reference_mean": band["mean"],
                    "reference_min": band["min"],
                    "reference_max": band["max"],
                    "percent_difference_from_reference": 100 * (row[metric] - band["mean"]) / abs(band["mean"]),
                }
            )
    return pd.DataFrame(rows)


def seed_stability(X: np.ndarray, k_values: list[int], n_seeds: int) -> pd.DataFrame:
    """Mean pairwise ARI between single-initialization K-Means runs with different seeds."""
    rows = []
    for k in k_values:
        partitions = [fit_kmeans(X, k, seed, n_init=1).labels_ for seed in range(n_seeds)]
        scores = [adjusted_rand_score(a, b) for a, b in itertools.combinations(partitions, 2)]
        rows.append({"k": k, "mean_ari": np.mean(scores), "min_ari": np.min(scores)})
    return pd.DataFrame(rows)


def subsample_stability(
    X: np.ndarray, fit_labels: Callable[[np.ndarray, int, int], np.ndarray], k: int, n_subsamples: int, subsample_size: int, seed: int
) -> dict:
    """Fit on independent subsamples and compare the partitions on a shared user set.

    Each model labels the same reference users with its nearest cluster mean, so
    partitions from different subsamples can be compared with ARI.
    """
    rng = np.random.default_rng(seed)
    shared_users = rng.choice(len(X), size=subsample_size, replace=False)
    partitions = []
    for run in range(n_subsamples):
        subset = rng.choice(len(X), size=subsample_size, replace=False)
        labels = fit_labels(X[subset], k, run)
        partitions.append(assign_to_nearest_centroid(X[subset], labels, X[shared_users]))
    scores = [adjusted_rand_score(a, b) for a, b in itertools.combinations(partitions, 2)]
    return {"k": k, "mean_ari": np.mean(scores), "min_ari": np.min(scores)}


def per_cluster_silhouette(X: np.ndarray, labels: np.ndarray, silhouette_index: np.ndarray, names: dict[int, str]) -> pd.DataFrame:
    """Mean silhouette and share of users closer to another cluster, per cluster."""
    values = silhouette_samples(X[silhouette_index], labels[silhouette_index])
    table = pd.DataFrame({"label": labels[silhouette_index], "silhouette": values})
    summary = table.groupby("label")["silhouette"].agg(
        mean_silhouette="mean", share_negative=lambda s: (s < 0).mean()
    )
    summary.insert(0, "segment", summary.index.map(names))
    return summary.reset_index(drop=True)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical variables, from 0 (none) to 1 (perfect)."""
    table = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Share of a numeric variable's variance explained by a grouping (0 to 1)."""
    grouped = values.groupby(groups)
    between = (grouped.count() * (grouped.mean() - values.mean()) ** 2).sum()
    return float(between / ((values - values.mean()) ** 2).sum())


def external_association_table(
    df: pd.DataFrame, labels: np.ndarray, numeric_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    """Effect size and p-value linking cluster labels to variables not used in clustering."""
    labels = pd.Series(labels, index=df.index)
    rows = []
    for column in numeric_columns:
        valid = df[column].notna()
        groups = [df.loc[valid & (labels == label), column] for label in np.unique(labels)]
        rows.append(
            {
                "variable": column,
                "test": "One-way ANOVA",
                "effect_measure": "eta squared",
                "effect_size": eta_squared(df.loc[valid, column], labels[valid]),
                "p_value": stats.f_oneway(*groups).pvalue,
            }
        )
    for column in categorical_columns:
        table = pd.crosstab(labels, df[column])
        rows.append(
            {
                "variable": column,
                "test": "Chi-square",
                "effect_measure": "Cramer's V",
                "effect_size": cramers_v(labels, df[column]),
                "p_value": stats.chi2_contingency(table, correction=False)[1],
            }
        )
    return pd.DataFrame(rows)


def size_table(segments: pd.Series) -> pd.DataFrame:
    """Number and share of users per segment."""
    counts = segments.value_counts().rename("users")
    return pd.DataFrame({"users": counts, "percent": (100 * counts / counts.sum()).round(2)}).rename_axis("segment")


def numeric_profile(df: pd.DataFrame, segments: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Mean and median per segment, in the original units of each column."""
    return df.groupby(segments)[columns].agg(["mean", "median"]).round(2).rename_axis("segment")


def categorical_profile(df: pd.DataFrame, segments: pd.Series, column: str) -> pd.DataFrame:
    """Percentage of each category within each segment."""
    return (100 * pd.crosstab(segments, df[column], normalize="index")).round(1).rename_axis("segment")


def stability_table(
    X: np.ndarray,
    fit_labels: Callable[[np.ndarray, int, int], np.ndarray],
    algorithm: str,
    k_values: list[int],
    n_subsamples: int,
    subsample_size: int,
    seed: int,
    n_permuted_datasets: int = 3,
) -> pd.DataFrame:
    """Subsample stability per k, on the real data and on column-permuted data.

    Permuted results are averaged over several datasets, because any single
    permutation can be unusually stable or unstable by chance.
    """
    rows = []
    for k in k_values:
        real = subsample_stability(X, fit_labels, k, n_subsamples, subsample_size, seed)
        rows.append({"algorithm": algorithm, "dataset": "Real data", **real})
        permuted = [
            subsample_stability(permute_columns(X, seed + i), fit_labels, k, n_subsamples, subsample_size, seed)
            for i in range(n_permuted_datasets)
        ]
        rows.append(
            {
                "algorithm": algorithm,
                "dataset": "Permuted reference",
                "k": k,
                "mean_ari": np.mean([result["mean_ari"] for result in permuted]),
                "min_ari": np.min([result["min_ari"] for result in permuted]),
            }
        )
    return pd.DataFrame(rows)
