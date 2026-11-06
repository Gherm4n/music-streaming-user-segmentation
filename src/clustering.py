"""Fit K-Means and hierarchical models and name the resulting segments."""

import fastcluster
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

# A cluster is named after the feature on which it differs most from the average
# user, and the direction of that difference (see name_segments).
SEGMENT_NAME_RULES = {
    ("avg_listening_hours_per_week", "high"): "Heavy Listeners",
    ("avg_listening_hours_per_week", "low"): "Light Listeners",
    ("avg_skips_per_day", "high"): "Frequent Skippers",
    ("avg_skips_per_day", "low"): "Low-Skip Listeners",
    ("playlists_created", "high"): "Playlist Builders",
    ("playlists_created", "low"): "Few-Playlist Users",
    ("music_suggestion_rating_1_to_5", "high"): "High-Rating Users",
    ("music_suggestion_rating_1_to_5", "low"): "Low-Rating Users",
}


def fit_kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 20, max_iter: int = 300) -> KMeans:
    """Fit K-Means with k-means++ initialization, keeping the best of n_init runs."""
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, max_iter=max_iter, random_state=seed).fit(X)


def sample_linkage(X: np.ndarray, method: str) -> np.ndarray:
    """Standard SciPy linkage. It stores all pairwise distances, so it suits samples only."""
    return linkage(X, method=method, metric="euclidean")


def full_ward_linkage(X: np.ndarray) -> np.ndarray:
    """Ward linkage computed from the feature vectors, without a distance matrix.

    Memory grows with n rather than n squared, which makes the full dataset
    feasible. Only Ward (with single, centroid and median) has this vector form.
    """
    return fastcluster.linkage_vector(X, method="ward", metric="euclidean")


def cut_tree(linkage_matrix: np.ndarray, k: int) -> np.ndarray:
    """Cut a hierarchy into k clusters and return labels numbered from 0."""
    return fcluster(linkage_matrix, t=k, criterion="maxclust") - 1


def cut_height(linkage_matrix: np.ndarray, k: int) -> float:
    """Height of a horizontal line that cuts the dendrogram into exactly k clusters."""
    heights = linkage_matrix[:, 2]
    return float((heights[-k] + heights[-k + 1]) / 2)


def permute_columns(X: np.ndarray, seed: int) -> np.ndarray:
    """Shuffle each column independently.

    Each feature's distribution is unchanged, but any relationship between
    features is destroyed. A dataset built this way can contain no joint groups.
    """
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(X[:, column]) for column in range(X.shape[1])])


def assign_to_nearest_centroid(X_fit: np.ndarray, labels_fit: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    """Label new observations with the nearest cluster mean.

    AgglomerativeClustering has no predict method. This is a separate, explicit
    rule used only when comparing partitions fitted on different subsamples.
    """
    centroids = np.vstack([X_fit[labels_fit == label].mean(axis=0) for label in np.unique(labels_fit)])
    squared_distances = ((X_new[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return squared_distances.argmin(axis=1)


def name_segments(z_profile: pd.DataFrame) -> dict[int, str]:
    """Name each cluster using its strongest standardized deviation from the mean.

    z_profile has one row per cluster and one column per feature, holding mean
    z-scores. The rule is mechanical so that a name always traces back to data.
    """
    names = {}
    for label, row in z_profile.iterrows():
        feature = row.abs().idxmax()
        direction = "high" if row[feature] > 0 else "low"
        names[label] = SEGMENT_NAME_RULES[(feature, direction)]
    if len(set(names.values())) != len(names):
        raise ValueError(f"Two clusters received the same name: {names}")
    return names
