"""The Dunn index of K-Means and Ward on all users, and why k = 2 scores best.

Dunn = shortest distance between two points of different clusters
       / largest distance between two points of the same cluster.
Both are exact on all 50,000 users: the nearest point of another cluster comes
from a k-d tree, and a cluster's diameter is reached between two vertices of its
convex hull. Run from the project root, after the notebooks:
.venv/bin/python scripts/course_dunn.py
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist, pdist

from course_metrics import project_path
from src.clustering import cut_tree, fit_kmeans
from src.data_loader import load_clean_data, load_config, load_representation


def dunn_index(X: np.ndarray, labels: np.ndarray) -> dict:
    groups = np.unique(labels)
    diameter = max(pdist(X[labels == g][ConvexHull(X[labels == g]).vertices]).max() for g in groups)
    separation = min(cKDTree(X[labels != g]).query(X[labels == g], k=1)[0].min() for g in groups)
    return {"dunn": separation / diameter, "min_inter_distance": separation, "max_diameter": diameter}


def check_on_small_sample(X: np.ndarray, labels: np.ndarray) -> None:
    """The fast computation must match the brute-force definition."""
    distances = cdist(X, X)
    same = labels[:, None] == labels[None, :]
    expected = distances[~same].min() / distances[same].max()
    assert np.isclose(dunn_index(X, labels)["dunn"], expected)


def main() -> None:
    config = load_config()
    seed, settings = config["random_seed"], config["kmeans"]
    columns = list(load_representation(config, "a").columns)
    X = load_representation(config, "a").to_numpy()
    users = load_clean_data(config)
    tree = np.load(project_path("data/processed") / "full_tree_ward.npy")
    sample = np.random.default_rng(seed).choice(len(X), 2000, replace=False)
    check_on_small_sample(X[sample], fit_kmeans(X[sample], 5, seed, 5, 300).labels_)

    rows, centroids = [], []
    for k in range(config["evaluation"]["k_min"], config["evaluation"]["k_max"] + 1):
        kmeans = fit_kmeans(X, k, seed, settings["n_init"], settings["max_iter"])
        rows.append({"method": "kmeans", "k": k, **dunn_index(X, kmeans.labels_)})
        rows.append({"method": "ward", "k": k, **dunn_index(X, cut_tree(tree, k))})
        centre = pd.DataFrame(kmeans.cluster_centers_, columns=columns).assign(k=k, cluster=range(k))
        centroids.append(centre.assign(share=np.bincount(kmeans.labels_) / len(X)))
        if k == 2:
            print(pd.crosstab(users["music_suggestion_rating_1_to_5"], kmeans.labels_, rownames=["rating"], colnames=["k = 2 cluster"]))

    metrics_dir = project_path(config["paths"]["metrics_dir"])
    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / "course_dunn.csv", index=False)
    pd.concat(centroids, ignore_index=True).to_csv(metrics_dir / "course_kmeans_centroids_by_k.csv", index=False)
    print(results.pivot(index="k", columns="method", values="dunn").round(5))
    print("One unit of each variable, in standardized units:", (1 / users[columns].std(ddof=0)).round(3).to_dict())


if __name__ == "__main__":
    main()
