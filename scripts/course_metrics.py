"""K-Means and CAH on all users, measured only with the course's indices.

Elbow (inertia), silhouette, R2 = 1 - I_intra / I_total, the Ward jump and the CPCC.
Run from the project root, after the notebooks: .venv/bin/python scripts/course_metrics.py

Complete and average linkage need the full distance matrix (10 GB for 50,000 users).
Each one runs in its own process under a memory limit, so a shortage fails cleanly.
"""

import itertools
import resource
import subprocess
import sys
import time
from pathlib import Path

import fastcluster
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_samples

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.clustering import cut_tree, fit_kmeans
from src.data_loader import load_config, load_representation, project_path

MEMORY_LIMIT_BYTES = 12 * 1024**3
SILHOUETTE_CHUNK = 500
PAIR_BLOCK = 2000
LINKAGE_METHODS = ["single", "complete", "average", "ward"]


def tree_path(method: str) -> Path:
    return project_path("data/processed") / f"full_tree_{method}.npy"


def linkage_in_place(distances: np.ndarray, method: str) -> np.ndarray:
    """Run fastcluster on a condensed matrix, overwriting it.

    fastcluster.linkage first copies its input (fastcluster 1.3.0 with NumPy 2),
    which would need a second 10 GB here, so this calls its routine directly.
    """
    n = int(np.ceil(np.sqrt(len(distances) * 2)))
    tree = np.empty((n - 1, 4))
    fastcluster.linkage_wrap(n, distances, tree, fastcluster.mthidx[method])
    return tree


def fit_tree_with_matrix(method: str) -> None:
    """Fit a linkage that needs the distance matrix, under a memory limit."""
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    X = load_representation(load_config(), "a").to_numpy()
    start = time.perf_counter()
    tree = linkage_in_place(pdist(X), method)
    np.save(tree_path(method), tree)
    print(f"{method}: {time.perf_counter() - start:.1f} s", flush=True)


def fit_all_trees(X: np.ndarray) -> dict[str, float]:
    """Fit each linkage on all users, and return the seconds taken."""
    seconds = {}
    for method in ["ward", "single"]:
        start = time.perf_counter()
        np.save(tree_path(method), fastcluster.linkage_vector(X, method=method, metric="euclidean"))
        seconds[method] = time.perf_counter() - start
    for method in ["complete", "average"]:
        start = time.perf_counter()
        subprocess.run([sys.executable, __file__, "--fit", method], check=True)
        seconds[method] = time.perf_counter() - start
    return seconds


def silhouettes_for_all_labelings(X: np.ndarray, labelings: dict) -> dict:
    """Per-user silhouette for many labelings, using each block of distances once.

    Follows the course formula, with s(i) = 0 for a user alone in a cluster.
    """
    n = len(X)
    one_hot = {key: csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, labels.max() + 1)) for key, labels in labelings.items()}
    sizes = {key: np.bincount(labels) for key, labels in labelings.items()}

    def process(rows: np.ndarray) -> dict:
        distances = cdist(X[rows], X)
        result = {}
        for key, labels in labelings.items():
            totals = (one_hot[key].T @ distances.T).T
            own = labels[rows]
            position = np.arange(len(rows))
            size = sizes[key][own]
            a = totals[position, own] / np.maximum(size - 1, 1)
            means = totals / sizes[key]
            means[position, own] = np.inf
            b = means.min(axis=1)
            values = np.where(size > 1, (b - a) / np.maximum(a, b), 0.0)
            result[key] = values
        return result

    chunks = [np.arange(start, min(start + SILHOUETTE_CHUNK, n)) for start in range(0, n, SILHOUETTE_CHUNK)]
    parts = Parallel(n_jobs=8, prefer="threads")(delayed(process)(rows) for rows in chunks)
    return {key: np.concatenate([part[key] for part in parts]) for key in labelings}


def leaf_order_and_ranges(tree: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order the users so that every cluster of the tree is one contiguous range."""
    n = len(tree) + 1
    order = np.empty(n, dtype=np.int64)
    position = 0
    stack = [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n:
            order[position] = node
            position += 1
        else:
            left, right = tree[node - n, :2].astype(np.int64)
            stack.extend([right, left])
    starts = np.empty(2 * n - 1, dtype=np.int64)
    stops = np.empty(2 * n - 1, dtype=np.int64)
    starts[order] = np.arange(n)
    stops[order] = np.arange(n) + 1
    for index in range(n - 1):
        left, right = tree[index, :2].astype(np.int64)
        starts[n + index] = min(starts[left], starts[right])
        stops[n + index] = max(stops[left], stops[right])
    return order, np.column_stack([starts, stops])


def cpcc(X: np.ndarray, tree: np.ndarray) -> float:
    """Pearson correlation between d(i, j) and the cophenetic distance c(i, j).

    All users in A and B first meet at the merge of A and B, so c(i, j) equals that
    merge's height for each such pair. This visits every pair once without storing them.
    """
    n = len(X)
    order, ranges = leaf_order_and_ranges(tree)
    ordered = X[order]

    def process(merges: np.ndarray) -> np.ndarray:
        totals = np.zeros(5)  # sum d, sum d^2, sum c, sum c^2, sum d*c
        for index in merges:
            left, right, height = int(tree[index, 0]), int(tree[index, 1]), tree[index, 2]
            a_start, a_stop = ranges[left]
            b_start, b_stop = ranges[right]
            if a_stop - a_start > b_stop - b_start:
                a_start, a_stop, b_start, b_stop = b_start, b_stop, a_start, a_stop
            pairs = (a_stop - a_start) * (b_stop - b_start)
            sum_d = sum_d2 = 0.0
            for block in range(a_start, a_stop, PAIR_BLOCK):
                distances = cdist(ordered[block : min(block + PAIR_BLOCK, a_stop)], ordered[b_start:b_stop])
                sum_d += distances.sum()
                sum_d2 += (distances**2).sum()
            totals += [sum_d, sum_d2, pairs * height, pairs * height**2, sum_d * height]
        return totals

    batches = np.array_split(np.arange(n - 1), 64)
    sum_d, sum_d2, sum_c, sum_c2, sum_dc = np.sum(Parallel(n_jobs=8, prefer="threads")(delayed(process)(batch) for batch in batches), axis=0)
    pairs = n * (n - 1) / 2
    covariance = sum_dc - sum_d * sum_c / pairs
    return float(covariance / np.sqrt((sum_d2 - sum_d**2 / pairs) * (sum_c2 - sum_c**2 / pairs)))


def within_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    return sum(float(((X[labels == label] - X[labels == label].mean(axis=0)) ** 2).sum()) for label in np.unique(labels))


def check_against_libraries(X: np.ndarray, seed: int, sample_size: int) -> None:
    """Both custom calculations must match SciPy and scikit-learn on a sample."""
    sample = X[np.random.default_rng(seed).choice(len(X), size=sample_size, replace=False)]
    for method in LINKAGE_METHODS:
        tree = linkage(sample, method=method)
        assert abs(cpcc(sample, tree) - cophenet(tree, pdist(sample))[0]) < 1e-9, method
        labels = cut_tree(tree, 5)
        ours = silhouettes_for_all_labelings(sample, {"x": labels})["x"]
        assert np.allclose(ours, silhouette_samples(sample, labels)), method
        assert np.allclose(linkage_in_place(pdist(sample), method)[:, 2], tree[:, 2]), method
    print("Silhouette, CPCC and in-place linkage match scikit-learn and SciPy on the sample.", flush=True)


def main() -> None:
    config = load_config()
    seed = config["random_seed"]
    k_values = list(range(config["evaluation"]["k_min"], config["evaluation"]["k_max"] + 1)) if "k_min" in config["evaluation"] else list(range(2, 11))
    settings = config["kmeans"]
    metrics_dir = project_path(config["paths"]["metrics_dir"])
    X = load_representation(config, "a").to_numpy()
    total_inertia = float(((X - X.mean(axis=0)) ** 2).sum())

    check_against_libraries(X, seed, config["hierarchical"]["sample_size"])

    seconds = fit_all_trees(X)
    trees = {method: np.load(tree_path(method)) for method in LINKAGE_METHODS}
    print("Trees fitted:", {method: round(value, 1) for method, value in seconds.items()}, flush=True)

    labelings, inertias = {}, {}
    for k in k_values:
        model = fit_kmeans(X, k, seed, settings["n_init"], settings["max_iter"])
        labelings[("kmeans", k)] = model.labels_
        inertias[("kmeans", k)] = model.inertia_
        for method, tree in trees.items():
            labels = cut_tree(tree, k)
            labelings[(method, k)] = labels
            inertias[(method, k)] = within_inertia(X, labels)
    print("Partitions ready.", flush=True)

    silhouettes = silhouettes_for_all_labelings(X, labelings)
    rows = []
    for (method, k), labels in labelings.items():
        shares = np.bincount(labels) / len(labels)
        rows.append(
            {
                "method": method,
                "k": k,
                "inertia": inertias[(method, k)],
                "r2": 1 - inertias[(method, k)] / total_inertia,
                "silhouette_all_users": silhouettes[(method, k)].mean(),
                "smallest_cluster_share": shares.min(),
                "largest_cluster_share": shares.max(),
                "smallest_cluster_users": int(np.bincount(labels).min()),
                "largest_cluster_users": int(np.bincount(labels).max()),
            }
        )
    pd.DataFrame(rows).to_csv(metrics_dir / "course_full_data_scan.csv", index=False)

    per_cluster = []
    for method in ["kmeans", "ward"]:
        labels, values = labelings[(method, 5)], silhouettes[(method, 5)]
        for label in np.unique(labels):
            per_cluster.append({"method": method, "label": label, "mean_silhouette": values[labels == label].mean(), "share_negative": (values[labels == label] < 0).mean()})
    pd.DataFrame(per_cluster).to_csv(metrics_dir / "course_silhouette_by_cluster.csv", index=False)

    print("Silhouettes ready.", flush=True)
    summary = [{"linkage": method, "cpcc_all_users": cpcc(X, tree), "fit_seconds": seconds[method]} for method, tree in trees.items()]
    pd.DataFrame(summary).to_csv(metrics_dir / "course_cpcc_all_users.csv", index=False)

    heights = trees["ward"][:, 2]
    jumps = heights**2 / 2  # SciPy displays sqrt(2 * delta I), as the course notes.
    pd.DataFrame(
        {
            "clusters_before_merge": range(2, 12),
            "height": heights[::-1][:10],
            "delta_inertia": jumps[::-1][:10],
        }
    ).to_csv(metrics_dir / "course_ward_jumps.csv", index=False)
    print(f"Huygens check: sum of Ward jumps = {jumps.sum():.3f}, total inertia = {total_inertia:.3f}", flush=True)

    starts = [fit_kmeans(X, settings["selected_k"], start, n_init=1, max_iter=settings["max_iter"]).inertia_ for start in range(settings["n_init"])]
    pd.DataFrame({"start": range(len(starts)), "inertia": starts}).to_csv(metrics_dir / "course_kmeans_starts.csv", index=False)
    print(f"Single starts at k = 5: inertia {min(starts):.1f} to {max(starts):.1f}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fit":
        fit_tree_with_matrix(sys.argv[2])
    else:
        main()
