"""Bootstrap stability of K-Means and Ward, as one plain percentage, on all users.

Each run draws 50,000 users with replacement and refits the model on them. The clusters
of the new model are paired one-to-one with the original clusters, keeping the pairing
with the most users in common. The score is the share of drawn users still together.
The same is done on shuffled data, where no groups formed by links between variables exist.

Run from the project root, after the notebooks: .venv/bin/python scripts/course_stability.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from course_metrics import project_path
from src.clustering import cut_tree, fit_kmeans, full_ward_linkage, permute_columns
from src.data_loader import load_config, load_representation

METHODS = ["kmeans", "ward"]


def matched_share(first: np.ndarray, second: np.ndarray) -> float:
    """Share of users in the same cluster, under the best one-to-one pairing of clusters."""
    table = np.zeros((first.max() + 1, second.max() + 1), dtype=np.int64)
    np.add.at(table, (first, second), 1)
    rows, columns = linear_sum_assignment(table, maximize=True)
    return table[rows, columns].sum() / len(first)


def fit_labels(X: np.ndarray, method: str, k: int, seed: int, n_init: int, max_iter: int) -> np.ndarray:
    if method == "kmeans":
        return fit_kmeans(X, k, seed, n_init, max_iter).labels_
    return cut_tree(full_ward_linkage(X), k)


def bootstrap_share(X: np.ndarray, base: np.ndarray, method: str, k: int, seed: int, n_init: int, max_iter: int) -> tuple[float, int]:
    """Refit on one bootstrap sample and compare with the original labels of the drawn users."""
    drawn = np.random.default_rng(seed).integers(0, len(X), size=len(X))
    labels = fit_labels(X[drawn], method, k, seed, n_init, max_iter)
    # Copies of a user are at distance 0, so they must share one label.
    assert pd.Series(labels).groupby(drawn).nunique().max() == 1
    users, first_position = np.unique(drawn, return_index=True)
    return matched_share(base[users], labels[first_position]), len(users)


def main() -> None:
    config = load_config()
    seed, settings = config["random_seed"], config["kmeans"]
    k, n_init, max_iter = settings["selected_k"], settings["n_init"], settings["max_iter"]
    runs = config["evaluation"]["stability_subsamples"]
    X = load_representation(config, "a").to_numpy()
    assignments = pd.read_csv(project_path(config["paths"]["profiles_dir"]) / "final_segment_assignments.csv")
    base_labels = {"kmeans": assignments["kmeans_label"].to_numpy(), "ward": assignments["ward_label"].to_numpy()}

    # A relabelled copy must score 100%, and two unrelated labelings must score far less.
    assert matched_share(base_labels["kmeans"], (base_labels["kmeans"] + 2) % k) == 1
    unrelated = np.random.default_rng(seed).permutation(base_labels["kmeans"])
    chance = {method: matched_share(base_labels[method], np.random.default_rng(seed).permutation(base_labels[method])) for method in METHODS}
    assert matched_share(base_labels["kmeans"], unrelated) < 0.5

    rows = []
    for run in range(runs):
        for method in METHODS:
            share, users = bootstrap_share(X, base_labels[method], method, k, seed + 1000 + run, n_init, max_iter)
            rows.append({"data": "real", "run": run, "method": method, "share_same_cluster": share, "users_compared": users})
        X_shuffled = permute_columns(X, seed + run)
        for method in METHODS:
            base = fit_labels(X_shuffled, method, k, seed + run, n_init, max_iter)
            share, users = bootstrap_share(X_shuffled, base, method, k, seed + 1000 + run, n_init, max_iter)
            rows.append({"data": "shuffled", "run": run, "method": method, "share_same_cluster": share, "users_compared": users})
        print(f"Run {run + 1} of {runs} done.", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(project_path(config["paths"]["metrics_dir"]) / "course_bootstrap_stability.csv", index=False)
    print(results.groupby(["method", "data"])["share_same_cluster"].agg(["mean", "min", "max"]).round(4))
    print("Users compared per run:", int(results["users_compared"].min()), "to", int(results["users_compared"].max()))
    print("Chance level (same sizes, random labels):", {method: round(value, 4) for method, value in chance.items()})


if __name__ == "__main__":
    main()
