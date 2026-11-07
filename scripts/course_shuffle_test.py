"""The shuffling test on all users, measured only with the course's indices.

Each column is shuffled on its own, which destroys every link between variables.
K-Means and Ward then run on each shuffled dataset, and silhouette, R2 and the Ward
jump are compared with the real data. Run from the project root, after course_metrics.py:
.venv/bin/python scripts/course_shuffle_test.py
"""

import numpy as np
import pandas as pd

from course_metrics import silhouettes_for_all_labelings, within_inertia
from src.clustering import cut_tree, fit_kmeans, full_ward_linkage, permute_columns
from src.data_loader import load_config, load_representation, project_path

MERGES_KEPT = 12


def jump_ratios(tree: np.ndarray) -> pd.DataFrame:
    """For each k, the cost of going from k to k-1 clusters over the cost of reaching k."""
    jumps = (tree[:, 2] ** 2 / 2)[::-1][:MERGES_KEPT]  # SciPy displays sqrt(2 * delta I).
    return pd.DataFrame({"k": range(2, MERGES_KEPT + 1), "delta_inertia": jumps[:-1], "ratio": jumps[:-1] / jumps[1:]})


def position_counts(real: pd.Series, low: pd.Series, high: pd.Series) -> dict:
    return {"inside": int(((real >= low) & (real <= high)).sum()), "above": int((real > high).sum()), "below": int((real < low).sum())}


def main() -> None:
    config = load_config()
    seed, datasets = config["random_seed"], config["evaluation"]["reference_datasets"]
    settings = config["kmeans"]
    k_values = list(range(config["evaluation"]["k_min"], config["evaluation"]["k_max"] + 1))
    metrics_dir = project_path(config["paths"]["metrics_dir"])
    X = load_representation(config, "a").to_numpy()

    scan_rows, jump_rows = [], [jump_ratios(np.load(project_path("data/processed") / "full_tree_ward.npy")).assign(dataset="real")]
    for dataset in range(datasets):
        X_shuffled = permute_columns(X, seed + dataset)
        total = float(((X_shuffled - X_shuffled.mean(axis=0)) ** 2).sum())
        tree = full_ward_linkage(X_shuffled)
        jump_rows.append(jump_ratios(tree).assign(dataset=dataset))
        labelings, inertias = {}, {}
        for k in k_values:
            model = fit_kmeans(X_shuffled, k, seed + dataset, settings["n_init"], settings["max_iter"])
            labelings[("kmeans", k)], inertias[("kmeans", k)] = model.labels_, model.inertia_
            labelings[("ward", k)] = cut_tree(tree, k)
            inertias[("ward", k)] = within_inertia(X_shuffled, labelings[("ward", k)])
        silhouettes = silhouettes_for_all_labelings(X_shuffled, labelings)
        for (method, k), values in silhouettes.items():
            scan_rows.append({"dataset": dataset, "method": method, "k": k, "r2": 1 - inertias[(method, k)] / total, "silhouette_all_users": values.mean()})
        print(f"Shuffled dataset {dataset + 1} of {datasets} done.", flush=True)

    scan = pd.DataFrame(scan_rows)
    jumps = pd.concat(jump_rows, ignore_index=True)
    scan.to_csv(metrics_dir / "course_shuffle_scan.csv", index=False)
    jumps.to_csv(metrics_dir / "course_shuffle_jumps.csv", index=False)

    real = pd.read_csv(metrics_dir / "course_full_data_scan.csv")
    bands = scan.groupby(["method", "k"])[["silhouette_all_users", "r2"]].agg(["min", "max"])
    for method in ["kmeans", "ward"]:
        rows = real[real["method"] == method].set_index("k")
        band = bands.loc[method]
        for metric in ["silhouette_all_users", "r2"]:
            print(method, metric, position_counts(rows[metric], band[metric]["min"], band[metric]["max"]))
            print(f"  k = 5: real {rows.loc[5, metric]:.4f}, shuffled {band.loc[5, (metric, 'min')]:.4f} to {band.loc[5, (metric, 'max')]:.4f}")
    shuffled = jumps[jumps["dataset"] != "real"]
    print("Ward ratio at k = 5, shuffled:", shuffled[shuffled["k"] == 5]["ratio"].round(2).tolist())
    print("k with the largest ratio, per dataset:", shuffled.loc[shuffled.groupby("dataset")["ratio"].idxmax()]["k"].tolist())
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
