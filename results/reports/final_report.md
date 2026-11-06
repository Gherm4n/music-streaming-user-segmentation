# Final Report: Music Streaming User Segmentation

## What was implemented

A complete, reproducible clustering workflow on 50,000 synthetic users: six executed notebooks, reusable source code, a central configuration, 27 figures, and saved metric and profile tables.

**Algorithms tested:** K-Means (k from 2 to 10) and agglomerative hierarchical clustering. Ward, average, complete and single linkage were compared on a 5,000-user sample, and Ward was then fitted on all 50,000 users.

## Selected models

Both use k = 5, on the same users and the same four standardized features: listening hours, playlists created, skips per day and suggestion rating.

| Metric | K-Means | Ward CAH |
|---|---:|---:|
| Silhouette (all users) | 0.1858 | 0.1034 |
| Davies-Bouldin | 1.3209 | 1.6308 |
| Calinski-Harabasz | 10,674.5 | 7,238.2 |
| Subsample stability (ARI) | 0.801 | 0.350 |

Agreement between the two solutions: ARI 0.306.

## Segments

Average user: 9.99 hours per week, 8.00 playlists, 10.03 skips per day, rating 3.64.

| K-Means segment | Share | Evidence | Ward segment | Share | Evidence |
|---|---:|---|---|---:|---|
| Light Listeners | 22.94% | 6.02 hours | Low-Skip Listeners | 28.76% | 7.79 skips, 12.46 hours |
| Heavy Listeners | 22.39% | 13.68 hours | Frequent Skippers | 27.21% | 13.41 skips |
| Playlist Builders | 19.35% | 11.78 playlists | Low-Rating Users | 16.25% | Rating 2.11 |
| Frequent Skippers | 19.11% | 14.28 skips | Light Listeners | 14.50% | 6.39 hours, 5.71 playlists |
| Low-Rating Users | 16.22% | Rating 1.83 | Playlist Builders | 13.28% | 11.29 playlists |

## Main conclusions

**Answer to the research question: partly.** Both algorithms produce interpretable segments, but the evidence does not show that they are natural or meaningful groups in this dataset.

1. **The features are almost independent.** Excluding the inactivity flag, the largest correlation between numeric variables is 0.008.
2. **Scores match data with no joint structure.** Of 54 comparisons with column-shuffled datasets, 46 fall inside the shuffled range, 2 are better and 6 are worse, in line with chance. Stability and the dendrogram gap at k = 5 are matched as well.
3. **Segments do not relate to outside variables.** The largest effect size is Cramér's V = 0.0145, and none of 26 tests is significant. No segment can be tied to churn, dissatisfaction or conversion.
4. **Results depend on analyst choices.** Switching to RobustScaler or removing the rating gives ARI of about 0.46 with the baseline for K-Means, and below 0.2 for Ward.
5. **K-Means is the better choice for description.** It has better metrics, more balanced segments and greater repeatability. Ward is more robust only to changes in k, because its partitions are nested.

## Limitations

- The data is synthetic, so conclusions do not apply to real users or any real platform.
- There are no ground-truth segments.
- There is no data dictionary, so column meanings and the tenure snapshot date are assumptions.
- The reference test uses 10 datasets. It gives a range, not a precise p-value.
- The linkages were compared only on a sample.
- PCA plots show only 50.3% of the variance.

## Departures from the original plan

- **Columns:** `engaging_score` and `engaging_level` were mentioned in the brief but are not in the dataset.
- **Added reference test:** column-permutation and Gaussian-blob comparisons were added, because the EDA suggested no natural structure.
- **Added dependency:** `fastcluster` allows true CAH on all users, rather than a sample-based approximation.
- **Stability:** this was tested directly, rather than left as future work.

## Incomplete tasks

None. All six notebooks executed without errors, and the validity checks in Notebook 06 passed.
