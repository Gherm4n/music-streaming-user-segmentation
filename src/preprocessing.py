"""Cleaning checks, feature engineering and feature-matrix construction."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

AVG_DAYS_PER_MONTH = 30.44
DAYS_PER_WEEK = 7


def add_tenure_months(df: pd.DataFrame) -> pd.DataFrame:
    """Add account tenure in months, measured up to the latest signup in the data.

    The dataset documents no snapshot date, so the latest signup date is an
    assumption. Tenure is therefore relative rather than an exact account age.
    """
    result = df.copy()
    snapshot_date = result["signup_date"].max()
    result["tenure_months"] = (snapshot_date - result["signup_date"]).dt.days / AVG_DAYS_PER_MONTH
    return result


def add_skips_per_listening_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add skips per listening hour; undefined (NaN) for users with zero listening hours."""
    result = df.copy()
    hours_per_day = result["avg_listening_hours_per_week"] / DAYS_PER_WEEK
    result["skips_per_listening_hour"] = result["avg_skips_per_day"] / hours_per_day.where(hours_per_day > 0)
    return result


def find_logical_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Count rows that break common-sense rules across columns.

    These rows are reported rather than deleted: each rule is an assumption
    about what a column means, because the dataset has no data dictionary.
    """
    is_free = df["subscription_type"] == "Free"
    checks = {
        "Zero listening hours but at least one skip per day": (
            (df["avg_listening_hours_per_week"] == 0) & (df["avg_skips_per_day"] > 0),
            "Yes: both are clustering features",
        ),
        "Status 'Active' with 3+ months inactive": (
            (df["subscription_status"] == "Active") & (df["inactive_3_months_flag"] == 1),
            "No: outcome-related columns",
        ),
        "Converted from an ad but still on the Free plan": (
            is_free & (df["ad_conversion_to_subscription"] == "Yes"),
            "No: interpretation columns",
        ),
        "Paid plan but interacted with an ad": (
            ~is_free & (df["ad_interaction"] == "Yes"),
            "No: interpretation column",
        ),
        "Student plan at age 30 or older": (
            (df["subscription_type"] == "Student") & (df["age"] >= 30),
            "No: interpretation columns",
        ),
        "Months inactive exceeds account tenure": (
            df["months_inactive"] > df["tenure_months"],
            "No: outcome-related column",
        ),
    }
    rows = [
        {
            "check": name,
            "rows": int(mask.sum()),
            "percent_of_users": round(100 * mask.mean(), 2),
            "affects_clustering_inputs": effect,
        }
        for name, (mask, effect) in checks.items()
    ]
    return pd.DataFrame(rows)


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count values outside the 1.5 x IQR fences for each column."""
    rows = []
    for column in columns:
        q1, q3 = df[column].quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers = (df[column] < low) | (df[column] > high)
        rows.append(
            {
                "feature": column,
                "lower_fence": low,
                "upper_fence": high,
                "min": df[column].min(),
                "max": df[column].max(),
                "outliers": int(outliers.sum()),
                "percent": round(100 * outliers.mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def skewness_with_log1p(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compare each column's skewness before and after a log1p transform."""
    return pd.DataFrame(
        {
            "skew_raw": df[columns].skew(),
            "skew_after_log1p": np.log1p(df[columns]).skew(),
        }
    ).round(3)


def cronbach_alpha(df: pd.DataFrame) -> float:
    """Cronbach's alpha: do these columns move together like one underlying score?"""
    n_items = df.shape[1]
    item_variance = df.var().sum()
    total_variance = df.sum(axis=1).var()
    return float(n_items / (n_items - 1) * (1 - item_variance / total_variance))


def build_numeric_representation(
    df: pd.DataFrame, numeric_columns: list[str], scaler_name: str = "standard"
) -> tuple[pd.DataFrame, StandardScaler | RobustScaler]:
    """Scale the numeric features and return them indexed by user_id, with the scaler."""
    scaler = StandardScaler() if scaler_name == "standard" else RobustScaler()
    scaled = scaler.fit_transform(df[numeric_columns])
    return pd.DataFrame(scaled, columns=numeric_columns, index=df["user_id"]), scaler


def build_mixed_representation(
    df: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    """Combine standardized numeric features with 0/1 one-hot categorical features.

    The dummy columns are deliberately not standardized. A 0/1 categorical
    variable with K balanced levels has total variance of about 1 - 1/K, close
    to the variance of 1 of a single standardized numeric feature, so each
    original variable carries a similar weight in the distance calculation.
    """
    numeric, _ = build_numeric_representation(df, numeric_columns)
    encoder = OneHotEncoder(sparse_output=False, dtype=float)
    dummies = pd.DataFrame(
        encoder.fit_transform(df[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df["user_id"],
    )
    return pd.concat([numeric, dummies], axis=1)


def validate_feature_matrix(
    matrix: pd.DataFrame, expected_rows: int, expected_columns: int, forbidden_columns: list[str]
) -> dict:
    """Raise an AssertionError if the matrix is unusable; otherwise return a summary."""
    assert matrix.shape == (expected_rows, expected_columns), f"Unexpected shape: {matrix.shape}"
    assert not matrix.isna().any().any(), "Missing values remain"
    assert np.isfinite(matrix.to_numpy()).all(), "Non-finite values remain"
    assert matrix.index.is_unique, "User ids are not unique"
    leaked = [
        column
        for column in matrix.columns
        for forbidden in forbidden_columns
        if column == forbidden or column.startswith(f"{forbidden}_")
    ]
    assert not leaked, f"Forbidden columns found: {leaked}"
    return {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "missing_values": int(matrix.isna().sum().sum()),
        "all_finite": True,
        "forbidden_columns_found": 0,
    }
