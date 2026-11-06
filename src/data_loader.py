"""Load the project configuration and the raw dataset."""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict:
    """Read config/config.yaml and return it as a dictionary."""
    path = config_path or PROJECT_ROOT / "config" / "config.yaml"
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(relative_path: str) -> Path:
    """Convert a path from the config file into an absolute project path."""
    return PROJECT_ROOT / relative_path


def ensure_output_dirs(config: dict) -> None:
    """Create the output folders listed in the config if they do not exist."""
    for key in ("processed_dir", "figures_dir", "metrics_dir", "profiles_dir", "reports_dir"):
        project_path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw CSV, parsing the signup date as a real date."""
    return pd.read_csv(project_path(config["paths"]["raw_data"]), parse_dates=["signup_date"])


def load_clean_data(config: dict) -> pd.DataFrame:
    """Load the cleaned user table written by Notebook 02."""
    path = project_path(config["paths"]["processed_dir"]) / "users_clean.csv"
    return pd.read_csv(path, parse_dates=["signup_date"])


def load_representation(config: dict, name: str) -> pd.DataFrame:
    """Load a scaled feature matrix ('a' or 'b'), indexed by user_id."""
    path = project_path(config["paths"]["processed_dir"]) / f"features_representation_{name}.csv"
    return pd.read_csv(path, index_col="user_id")
