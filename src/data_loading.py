"""
Data loading and preprocessing utilities for the predictive quality monitoring project.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


def load_dataset(
    csv_path: str = "dataset/All_Features_New.csv"
) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load the dataset from CSV and build a 4-class target label.

    Args:
        csv_path: Path to the CSV file containing the dataset.

    Returns:
        X_df: DataFrame of numeric features only (shape: n_samples × 96)
        y: Series of target labels ("ok", "oil", "offset", "tilt")
        feature_names: list of feature column names (strings, length 96)
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate shape
    if df.shape[0] != 200:
        print(f"Warning: Expected 200 rows, got {df.shape[0]}")

    # Identify non-feature columns
    non_feature_cols = ["Filename", "Label", "Error_Description"]

    # Extract feature columns (all columns except non-feature ones)
    feature_names = [col for col in df.columns if col not in non_feature_cols]
    X_df = df[feature_names].copy()

    # Validate feature count
    if len(feature_names) != 96:
        print(f"Warning: Expected 96 features, got {len(feature_names)}")

    # Build 4-class target label
    y = pd.Series(index=df.index, dtype=str)

    for idx in df.index:
        label = df.loc[idx, "Label"]
        error_desc = df.loc[idx, "Error_Description"]

        if label == "iO":
            y.loc[idx] = "ok"
        elif label == "niO":
            if pd.isna(error_desc):
                raise ValueError(
                    f"Row {idx}: Label is 'niO' but Error_Description is NaN"
                )
            error_desc_lower = str(error_desc).lower()
            if error_desc_lower == "oil":
                y.loc[idx] = "oil"
            elif error_desc_lower == "offset":
                y.loc[idx] = "offset"
            elif error_desc_lower == "tilt":
                y.loc[idx] = "tilt"
            else:
                raise ValueError(
                    f"Row {idx}: Unknown Error_Description '{error_desc}' for 'niO' label"
                )
        else:
            raise ValueError(f"Row {idx}: Unknown Label '{label}'")

    # Validate class distribution
    class_counts = y.value_counts()
    expected_classes = {"ok", "oil", "offset", "tilt"}
    actual_classes = set(class_counts.index)

    if actual_classes != expected_classes:
        raise ValueError(
            f"Unexpected classes: expected {expected_classes}, got {actual_classes}"
        )

    # Check if each class has 50 samples
    for cls in expected_classes:
        count = class_counts.get(cls, 0)
        if count != 50:
            print(f"Warning: Expected 50 samples for class '{cls}', got {count}")

    print(f"Loaded dataset: {X_df.shape[0]} samples, {len(feature_names)} features")
    print(f"Class distribution:\n{class_counts}")

    return X_df, y, feature_names


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified train/test split and return numpy arrays.

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test

