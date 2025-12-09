"""
Model pipeline builders for SVM, kNN, and Gradient Boosting classifiers.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from typing import Literal


def build_svm_pipeline(
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
) -> Pipeline:
    """
    Build a pipeline for Support Vector Machine classifier.

    Args:
        kernel: Kernel type for SVM (default: "rbf")
        C: Regularization parameter (default: 1.0)
        gamma: Kernel coefficient (default: "scale")

    Returns:
        sklearn Pipeline with StandardScaler and SVC
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)),
        ]
    )
    return pipeline


def build_knn_pipeline(
    n_neighbors: int = 5,
    weights: Literal["uniform", "distance"] = "uniform",
) -> Pipeline:
    """
    Build a pipeline for k-Nearest Neighbors classifier.

    Args:
        n_neighbors: Number of neighbors to use (default: 5)
        weights: Weight function used in prediction (default: "uniform")

    Returns:
        sklearn Pipeline with StandardScaler and KNeighborsClassifier
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)),
        ]
    )
    return pipeline


def build_gb_pipeline(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
) -> Pipeline:
    """
    Build a pipeline for Gradient Boosting classifier.

    Args:
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Learning rate (default: 0.1)
        max_depth: Maximum depth of trees (default: 3)

    Returns:
        sklearn Pipeline with StandardScaler and GradientBoostingClassifier
    """
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline

