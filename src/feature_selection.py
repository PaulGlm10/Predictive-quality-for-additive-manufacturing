"""
Forward feature selection wrapper method implementation.
"""

from typing import Tuple, List, Dict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score


def forward_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    base_model: BaseEstimator,
    scoring: str = "f1_macro",
    cv: int = 5,
    max_features: int | None = None,
    min_improvement: float = 0.001,
) -> Tuple[List[int], List[Dict]]:
    """
    Perform forward feature selection.

    Args:
        X_train: Training feature matrix (n_samples, n_features)
        y_train: Training target vector
        feature_names: List of feature names corresponding to columns
        base_model: sklearn-compatible model/pipeline to evaluate
        scoring: Scoring metric for cross-validation (default: "f1_macro")
        cv: Number of cross-validation folds (default: 5)
        max_features: Maximum number of features to select (None = no limit)
        min_improvement: Minimum improvement threshold to continue (default: 0.001)

    Returns:
        selected_indices: List of selected feature indices
        history: List of dicts containing selection history
    """
    n_features = X_train.shape[1]
    selected = []
    remaining = list(range(n_features))
    best_score = 0.0
    history = []

    iteration = 0

    while remaining and (max_features is None or len(selected) < max_features):
        iteration += 1
        best_feat = None
        best_feat_score = -np.inf

        # Try each remaining feature
        for feat in remaining:
            candidate = selected + [feat]
            X_subset = X_train[:, candidate]

            # Compute cross-validation score
            scores = cross_val_score(
                base_model, X_subset, y_train, cv=cv, scoring=scoring
            )
            score = scores.mean()

            if score > best_feat_score:
                best_feat_score = score
                best_feat = feat

        # Check if improvement is sufficient
        improvement = best_feat_score - best_score
        if improvement < min_improvement:
            # Stop if improvement is below threshold
            break

        # Add best feature to selected set
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_feat_score

        # Record history
        history.append(
            {
                "n_features": len(selected),
                "score": best_score,
                "selected_indices": selected.copy(),
                "selected_names": [feature_names[i] for i in selected],
            }
        )

    return selected, history


def run_forward_selection_for_model(
    model_name: str,
    base_model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    scoring: str = "f1_macro",
    cv: int = 5,
    max_features: int | None = None,
    min_improvement: float = 0.001,
) -> Tuple[List[int], List[Dict]]:
    """
    Run forward selection for a specific model with progress logging.

    Args:
        model_name: Name of the model (e.g., "svm", "knn", "gb")
        base_model: sklearn-compatible model/pipeline to evaluate
        X_train: Training feature matrix
        y_train: Training target vector
        feature_names: List of feature names
        scoring: Scoring metric for cross-validation
        cv: Number of cross-validation folds
        max_features: Maximum number of features to select
        min_improvement: Minimum improvement threshold

    Returns:
        selected_indices: List of selected feature indices
        history: List of dicts containing selection history
    """
    print(f"\n{'='*60}")
    print(f"Forward Selection for {model_name.upper()}")
    print(f"{'='*60}")

    selected, history = forward_selection(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        base_model=base_model,
        scoring=scoring,
        cv=cv,
        max_features=max_features,
        min_improvement=min_improvement,
    )

    # Print summary
    if history:
        print(f"\nSelection complete:")
        print(f"  - Number of features selected: {len(selected)}")
        print(f"  - Best CV score ({scoring}): {history[-1]['score']:.4f}")
        print(f"  - Selected features (first 5):")
        for i, feat_name in enumerate(history[-1]["selected_names"][:5]):
            print(f"      {i+1}. {feat_name}")
        if len(selected) > 5:
            print(f"      ... and {len(selected) - 5} more")
    else:
        print("No features selected (stopped at first iteration)")

    return selected, history

