"""
Model evaluation utilities for computing metrics and summarizing results.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model_on_test(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_indices: List[int] | None = None,
    average: str = "macro",
) -> Dict:
    """
    Fit the model on training data and evaluate on test data.

    Args:
        model: sklearn-compatible model/pipeline
        X_train: Training feature matrix
        y_train: Training target vector
        X_test: Test feature matrix
        y_test: Test target vector
        selected_indices: Optional list of feature indices to use (None = all features)
        average: Averaging method for multi-class metrics (default: "macro")

    Returns:
        Dictionary containing:
            - accuracy: float
            - f1: float
            - precision: float
            - recall: float
            - confusion_matrix: np.ndarray
            - classification_report: str
            - n_features_used: int
    """
    # Subset features if selected_indices provided
    if selected_indices is not None:
        X_train_subset = X_train[:, selected_indices]
        X_test_subset = X_test[:, selected_indices]
        n_features_used = len(selected_indices)
    else:
        X_train_subset = X_train
        X_test_subset = X_test
        n_features_used = X_train.shape[1]

    # Fit model
    model.fit(X_train_subset, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_subset)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average=average)
    precision = precision_score(y_test, y_pred, average=average)
    recall = recall_score(y_test, y_pred, average=average)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_features_used": n_features_used,
    }


def summarize_results(results: Dict) -> None:
    """
    Print a formatted summary table of all model results.

    Args:
        results: Dictionary with keys like "svm_baseline", "svm_selected", etc.
                Each value is a dict from evaluate_model_on_test()
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Define model names and their display names
    model_configs = [
        ("svm_baseline", "SVM", "Baseline"),
        ("svm_selected", "SVM", "Selected"),
        ("knn_baseline", "kNN", "Baseline"),
        ("knn_selected", "kNN", "Selected"),
        ("gb_baseline", "GB", "Baseline"),
        ("gb_selected", "GB", "Selected"),
    ]

    # Print header
    header = f"{'Model':<8} {'Variant':<12} {'Accuracy':<12} {'F1-Macro':<12} {'Precision-Macro':<18} {'Recall-Macro':<14} {'Features':<10}"
    print(header)
    print("-" * 80)

    # Print each result
    for key, model_name, variant in model_configs:
        if key in results:
            r = results[key]
            print(
                f"{model_name:<8} {variant:<12} "
                f"{r['accuracy']:<12.4f} {r['f1']:<12.4f} "
                f"{r['precision']:<18.4f} {r['recall']:<14.4f} "
                f"{r['n_features_used']:<10}"
            )
        else:
            print(f"{model_name:<8} {variant:<12} {'N/A':<12} {'N/A':<12} {'N/A':<18} {'N/A':<14} {'N/A':<10}")

    print("=" * 80)

