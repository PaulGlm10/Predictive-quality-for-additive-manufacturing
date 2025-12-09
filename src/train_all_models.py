"""
Main orchestration script for training and evaluating all models.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loading import load_dataset, train_test_split_stratified
from src.models import build_svm_pipeline, build_knn_pipeline, build_gb_pipeline
from src.feature_selection import run_forward_selection_for_model
from src.evaluation import evaluate_model_on_test, summarize_results


def save_history_json(history: list, model_name: str, output_dir: str = "results") -> None:
    """Save forward selection history as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_forward_selection_history.json")
    with open(output_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {output_path}")


def plot_feature_selection_curve(history: list, model_name: str, output_dir: str = "results/plots") -> None:
    """Plot feature count vs CV score curve."""
    if not history:
        return

    os.makedirs(output_dir, exist_ok=True)

    n_features = [h["n_features"] for h in history]
    scores = [h["score"] for h in history]

    plt.figure(figsize=(8, 6))
    plt.plot(n_features, scores, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("CV Score (F1-Macro)", fontsize=12)
    plt.title(f"Forward Selection: {model_name.upper()}", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{model_name}_feature_selection_curve.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def main():
    """Main function to orchestrate the full pipeline."""
    print("=" * 80)
    print("PREDICTIVE QUALITY MONITORING - CLASSICAL ML MODELS")
    print("=" * 80)

    # 1. Load dataset
    print("\n[1/9] Loading dataset...")
    X_df, y, feature_names = load_dataset()

    # 2. Train/test split
    print("\n[2/9] Performing stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X_df, y, test_size=0.2, random_state=42
    )
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # 3. Build model pipelines
    print("\n[3/9] Building model pipelines...")
    svm_pipeline = build_svm_pipeline()
    knn_pipeline = build_knn_pipeline()
    gb_pipeline = build_gb_pipeline()
    print("  ✓ SVM pipeline created")
    print("  ✓ kNN pipeline created")
    print("  ✓ Gradient Boosting pipeline created")

    # 4. Baseline evaluations (all features)
    print("\n[4/9] Evaluating baseline models (all 96 features)...")
    results = {}

    print("  Evaluating SVM baseline...")
    results["svm_baseline"] = evaluate_model_on_test(
        svm_pipeline, X_train, y_train, X_test, y_test, selected_indices=None
    )

    print("  Evaluating kNN baseline...")
    results["knn_baseline"] = evaluate_model_on_test(
        knn_pipeline, X_train, y_train, X_test, y_test, selected_indices=None
    )

    print("  Evaluating GB baseline...")
    results["gb_baseline"] = evaluate_model_on_test(
        gb_pipeline, X_train, y_train, X_test, y_test, selected_indices=None
    )

    # 5. Forward selection per model
    print("\n[5/9] Running forward feature selection...")

    print("\n  Running forward selection for SVM...")
    selected_svm, hist_svm = run_forward_selection_for_model(
        "svm",
        svm_pipeline,
        X_train,
        y_train,
        feature_names,
        scoring="f1_macro",
        cv=5,
        max_features=None,
        min_improvement=0.001,
    )

    print("\n  Running forward selection for kNN...")
    selected_knn, hist_knn = run_forward_selection_for_model(
        "knn",
        knn_pipeline,
        X_train,
        y_train,
        feature_names,
        scoring="f1_macro",
        cv=5,
        max_features=None,
        min_improvement=0.001,
    )

    print("\n  Running forward selection for Gradient Boosting...")
    selected_gb, hist_gb = run_forward_selection_for_model(
        "gb",
        gb_pipeline,
        X_train,
        y_train,
        feature_names,
        scoring="f1_macro",
        cv=5,
        max_features=None,
        min_improvement=0.001,
    )

    # 6. Selected-feature evaluations
    print("\n[6/9] Evaluating models with selected features...")

    print("  Evaluating SVM with selected features...")
    results["svm_selected"] = evaluate_model_on_test(
        svm_pipeline, X_train, y_train, X_test, y_test, selected_indices=selected_svm
    )

    print("  Evaluating kNN with selected features...")
    results["knn_selected"] = evaluate_model_on_test(
        knn_pipeline, X_train, y_train, X_test, y_test, selected_indices=selected_knn
    )

    print("  Evaluating GB with selected features...")
    results["gb_selected"] = evaluate_model_on_test(
        gb_pipeline, X_train, y_train, X_test, y_test, selected_indices=selected_gb
    )

    # 7. Results summary
    print("\n[7/9] Generating results summary...")
    summarize_results(results)

    # 8. Save outputs
    print("\n[8/9] Saving results...")

    # Save history JSON files
    save_history_json(hist_svm, "svm")
    save_history_json(hist_knn, "knn")
    save_history_json(hist_gb, "gb")

    # Plot feature selection curves
    plot_feature_selection_curve(hist_svm, "svm")
    plot_feature_selection_curve(hist_knn, "knn")
    plot_feature_selection_curve(hist_gb, "gb")

    # 9. Print detailed classification reports
    print("\n[9/9] Detailed Classification Reports")
    print("=" * 80)

    for key in ["svm_baseline", "svm_selected", "knn_baseline", "knn_selected", "gb_baseline", "gb_selected"]:
        if key in results:
            model_name = key.split("_")[0].upper()
            variant = key.split("_")[1].capitalize()
            print(f"\n{model_name} - {variant}:")
            print("-" * 80)
            print(results[key]["classification_report"])

    print("\n" + "=" * 80)
    print("Pipeline execution complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

