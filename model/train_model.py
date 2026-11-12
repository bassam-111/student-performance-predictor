"""Train and evaluate pass/fail classifiers on dataset/students.csv.

Evaluation notes (read before quoting any number):
- The bundled dataset has only 20 rows, so the 80/20 holdout test set is
  just 4 rows and a 1.0 holdout accuracy on it means very little.
- Model selection therefore uses 5-fold stratified cross-validation over
  the full dataset, which is the honest headline number. The holdout
  accuracy is still reported for completeness.
"""
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "students.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
METRICS_PLOT_PATH = os.path.join(BASE_DIR, "static", "metrics.png")


def build_preprocessor(categorical_features):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, preds).tolist()
    return {
        "model": name,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def cross_validate_model(pipeline, X, y, cv):
    """5-fold stratified CV accuracy; the honest headline metric here."""
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    return {
        "cv_accuracy_mean": round(float(scores.mean()), 4),
        "cv_accuracy_std": round(float(scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in scores],
    }


def save_confusion_matrix(confusion_matrix_values, model_name):
    matrix = np.array(confusion_matrix_values)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"])
    ax.set_yticklabels(["Fail", "Pass"])

    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, value, ha="center", va="center", color="#111827")

    fig.tight_layout()
    fig.savefig(METRICS_PLOT_PATH, dpi=160)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("pass", axis=1)
    y = df["pass"]

    categorical_features = ["gender", "parental_education"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(categorical_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    evaluations = []
    trained_models = {}

    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", clf),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        evaluation = evaluate_model(name, pipeline, X_test, y_test)
        # Cross-validation on the full dataset: the reliable comparison.
        evaluation.update(cross_validate_model(pipeline, X, y, cv))
        evaluations.append(evaluation)

    # Best model by CV accuracy; holdout accuracy breaks ties.
    best_eval = max(
        evaluations,
        key=lambda item: (item["cv_accuracy_mean"], item["accuracy"]),
    )
    best_model = trained_models[best_eval["model"]]

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset_rows": len(df),
                "holdout_test_rows": len(y_test),
                "best_model": best_eval,
                "all_models": evaluations,
            },
            file,
            indent=2,
        )

    save_confusion_matrix(best_eval["confusion_matrix"], best_eval["model"])

    print(f"Dataset: {len(df)} rows ({int(y.sum())} pass, {int((1 - y).sum())} fail)")
    print("Training complete. Best model (by 5-fold CV):", best_eval["model"])
    print(
        "CV accuracy:",
        f"{best_eval['cv_accuracy_mean']:.2%} "
        f"(+/- {best_eval['cv_accuracy_std']:.2%})",
    )
    print("Holdout accuracy (4-row test set):", f"{best_eval['accuracy']:.2%}")
    print("Confusion Matrix:")
    print(best_eval["confusion_matrix"])


if __name__ == "__main__":
    main()
