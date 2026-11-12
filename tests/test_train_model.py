"""Tests for model training and evaluation."""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from model.train_model import (
    MODEL_PATH,
    METRICS_PATH,
    build_preprocessor,
    cross_validate_model,
    evaluate_model,
    main,
)


def _sample_frame():
    return pd.DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "parental_education": ["bachelor", "high_school", "master", "associate"],
            "study_time": [2.5, 1.5, 3.0, 2.0],
            "attendance": [88, 72, 95, 80],
            "failures": [0, 1, 0, 0],
            "previous_grade": [78, 65, 88, 74],
        }
    )


def test_preprocessor_handles_unknown_categories_and_no_nan():
    prep = build_preprocessor(["gender", "parental_education"])
    out = prep.fit_transform(_sample_frame())
    assert not np.isnan(out).any()
    # Unseen categories must not crash (handle_unknown="ignore").
    unseen = _sample_frame().iloc[[0]].copy()
    unseen.loc[unseen.index[0], "parental_education"] = "phd"
    assert prep.transform(unseen).shape[1] == out.shape[1]


def test_evaluate_model_reports_accuracy_and_confusion_matrix():
    class StubModel:
        def predict(self, X):
            return np.array([1, 0, 1, 1])

    result = evaluate_model("stub", StubModel(), None, np.array([1, 0, 1, 0]))
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["accuracy"] == 0.75
    assert np.array(result["confusion_matrix"]).shape == (2, 2)


def test_cross_validation_returns_five_fold_scores():
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv("dataset/students.csv")
    X, y = df.drop("pass", axis=1), df["pass"]
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(["gender", "parental_education"])),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    result = cross_validate_model(pipe, X, y, cv)
    assert len(result["cv_scores"]) == 5
    assert all(0.0 <= s <= 1.0 for s in result["cv_scores"])
    assert abs(result["cv_accuracy_mean"] - float(np.mean(result["cv_scores"]))) < 1e-9


def test_training_run_produces_model_and_metrics():
    main()
    assert os.path.exists(MODEL_PATH)
    with open(MODEL_PATH, "rb") as file:
        saved = pickle.load(file)
    assert hasattr(saved, "predict")

    metrics = json.load(open(METRICS_PATH))
    assert metrics["dataset_rows"] == 20
    assert metrics["holdout_test_rows"] == 4
    assert metrics["best_model"]["model"] in {
        e["model"] for e in metrics["all_models"]
    }
    for entry in metrics["all_models"]:
        assert 0.0 <= entry["cv_accuracy_mean"] <= 1.0
        assert len(entry["cv_scores"]) == 5
