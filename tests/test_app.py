"""Tests for the Flask prediction app, using its test client."""
import pytest

from app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def _valid_form(**overrides):
    form = {
        "gender": "female",
        "parental_education": "bachelor",
        "study_time": "2.5",
        "attendance": "88",
        "failures": "0",
        "previous_grade": "78",
    }
    form.update(overrides)
    return form


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Student Performance Prediction System" in response.data
    assert b"Predict" in response.data


def test_predict_strong_student_returns_pass(client):
    response = client.post("/predict", data=_valid_form())
    assert response.status_code == 200
    assert b"PASS" in response.data


def test_predict_weak_student_returns_fail(client):
    response = client.post(
        "/predict",
        data=_valid_form(
            gender="male",
            parental_education="high_school",
            study_time="1.0",
            attendance="58",
            failures="2",
            previous_grade="50",
        ),
    )
    assert response.status_code == 200
    assert b"FAIL" in response.data


def test_predict_rejects_non_numeric_input(client):
    response = client.post("/predict", data=_valid_form(study_time="abc"))
    assert response.status_code == 400
    assert b"valid number" in response.data


def test_predict_rejects_out_of_range_input(client):
    response = client.post("/predict", data=_valid_form(attendance="150"))
    assert response.status_code == 400
    assert b"between" in response.data


def test_predict_rejects_unknown_category(client):
    response = client.post("/predict", data=_valid_form(gender="other"))
    assert response.status_code == 400
    assert b"valid gender" in response.data
