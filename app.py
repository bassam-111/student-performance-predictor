import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

GENDERS = ("female", "male")
PARENTAL_EDUCATION = ("high_school", "associate", "bachelor", "master")
# Valid ranges, matching the limits declared on the form.
NUMERIC_RANGES = {
    "study_time": (0.0, 10.0),
    "attendance": (0.0, 100.0),
    "failures": (0, 5),
    "previous_grade": (0.0, 100.0),
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found at model/model.pkl. "
            "Train it first with: python model/train_model.py"
        )
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()


@app.route("/")
def home():
    return render_template("index.html")


def parse_form(form):
    """Validate the prediction form. Returns (data, error_message)."""
    gender = form.get("gender")
    parental_education = form.get("parental_education")
    if gender not in GENDERS:
        return None, "Please choose a valid gender."
    if parental_education not in PARENTAL_EDUCATION:
        return None, "Please choose a valid parental education level."

    data = {"gender": gender, "parental_education": parental_education}
    for field, (low, high) in NUMERIC_RANGES.items():
        raw = form.get(field)
        try:
            value = float(raw) if field != "failures" else int(raw)
        except (TypeError, ValueError):
            return None, f"Please enter a valid number for {field.replace('_', ' ')}."
        if not low <= value <= high:
            return None, (
                f"{field.replace('_', ' ').capitalize()} must be between "
                f"{low:g} and {high:g}."
            )
        data[field] = value
    return data, None


@app.route("/predict", methods=["POST"])
def predict():
    form_data, error = parse_form(request.form)
    if error is not None:
        return render_template("index.html", error_text=error), 400

    input_df = pd.DataFrame([form_data])
    prediction = model.predict(input_df)[0]

    result = "PASS 🎉" if int(prediction) == 1 else "FAIL ⚠️"
    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    # Debug mode stays off: never run a public-facing app with the
    # debugger/reloader enabled.
    app.run(host="127.0.0.1", port=5000)
