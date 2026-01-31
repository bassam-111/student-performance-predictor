import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "gender": request.form.get("gender"),
        "parental_education": request.form.get("parental_education"),
        "study_time": float(request.form.get("study_time")),
        "attendance": float(request.form.get("attendance")),
        "failures": int(request.form.get("failures")),
        "previous_grade": float(request.form.get("previous_grade")),
    }

    input_df = pd.DataFrame([form_data])
    prediction = model.predict(input_df)[0]

    result = "PASS 🎉" if int(prediction) == 1 else "FAIL ⚠️"
    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
