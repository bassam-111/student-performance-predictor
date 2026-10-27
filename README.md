# Student Performance Prediction System

ML-powered student performance prediction system. Train models, evaluate accuracy, and make real-time predictions via a Flask web interface.

## What it does

- Trains three scikit-learn classifiers — Random Forest, Logistic Regression, and Decision Tree — on `dataset/students.csv` to predict whether a student will pass or fail.
- Preprocesses the data with a scikit-learn pipeline: a `ColumnTransformer` with `OneHotEncoder` for categorical features, with numeric features passed through.
- Evaluates each model with the accuracy score, classification report, and confusion matrix.
- Saves the trained best model to `model/model.pkl`, the evaluation metrics to `model/metrics.json`, and a confusion-matrix plot to `static/metrics.png`.
- Serves real-time predictions through a Flask web app: fill in a student's details and get a PASS or FAIL result.

## Tech stack

- Python
- Flask — web app (`app.py`, `templates/index.html`)
- pandas and NumPy — data handling
- scikit-learn — model training, preprocessing, and evaluation
- Matplotlib — confusion-matrix plot

Full dependency list: see `requirements.txt`.

## Project structure

```
student-performance-predictor/
├── app.py               # Flask web app (entry point)
├── requirements.txt     # Python dependencies
├── dataset/
│   └── students.csv     # training data
├── model/
│   ├── train_model.py   # model training and evaluation script
│   ├── model.pkl        # trained model (generated)
│   └── metrics.json     # evaluation metrics (generated)
├── static/
│   └── metrics.png      # confusion-matrix plot (generated)
└── templates/
    └── index.html       # prediction form page
```

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the models (reads `dataset/students.csv`, writes `model/model.pkl`, `model/metrics.json`, and `static/metrics.png`):
   ```
   python model/train_model.py
   ```
3. Start the web app:
   ```
   python app.py
   ```
   Then open the URL printed in the terminal (Flask's default is http://127.0.0.1:5000/).

On the web page, enter the student's gender, parental education, study time, attendance, failures, and previous grade, then submit the form to get a PASS or FAIL prediction.

## Evaluation

The metrics recorded in `model/metrics.json` from the last training run list Logistic Regression as the best model, with accuracy 1.0 on the bundled sample dataset. The bundled `dataset/students.csv` is a small sample dataset — replace it with the official UCI Student Performance data for real-world results.

## License

MIT — see `LICENSE`.
