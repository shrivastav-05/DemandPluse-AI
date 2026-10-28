from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_pipeline.pkl"

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-for-local-use"

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]
OPTIONS = {
    "Sex": {"M", "F"},
    "ChestPainType": {"TA", "ATA", "NAP", "ASY"},
    "FastingBS": {"0", "1"},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"N", "Y"},
    "ST_Slope": {"Up", "Flat", "Down"},
}

def validate_form(form):
    values = {field: form.get(field, "").strip() for field in FEATURES}
    for field, value in values.items():
        if not value:
            raise ValueError(f"Please fill in {field}.")
    for field, allowed in OPTIONS.items():
        if values[field] not in allowed:
            raise ValueError(f"Invalid option for {field}.")
    try:
        for field in ("Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR"):
            values[field] = int(values[field])
        values["Oldpeak"] = float(values["Oldpeak"])
    except (TypeError, ValueError):
        raise ValueError("Please enter valid numeric values.")
    if not 1 <= values["Age"] <= 120:
        raise ValueError("Age must be between 1 and 120.")
    if min(values["RestingBP"], values["Cholesterol"], values["MaxHR"]) < 0:
        raise ValueError("Blood pressure, cholesterol and maximum heart rate cannot be negative.")
    if values["FastingBS"] not in (0, 1):
        raise ValueError("FastingBS must be 0 or 1.")
    return values

def build_model(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical)
    ])
    return Pipeline([
        ("preprocessor", prep),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])

def predict_saved_model(values):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Saved model not found. Select Train a New Model first.")
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([{col: values[col] for col in FEATURES}], columns=FEATURES)
    pred = int(model.predict(input_df)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        if 1 in classes:
            prob = float(probs[classes.index(1)]) * 100
    return pred, prob, None

def train_model(values, testing_size, cv_folds, split_strategy):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "HeartDisease" not in df.columns:
        raise ValueError("Dataset must contain HeartDisease target column.")
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    if split_strategy == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testing_size, random_state=42, stratify=y
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testing_size, random_state=42, stratify=None
        )
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    model = build_model(X)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    input_df = pd.DataFrame([{col: values[col] for col in FEATURES}], columns=FEATURES)
    prediction = int(model.predict(input_df)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probs[classes.index(1)]) * 100

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "cv_mean": round(float(np.mean(cv_scores)) * 100, 2),
        "cv_std": round(float(np.std(cv_scores)) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "testing_size": int(testing_size * 100),
        "cv_folds": cv_folds,
        "split_strategy": split_strategy.title()
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return prediction, probability, metrics

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", values={}, error=None)

@app.route("/choose-method", methods=["POST"])
def choose_method():
    try:
        values = validate_form(request.form)
        session["form_values"] = values
        return render_template("choose_method.html", error=None)
    except ValueError as exc:
        return render_template("index.html", values=request.form, error=str(exc)), 400

@app.route("/predict-existing", methods=["POST"])
def predict_existing():
    values = session.get("form_values")
    if not values:
        return redirect(url_for("home"))
    try:
        prediction, probability, metrics = predict_saved_model(values)
        session["result"] = {
            "prediction": prediction, "probability": probability,
            "metrics": metrics, "method": "Existing Saved Model"
        }
        return redirect(url_for("result"))
    except Exception as exc:
        return render_template("choose_method.html", error=str(exc)), 400

@app.route("/training-configuration", methods=["GET", "POST"])
def training_configuration():
    if not session.get("form_values"):
        return redirect(url_for("home"))
    if request.method == "POST":
        try:
            testing_percent = int(request.form.get("testing_size", "20"))
            cv_folds = int(request.form.get("cv_folds", "5"))
            split_strategy = request.form.get("split_strategy", "stratified")
            if not 1 <= testing_percent < 100:
                raise ValueError("Testing size must be between 1% and 99%. At least 1% must remain for training.")
            if not 2 <= cv_folds <= 20:
                raise ValueError("Cross-validation folds must be between 2 and 20.")
            if split_strategy not in {"stratified", "non_stratified"}:
                raise ValueError("Choose a valid data splitting strategy.")
            # Validate the chosen CV folds against the available training rows/classes.
            dataset = pd.read_csv(DATA_PATH)
            y_all = dataset["HeartDisease"]
            approx_train_rows = int(len(dataset) * (1 - testing_percent / 100))
            if cv_folds > approx_train_rows:
                raise ValueError("CV folds cannot exceed the number of training rows.")
            if split_strategy == "stratified" and cv_folds > int(y_all.value_counts().min()):
                raise ValueError("For stratified CV, folds cannot exceed the smallest target-class count.")

            session["training_config"] = {
                "testing_size": testing_percent / 100,
                "cv_folds": cv_folds,
                "split_strategy": split_strategy
            }
            values = session["form_values"]
            config = session["training_config"]
            prediction, probability, metrics = train_model(
                values, config["testing_size"], config["cv_folds"], config["split_strategy"]
            )
            session["result"] = {
                "prediction": prediction, "probability": probability,
                "metrics": metrics, "method": "Newly Trained Model"
            }
            return redirect(url_for("result"))
        except Exception as exc:
            return render_template("training_configuration.html", error=str(exc)), 400
    return render_template("training_configuration.html", error=None)

@app.route("/train-new", methods=["POST"])
def train_new():
    return redirect(url_for("training_configuration"))

@app.route("/result", methods=["GET"])
def result():
    result_data = session.get("result")
    if not result_data:
        return redirect(url_for("home"))
    pred = int(result_data["prediction"])
    result_view = {
        "prediction": pred,
        "label": "Heart Disease: Positive (1)" if pred == 1 else "Heart Disease: Negative (0)",
        "probability": result_data.get("probability"),
        "method": result_data.get("method", "Saved Model"),
        "metrics": result_data.get("metrics")
    }
    return render_template("result.html", result=result_view)

@app.route("/again", methods=["POST"])
def again():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)
