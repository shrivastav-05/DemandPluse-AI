
import os
import time
import joblib
import pandas as pd

from flask import Flask, render_template, request

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold,
    cross_val_score
)

from sklearn.metrics import accuracy_score, precision_score


# 1. FLASK CONFIGURATION

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "ai4i2020.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "machine_failure_model.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# 2. FEATURES - MATCH THE TRAINING NOTEBOOK

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

TARGET = "Machine failure"


# 3. HOME PAGE
# Support both url_for('home') and url_for('index')

@app.route("/", endpoint="home")
@app.route("/", endpoint="index")
def index():
    return render_template("index.html")



# 4. FORM INPUT HELPERS

def get_form_values(form):
    """Read machine inputs from the supported form field names."""

    return {
        "machine_type": (
            form.get("machine_type")
            or form.get("type")
            or form.get("Type")
            or ""
        ),

        "air_temp": (
            form.get("air_temp")
            or form.get("air_temperature")
            or form.get("Air temperature [K]")
            or ""
        ),

        "process_temp": (
            form.get("process_temp")
            or form.get("process_temperature")
            or form.get("Process temperature [K]")
            or ""
        ),

        "rotational_speed": (
            form.get("rotational_speed")
            or form.get("rotational_speed_rpm")
            or form.get("Rotational speed [rpm]")
            or ""
        ),

        "torque": (
            form.get("torque")
            or form.get("Torque [Nm]")
            or ""
        ),

        "tool_wear": (
            form.get("tool_wear")
            or form.get("tool_wear_min")
            or form.get("Tool wear [min]")
            or ""
        )
    }


def get_machine_data(form):
    """Validate machine inputs and create the model input DataFrame."""

    values = get_form_values(form)

    machine_type = str(
        values["machine_type"]
    ).strip().upper()

    if machine_type not in ["L", "M", "H"]:
        raise ValueError(
            "Select a valid machine type: L, M, or H."
        )

    raw_numeric = {
        "Air temperature [K]": values["air_temp"],
        "Process temperature [K]": values["process_temp"],
        "Rotational speed [rpm]": values["rotational_speed"],
        "Torque [Nm]": values["torque"],
        "Tool wear [min]": values["tool_wear"]
    }

    numeric_values = {}

    for feature, raw_value in raw_numeric.items():
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Please enter a valid number for {feature}."
            )

        if not pd.notna(number) or abs(number) == float("inf"):
            raise ValueError(
                f"Invalid value for {feature}."
            )

        numeric_values[feature] = number

    if numeric_values["Air temperature [K]"] <= 0:
        raise ValueError("Air temperature must be greater than zero.")

    if numeric_values["Process temperature [K]"] <= 0:
        raise ValueError("Process temperature must be greater than zero.")

    if numeric_values["Rotational speed [rpm]"] <= 0:
        raise ValueError("Rotational speed must be greater than zero.")

    if numeric_values["Torque [Nm]"] < 0:
        raise ValueError("Torque cannot be negative.")

    if numeric_values["Tool wear [min]"] < 0:
        raise ValueError("Tool wear cannot be negative.")

    machine_data = pd.DataFrame(
        [{
            "Type": machine_type,
            **numeric_values
        }],
        columns=FEATURES
    )

    return machine_data


def load_saved_model():
    """Load the pipeline saved by the notebook or Flask training."""

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run the training notebook "
            "or choose Train New Model."
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise ValueError("The saved file is not a valid model.")

    return model


def get_training_settings(form):
    """Validate training settings submitted by the HTML form."""

    test_size_raw = (
        form.get("test_size")
        or form.get("testing_size")
        or form.get("test_percentage")
        or ""
    )

    folds_raw = (
        form.get("cv_folds")
        or form.get("folds")
        or ""
    )

    strategy = (
        form.get("split_strategy")
        or form.get("splitting_strategy")
        or "stratified"
    ).strip().lower()

    try:
        test_size = float(test_size_raw)
        cv_folds = int(folds_raw)
    except (TypeError, ValueError):
        raise ValueError(
            "Enter a valid testing size and CV fold count."
        )

    # Accept 20 or 0.20 for a 20% test split.
    if test_size > 1:
        test_size /= 100

    if not 0.01 <= test_size <= 0.99:
        raise ValueError(
            "Testing size must be between 1% and 99%."
        )

    if not 2 <= cv_folds <= 10:
        raise ValueError(
            "CV folds must be between 2 and 10."
        )

    if strategy in ["random", "non-stratified", "non_stratified"]:
        strategy = "random"
    else:
        strategy = "stratified"

    return test_size, cv_folds, strategy


# 5. PREDICTION / MODEL CHOICE PAGE

@app.route("/predict", methods=["POST"])
def predict():
    try:
        get_machine_data(request.form)

        return render_template(
            "model_choice.html",
            **get_form_values(request.form)
        )

    except ValueError as error:
        return render_template(
            "index.html",
            error_message=str(error)
        ), 400


@app.route("/model-choice", methods=["POST"])
def model_choice():
    return render_template(
        "model_choice.html",
        **get_form_values(request.form)
    )


# 6. EXISTING MODEL OPTION


@app.route("/existing-model", methods=["POST"])
def existing_model():
    try:
        # Confirm that the saved model is available.
        load_saved_model()

        return render_template(
            "existing_model_confirm.html",
            **get_form_values(request.form)
        )

    except Exception as error:
        app.logger.exception("Existing model is unavailable.")

        return render_template(
            "model_unavailable.html",
            message="The existing model is unavailable.",
            error_message=str(error),
            show_train_button=True,
            **get_form_values(request.form)
        )


# 7. PREDICT USING EXISTING SAVED MODEL

@app.route(
    "/confirm-existing-prediction",
    methods=["POST"]
)
def confirm_existing_prediction():
    try:
        machine_data = get_machine_data(request.form)
        model = load_saved_model()

        # Existing model is loaded, not retrained.
        prediction = int(model.predict(machine_data)[0])

        return render_template(
            "result.html",
            prediction=prediction,
            method="Existing Model"
        )

    except Exception as error:
        app.logger.exception("Existing model prediction failed.")

        return render_template(
            "model_unavailable.html",
            message="Prediction could not be completed.",
            error_message=str(error),
            show_train_button=True,
            **get_form_values(request.form)
        )


# 8. TRAIN NEW MODEL PAGE

@app.route("/train", methods=["POST"])
def train():
    return render_template(
        "training_config.html",
        **get_form_values(request.form)
    )


# 9. VALIDATE SETTINGS AND OPEN LOADING PAGE

@app.route("/training-loading", methods=["POST"])
def training_loading():
    try:
        get_machine_data(request.form)
        get_training_settings(request.form)

        return render_template(
            "loading.html",
            form_data=request.form.to_dict()
        )

    except ValueError as error:
        return render_template(
            "training_config.html",
            error_message=str(error),
            **get_form_values(request.form)
        ), 400


# 10. TRAIN, EVALUATE, SAVE AND PREDICT

@app.route("/start-training", methods=["POST"])
def start_training():
    try:
        # A. Validate machine inputs and settings
        machine_data = get_machine_data(request.form)

        test_size, cv_folds, strategy = (
            get_training_settings(request.form)
        )

        # B. Load the dataset
        if not os.path.isfile(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}"
            )

        df = pd.read_csv(DATA_PATH)

        required_columns = FEATURES + [TARGET]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing dataset columns: "
                + ", ".join(missing_columns)
            )

        df = df[required_columns].dropna()

        X = df[FEATURES]
        y = df[TARGET]

        if y.nunique() != 2:
            raise ValueError(
                "Machine failure target must contain two classes."
            )

        # C. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if strategy == "stratified" else None
        )

        # D. Validate CV fold count
        if cv_folds > len(X_train):
            raise ValueError(
                "CV folds cannot exceed the number of training rows."
            )

        if strategy == "stratified":
            class_counts = y_train.value_counts()

            if class_counts.min() < cv_folds:
                raise ValueError(
                    "Not enough minority-class samples for these "
                    "CV folds. Reduce folds or change test size."
                )

            cv = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=42
            )
        else:
            cv = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=42
            )

        # E. Build the preprocessing and model pipeline
        # Same features and model setup as the notebook.
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Type"]
                )
            ],
            remainder="passthrough"
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1
                    )
                )
            ]
        )

        # F. Cross-validation
        start_time = time.time()

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=1
        )

        # G. Fit the final model
        model.fit(X_train, y_train)

        # H. Evaluate on held-out test data
        test_predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        # I. Save the fitted model
        joblib.dump(model, MODEL_PATH)

        training_time = round(
            time.time() - start_time, 2
        )

        # J. Predict the machine entered by the user
        machine_prediction = int(
            model.predict(machine_data)[0]
        )

        return render_template(
            "result.html",
            prediction=machine_prediction,
            method="Newly Trained Model",
            accuracy=round(accuracy * 100, 2),
            precision=round(precision * 100, 2),
            cv_accuracy=round(cv_scores.mean() * 100, 2),
            training_time=training_time,
            train_size=round((1 - test_size) * 100, 2),
            test_size=round(test_size * 100, 2),
            cv_folds=cv_folds
        )

    except FileNotFoundError as error:
        app.logger.exception("Required file was not found.")

        return render_template(
            "training_config.html",
            error_message=str(error),
            **get_form_values(request.form)
        ), 400

    except ValueError as error:
        return render_template(
            "training_config.html",
            error_message=str(error),
            **get_form_values(request.form)
        ), 400

    except Exception:
        app.logger.exception("Model training failed.")

        return render_template(
            "training_config.html",
            error_message=(
                "Training failed. Check the VS Code terminal "
                "for the detailed error."
            ),
            **get_form_values(request.form)
        ), 500


# 11. RUN APPLICATION
if __name__ == "__main__":
    app.run(debug=True)
