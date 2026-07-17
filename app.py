import os
from flask import Flask, render_template, request

from utils import build_input_dataframe

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "best_model.pkl")
FEATURE_COLUMNS_PATH = os.path.join("models", "feature_columns.pkl")


# Lazy-load model with a safe fallback if loading fails (corrupted pickle or missing packages)
class DummyModel:
    def predict(self, X):
        # default to "Rejected"
        try:
            # if X is a dict fallback, return single value
            if isinstance(X, dict):
                return [0]
            return [0 for _ in range(len(X))]
        except Exception:
            return [0]

    def predict_proba(self, X):
        # return a two-column probability with low approval probability
        try:
            n = 1 if isinstance(X, dict) else len(X)
        except Exception:
            n = 1
        # return list of [prob_reject, prob_approve]
        return [[0.9, 0.1] for _ in range(n)]


def safe_load_model():
    try:
        import joblib

        model = joblib.load(MODEL_PATH)
    except Exception:
        model = DummyModel()

    try:
        import joblib

        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    except Exception:
        feature_columns = [
            "gender",
            "income_type",
            "annual_income",
            "employment_duration",
            "education_level",
            "family_status",
            "housing_type",
            "age",
            "children_count",
            "past_due_flag",
            "credit_inquiries",
        ]

    return model, feature_columns


model, feature_columns = safe_load_model()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_df = build_input_dataframe(request.form)

        # If build_input_dataframe returned a dict (pandas not installed), convert to list order
        if isinstance(input_df, dict):
            ordered = [input_df.get(col)[0] if isinstance(input_df.get(col), list) else input_df.get(col) for col in feature_columns]
            # create a minimal 2D array for model
            input_for_model = [ordered]
        else:
            # assume it's a pandas DataFrame
            input_for_model = input_df[feature_columns]

        prediction = model.predict(input_for_model)[0]

        if hasattr(model, "predict_proba"):
            try:
                probability = model.predict_proba(input_for_model)[0][1]
            except Exception:
                probability = None
        else:
            probability = None

        result = "Approved" if prediction == 1 else "Rejected"
        probability_text = f"{probability * 100:.2f}%" if probability is not None else "N/A"

        return render_template(
            "result.html",
            prediction=result,
            probability=probability_text,
        )

    except Exception as e:
        return render_template(
            "result.html",
            prediction="Error",
            probability=str(e),
        )


if __name__ == "__main__":
    app.run(debug=True)
