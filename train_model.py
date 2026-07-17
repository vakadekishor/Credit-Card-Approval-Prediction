import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from utils import clean_column_names, preprocess_target, convert_payment_status_to_binary


DATA_PATH = "data/credit_card_data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)
    return df


def basic_eda(df):
    print("\nDataset shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nFirst 5 rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    os.makedirs("eda_outputs", exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["approval_status"])
    plt.title("Approval Status Distribution")
    plt.tight_layout()
    plt.savefig("eda_outputs/approval_status_countplot.png")
    plt.close()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(figsize=(14, 10), bins=20)
        plt.tight_layout()
        plt.savefig("eda_outputs/numeric_distributions.png")
        plt.close()

    if "annual_income" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="approval_status", y="annual_income", data=df)
        plt.title("Annual Income vs Approval Status")
        plt.tight_layout()
        plt.savefig("eda_outputs/income_vs_approval.png")
        plt.close()


def preprocess_data(df):
    df = df.drop_duplicates()

    # Example: convert a payment status column to binary if it exists
    if "payment_status" in df.columns:
        df["past_due_flag"] = df["payment_status"].apply(convert_payment_status_to_binary)

    # Ensure target exists
    if "approval_status" not in df.columns:
        raise ValueError(
            "Target column 'approval_status' not found in dataset. "
            "Rename your target column to 'approval_status' or update the script."
        )

    df = preprocess_target(df, "approval_status")

    # Drop rows where target became invalid
    df = df.dropna(subset=["approval_status"])
    df["approval_status"] = df["approval_status"].astype(int)

    # If some expected columns are missing, create them
    expected_columns = [
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

    for col in expected_columns:
        if col not in df.columns:
            if col in ["annual_income", "employment_duration", "age", "children_count", "past_due_flag", "credit_inquiries"]:
                df[col] = 0
            else:
                df[col] = "Unknown"

    features = expected_columns
    target = "approval_status"

    X = df[features]
    y = df[target]

    categorical_features = [
        "gender",
        "income_type",
        "education_level",
        "family_status",
        "housing_type",
    ]

    numeric_features = [
        "annual_income",
        "employment_duration",
        "age",
        "children_count",
        "past_due_flag",
        "credit_inquiries",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return X, y, preprocessor, features


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return acc


def main():
    df = load_data()
    basic_eda(df)

    X, y, preprocessor, features = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    best_model_name = None
    best_model = None
    best_score = -1

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        acc = evaluate_model(name, pipeline, X_test, y_test)

        if acc > best_score:
            best_score = acc
            best_model_name = name
            best_model = pipeline

    print(f"\nBest Model: {best_model_name} with accuracy {round(best_score, 4)}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(features, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print("\nModel and feature columns saved successfully.")


if __name__ == "__main__":
    main()
