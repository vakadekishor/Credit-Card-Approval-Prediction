# avoid importing numpy at module import time


def clean_column_names(df):
    # import pandas lazily to avoid build-time dependency when running the app
    import pandas as pd

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def convert_payment_status_to_binary(value):
    """
    Example converter:
    If payment status indicates overdue/high risk -> 1
    else -> 0

    Modify based on your dataset values.
    """
    import pandas as pd

    if pd.isna(value):
        return 0

    str_val = str(value).strip().lower()

    risky_values = {"1", "2", "3", "4", "5", "yes", "y", "overdue", "late", "past_due"}
    return 1 if str_val in risky_values else 0


def preprocess_target(df, target_col: str):
    """
    Converts common approval labels to binary:
    approved -> 1
    rejected -> 0
    """
    mapping = {
        "approved": 1,
        "approve": 1,
        "yes": 1,
        "y": 1,
        "1": 1,
        1: 1,
        True: 1,
        "rejected": 0,
        "reject": 0,
        "no": 0,
        "n": 0,
        "0": 0,
        0: 0,
        False: 0,
    }

    import pandas as pd

    def _map_target(x):
        try:
            key = str(x).strip().lower()
        except Exception:
            key = x
        return mapping.get(key, x)

    df[target_col] = df[target_col].map(_map_target)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def build_input_dataframe(form_data: dict):
    """
    Build a dataframe from Flask form input.
    Make sure names match the training features.
    """
    data = {
        "gender": [form_data.get("gender", "Male")],
        "income_type": [form_data.get("income_type", "Working")],
        "annual_income": [safe_float(form_data.get("annual_income", 0))],
        "employment_duration": [safe_float(form_data.get("employment_duration", 0))],
        "education_level": [form_data.get("education_level", "Secondary")],
        "family_status": [form_data.get("family_status", "Married")],
        "housing_type": [form_data.get("housing_type", "House / apartment")],
        "age": [safe_int(form_data.get("age", 30))],
        "children_count": [safe_int(form_data.get("children_count", 0))],
        "past_due_flag": [safe_int(form_data.get("past_due_flag", 0))],
        "credit_inquiries": [safe_int(form_data.get("credit_inquiries", 0))],
    }

    # Import pandas only when building the dataframe to avoid requiring pandas at import time
    try:
        import pandas as pd
        return pd.DataFrame(data)
    except Exception:
        # Fallback: return a simple dict that the caller can handle
        return data
