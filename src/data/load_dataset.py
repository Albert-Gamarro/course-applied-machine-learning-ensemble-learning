# --------------------------------
# Imports
# --------------------------------
import pandas as pd
from sklearn.datasets import fetch_openml


# --------------------------------
# Load & minimally clean dataset
# --------------------------------
def load_clean_adult_dataset():
    """
    Loads the Adult Census Income dataset from OpenML,
    applies minimal dtype fixes, and returns X, y.
    """
    # Download dataset from OpenML
    adult_data = fetch_openml(data_id=1590, as_frame=True)  # Adult dataset
    X = adult_data.data
    y = adult_data.target

    # Ensure categorical columns are set as 'category' dtype
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    # No encoding or imputing here — keep it raw but dtype-clean
    return X, y


# --------------------------------
# Script entry point (optional)
# --------------------------------
if __name__ == "__main__":
    X, y = load_clean_adult_dataset()

    print("Dataset shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
    print("Missing values per column:\n", X.isna().sum())

    # Optionally save cleaned dataset to local file
    df = X.assign(target=y)
    df.to_parquet("data/adult_clean.parquet", index=False)
    print("✅ Saved cleaned dataset to data/adult_clean.parquet")
