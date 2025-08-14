# --------------------------------
# 1. Imports
# --------------------------------
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --------------------------------
# 2. Load dataset
# --------------------------------
adult_data = fetch_openml(data_id=1590, as_frame=True)
X = adult_data.data
y = adult_data.target

# Convert object columns to category
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# --------------------------------
# 3. Identify categorical features
# --------------------------------
categorical_features = X.select_dtypes(include=["category"]).columns

# Preprocessing for categorical features
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# --------------------------------
# 4. Combine preprocessing in ColumnTransformer
# --------------------------------
preprocessor = ColumnTransformer(
    transformers=[("cat", categorical_transformer, categorical_features)]
)

# --------------------------------
# 5. Build full pipeline (preprocessing + model)
# --------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]
)

# --------------------------------
# 6. Train/test split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --------------------------------
# 7. Fit model and evaluate
# --------------------------------
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.3f}")
