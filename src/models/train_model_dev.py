from data.load_dataset import load_clean_adult_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformee

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

# --------------------------------
# Load data
# --------------------------------


# 1️⃣ Load data (X = features, y = target
X, y = load_clean_adult_dataset()

# 2️⃣ Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# 3️⃣ Preprocessor: only encode categorical variables
# handle_unknown="ignore" ensures it won't crash on unseen categories in validation or prod
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        )
        # We don't touch numeric columns — they pass through unchanged by default
    ],
    remainder="passthrough",  # keep numeric columns as they are
)

# --------------------------------
#  Initialize model
# --------------------------------

rf = RandomForestClassifier(random_state=42)


# --------------------------------
#  Create pipeline
# --------------------------------

# 🚨 This is critical: during cross-validation, preprocessing happens *inside each fold*
# so that the encoder never "sees" the validation data before transforming it.
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

# --------------------------------
# Grid Search: Define Parameter Grid and Run Pipeline
# --------------------------------

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 10, 20],
    "model__max_features": [
        "sqrt",
        "log2",
    ],  # how many features to consider at each split
    "model__max_samples": [
        None,
        0.5,
        0.8,
    ],  # fraction of samples for each tree (bagging)
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    pipeline,  # pipeline = preprocessing + model
    param_grid,  # parameters to try
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",  # metric to optimize
    n_jobs=-1,  # use all CPU cores
)

#  Run grid search (this may take a while depending on grid size)
grid_search.fit(X, y)


# --------------------------------
#  Print results
# --------------------------------

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)


# --------------------------------
#  Analysing results
# --------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


results = pd.DataFrame(grid_search.cv_results_)

# show mean test score and params
display(
    results[["params", "mean_test_score", "std_test_score"]].sort_values(
        by="mean_test_score", ascending=False
    )
)


# if you only varied n_estimators and max_depth
pivot_table = results.pivot(
    index="param_max_depth", columns="param_n_estimators", values="mean_test_score"
)
sns.heatmap(pivot_table, annot=True, fmt=".3f")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.show()


# 6. Save the best model
best_model = grid_search.best_estimator_
joblib.dump(grid_search.best_estimator_, "best_random_forest.pkl")
