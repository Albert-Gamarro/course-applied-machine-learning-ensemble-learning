# train_model_multi.py  (Script 2.1)

from data.load_dataset import load_clean_adult_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Search & metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import pandas as pd

# --------------------------------
# Load data
# --------------------------------
X, y = load_clean_adult_dataset()
y = y.map({">50K": 1, "<=50K": 0})

categorical_cols = X.select_dtypes(include=["object", "category"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        )
    ],
    remainder="passthrough",
)

# --------------------------------
# Define models + parameter grids
# --------------------------------
models_and_grids = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [None, 10, 20],
            "model__max_features": ["sqrt", "log2"],
            "model__max_samples": [None, 0.8],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [1.0, 0.8],
        },
    },
}

# Scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
}

# --------------------------------
# Run GridSearch for each model
# --------------------------------
all_results = []

for model_name, config in models_and_grids.items():
    print(f"\n🔍 Running GridSearch for {model_name}...")

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", config["model"])]
    )

    grid_search = GridSearchCV(
        pipeline,
        config["params"],
        cv=3,  # keep 3 folds for speed (can be 5 in serious runs)
        scoring=scoring,
        refit="f1",  # optimize by F1
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    print(f"Best params for {model_name}: {grid_search.best_params_}")
    print(f"Best F1 for {model_name}: {grid_search.best_score_:.4f}")

    # store results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df["model"] = model_name
    all_results.append(results_df)

# Combine all models' results
final_results = pd.concat(all_results, ignore_index=True)

# Keep only useful cols
final_results = final_results[
    [
        "model",
        "params",
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "rank_test_f1",
    ]
].sort_values(by="mean_test_f1", ascending=False)

print("\n🏆 Top 10 model + parameter combos across all models:")
print(final_results.head(10))


# --------------------------------
# Visualization: Best scores per model family
# --------------------------------
import matplotlib.pyplot as plt
import seaborn as sns


summary = (
    final_results.groupby("model")[
        [
            "mean_test_accuracy",
            "mean_test_precision",
            "mean_test_recall",
            "mean_test_f1",
        ]
    ]
    .max()  # best score for each metric within each model family
    .reset_index()
)

print("\n📊 Best scores per model family:")
print(summary)

# Melt for plotting
summary_melted = summary.melt(id_vars="model", var_name="metric", value_name="score")

plt.figure(figsize=(8, 5))
sns.barplot(data=summary_melted, x="metric", y="score", hue="model")
plt.title("Best scores per model family")
plt.ylabel("Score")
plt.ylim(0, 1)  # metrics are between 0–1
plt.legend(title="Model")
plt.tight_layout()
plt.show()
