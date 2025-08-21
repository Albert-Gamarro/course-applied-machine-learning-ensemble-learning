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

# Keep only useful cols: all mean and std metrics
final_results = final_results[
    [
        "model",
        "params",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "rank_test_f1",
    ]
].sort_values(by="mean_test_f1", ascending=False)

print("\n🏆 Top 10 model + parameter combos across all models:")
print(final_results.head(10))


# --------------------------------
# Analyzing the results
# --------------------------------

### STEP 1 - Identify the best hyperparameter set per metric ###

best_rows = []
for metric in [
    "mean_test_accuracy",
    "mean_test_precision",
    "mean_test_recall",
    "mean_test_f1",
]:
    idx = final_results[metric].idxmax()  # row index of the best model for this metric
    row = final_results.loc[idx].copy()  # copy row info
    row["metric_winner"] = metric  # add column marking which metric it won
    best_rows.append(row)

# Collect into DataFrame
winners_df = pd.DataFrame(best_rows)


### STEP 2 - Holistic analysis of these winners ###
### Build a clean table with all info we need ###

# Flatten 'params' into separate columns
params_df = pd.json_normalize(winners_df["params"])

# Merge with winners
winners_clean = pd.concat([winners_df.reset_index(drop=True), params_df], axis=1)


# Reorder columns for readability
cols_order = (
    ["metric_winner", "model"]
    + [c for c in winners_clean.columns if "mean_test" in c]
    + [c for c in winners_clean.columns if "std_test" in c]
    + [
        c
        for c in winners_clean.columns
        if c not in ["metric_winner", "model"] and "mean_" not in c and "std_" not in c
    ]
)
winners_clean = winners_clean[cols_order]

print("\n🏆 Best models per metric (full comparison):")
print(winners_clean)


# --------------------------------
# Heatmap: Performance across metrics
# --------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

metrics = [
    "mean_test_accuracy",
    "mean_test_precision",
    "mean_test_recall",
    "mean_test_f1",
]

perf_matrix = winners_clean.set_index("metric_winner")[metrics]

plt.figure(figsize=(8, 5))
sns.heatmap(
    perf_matrix, annot=True, cmap="YlGnBu", fmt=".3f", cbar_kws={"label": "Score"}
)
plt.title("Performance Heatmap (Higher is Better)")
plt.ylabel("Winning Config (per metric)")
plt.xlabel("Metric")
plt.show()

# --------------------------------
# Heatmap: Robustness (std across folds)
# --------------------------------
std_metrics = [
    "std_test_accuracy",
    "std_test_precision",
    "std_test_recall",
    "std_test_f1",
]

std_matrix = winners_clean.set_index("metric_winner")[std_metrics]

plt.figure(figsize=(8, 5))
sns.heatmap(
    std_matrix,
    annot=True,
    cmap="YlOrRd_r",
    fmt=".3f",
    cbar_kws={"label": "Std (Lower is Better)"},
)
plt.title("Robustness Heatmap (Lower is Better)")
plt.ylabel("Winning Config (per metric)")
plt.xlabel("Metric")
plt.show()
