import os
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import optuna

from src.models.utils.optuna_utils import create_objective, load_class
from src.models.utils.common import save_experiment, get_project_root
from data.load_dataset import load_clean_adult_dataset  # Your dataset loader

# --------------------------------
# Configuration
# --------------------------------

config_path = os.path.join(get_project_root(), "config", "config.yaml")

# Load config from config/
# parse the YAML file into a Python dictionary.
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Models in config: {list(config['models'].keys())}")

# --------------------------------
# Load data
# --------------------------------
X, y = load_clean_adult_dataset()
y = y.map({">50K": 1, "<=50K": 0})

# Define preprocessor
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
    ],
    remainder="passthrough",
)

# Scoring metrics
scoring = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

print(f"Scoring keys defined: {list(scoring.keys())}")


# --------------------------------
# Run Optuna for each model
# --------------------------------

all_results = []  # Initialize an empty list to store results (DataFrames) for all models
studies = []

for model_name, model_config in config["models"].items():
    # model_config: Dictionary with the model class and parameter ranges

    print(f"\n🔍 Running Optuna for {model_name}...")
    print(f"Parameter ranges for {model_name}: {model_config['params']}")

    try:
        # Initialize model
        model = load_class(model_config["class"])(random_state=config["random_seed"])
        if model_name == "XGBoost":
            model.set_params(enable_categorical=True, eval_metric="logloss")

        # Create pipeline: preprocessor → model
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        # Define the pipeline with two steps:
        # 1. "preprocessor": Applies ColumnTransformer (one-hot encoding for categorical columns)
        # 2. "model": The initialized model (e.g., RandomForestClassifier)
        # This ensures data is preprocessed before being fed to the model

        # Create Optuna study and objective function
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config["random_seed"]),
        )

        # direction="maximize" means higher F1 scores are better
        objective = create_objective(
            pipeline,
            X,
            y,
            model_name,
            model_config["params"],
            scoring,
            config["cv_folds"],
        )

        # Run Optuna optimization for the specified number of trials (e.g., 20)
        study.optimize(objective, n_trials=config["optuna_trials"], n_jobs=-1)
        # Run Optuna optimization for the specified number of trials (e.g., 20)
        # n_jobs=-1 parallelizes trials across CPU cores for speed
        # Each trial calls the objective function, which:
        # - Suggests hyperparameters
        # - Updates the pipeline’s model step
        # - Runs cross-validation
        # - Returns the F1 score and stores all metrics

        # Collect results
        trials_df = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "params": trial.user_attrs["params"],
                    **trial.user_attrs["scores"],
                    "overfit": trial.user_attrs["scores"]["mean_train_f1"]
                    - trial.user_attrs["scores"]["mean_test_f1"],
                    "overfit_flg": trial.user_attrs["scores"]["mean_train_f1"]
                    - trial.user_attrs["scores"]["mean_test_f1"]
                    > 0.05,
                }
                for trial in study.trials
            ]
        )
        all_results.append(trials_df)
        studies[model_name] = study  # Store study object

        # Save best model
        best_params = study.best_trial.user_attrs["params"]
        pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})
        pipeline.fit(X, y)
        joblib.dump(
            pipeline, os.path.join(config["output_dir"], f"{model_name}_best_model.pkl")
        )

        print(f"Best params for {model_name}: {best_params}")
        print(f"Best F1 for {model_name}: {study.best_value:.4f}")

    except Exception as e:
        print(f"Error running {model_name}: {e}")
        continue
    print("\n✅ All models processed.")

# Combine and save results
results = pd.concat(all_results, ignore_index=True)

# Save config and results
results_dir = save_experiment(
    all_results, studies, config, results_root=config["results_dir"]
)

print("\n--- Step 1: Sanity Check ---")
model_counts = results["model"].value_counts()
print("Number of trials per model:\n", model_counts)


# --------------------------------
# Analysing Optuna results
# --------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import ast

plt.rcParams["figure.figsize"] = (10, 6)

experiment_name = "v1_initial_run"  # as desired
results_dir = os.path.join(get_project_root(), "results", experiment_name)
results = pd.read_csv(os.path.join(results_dir, "optuna_results.csv"))

# Convert stringified dicts to real dicts
results["params"] = results["params"].apply(ast.literal_eval)
# Parse params for readability
params_df = pd.json_normalize(results["params"])
results = pd.concat([results.drop(columns="params"), params_df], axis=1)


# --------------------------------
# Overfitting Analysis
# --------------------------------

# Train vs Test F1 Distribution
# Do models maintain similar performance on test compared to train
plt.figure(figsize=(10, 6))
final_melt = results.melt(
    id_vars=["model"],
    value_vars=["mean_train_f1", "mean_test_f1"],
    var_name="Dataset",
    value_name="F1 Score",
)
sns.boxplot(x="model", y="F1 Score", hue="Dataset", data=final_melt)
plt.title("Train vs Test F1 Distribution per Model")
plt.legend(title="Dataset")
plt.tight_layout()
plt.show()

# Overfit Distribution per Model
# How much do models overfit? Which models tend to have a smaller train–test gap (more robust) vs those exceeding our overfit threshold (riskier)?
plt.figure(figsize=(8, 5))
sns.boxplot(data=results, x="model", y="overfit")
plt.axhline(0.05, color="red", linestyle="--", label="Overfit Threshold")
plt.ylabel("Train-Test F1 Gap (Overfit)")
plt.title("Overfit Distribution per Model")
plt.legend()
plt.tight_layout()
plt.show()


# Train vs Test F1 per Trial
# How stable is train vs test performance across trials?

models = results["model"].unique()
n_models = len(models)
fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), sharey=True)
if n_models == 1:  # handle case of only one model
    axes = [axes]
for ax, model in zip(axes, models):
    df_model = results[results["model"] == model].reset_index(drop=True)

    ax.plot(
        df_model.index,
        df_model["mean_train_f1"],
        marker="o",
        linestyle="-",
        color="tab:blue",
        alpha=0.9,
        label="Train F1",
    )
    ax.plot(
        df_model.index,
        df_model["mean_test_f1"],
        marker="x",
        linestyle="--",
        color="tab:orange",
        alpha=0.9,
        label="Test F1",
    )

    ax.set_title(f"{model}: Train vs Test F1 across Trials")
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("F1 Score")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

plt.tight_layout()
plt.show()


# Scatter plot with overfit threshold line
# What’s the trade-off between performance and overfitting?
# Which models achieve high test F1 while staying below the overfit threshold
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results,
    x="mean_test_f1",
    y="overfit",
    hue="model",
    size="mean_test_accuracy",
    sizes=(50, 400),
)
plt.axhline(0.05, color="red", linestyle="--", label="Overfit Threshold")
plt.axhspan(0, 0.05, color="green", alpha=0.1, label="Safe Zone")
plt.xlabel("Test F1")
plt.ylabel("Train-Test F1 Gap (Overfit)")
plt.title("Overfit vs Test Performance")
plt.tight_layout()
plt.show()


# --------------------------------
# Model Selection
# --------------------------------

# Filter out overfitted trials s
non_overfitted = results[
    (results["overfit"] < 0.05)
    & (
        results["model"] != "RandomForest"
    )  # Exclude RandomForest due to consistent overfitting (not generalizing well)
    & (results["mean_test_precision"] > 0.78)  # Marketing: minimize false positives
].copy()
print(f"\nNumber of non-overfitted trials: {len(non_overfitted)}")

print(non_overfitted["model"].value_counts())

# Calculate robust F1
non_overfitted["robust_f1"] = (
    non_overfitted["mean_test_f1"] - 0.3 * non_overfitted["overfit"]
)


# Top 3 per model by robust F1
top_models = []
for model in non_overfitted["model"].unique():
    model_top = non_overfitted[non_overfitted["model"] == model].nlargest(
        3, "robust_f1"
    )
    top_models.append(model_top)
top_models = pd.concat(top_models).reset_index(drop=True)


# Build summary table
summary_cols = [
    "model",
    "mean_test_f1",
    "mean_test_accuracy",
    "mean_test_precision",
    "mean_test_recall",
    "overfit",
    "params",
]

non_overfitted.groupby("model")[summary_cols].describe().T


# Scatterplot: Robust F1 vs Precision

sns.scatterplot(
    data=top_models,
    x="robust_f1",
    y="mean_test_precision",
    hue="model",
    size="mean_test_accuracy",
    sizes=(50, 200),
)
plt.axhline(0.78, color="red", linestyle="--", label="Precision Threshold (business)")
plt.xlabel("Robust F1 (F1 - Overfit)")
plt.ylabel("Test Precision")
plt.title("Top Models: Robust F1 vs Precision")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

"""
Model Selection
- Non-overfitted trials: {len(non_overfitted)} (GradientBoosting/XGBoost, precision > 0.8).
- Top models: {len(top_models)} (3 per model by robust F1).
- Selected for refinement: 3 (2 XGBoost, 1 GradientBoosting).
- Criteria: High robust F1, precision > 0.8 for marketing.
- Recommendation: XGBoost leads; GradientBoosting competitive, included for refinement.
- Next: Hyperparameter analysis in `hyperparameter_analysis.ipynb`.
"""

# Select top models for refinement
selected_models = pd.concat(
    [
        top_models[top_models["model"] == "XGBoost"].nlargest(2, "robust_f1"),
        top_models[top_models["model"] == "GradientBoosting"].nlargest(1, "robust_f1"),
    ]
).reset_index(drop=True)

# Save results
selected_models.to_csv(os.path.join(results_dir, "selected_models.csv"), index=False)

# --------------------------------
# Hyperparameter Analysis


# ------------------------------


# Load Optuna studies
xgboost_study = joblib.load(os.path.join(results_dir, "XGBoost_study.pkl"))
gb_study = joblib.load(os.path.join(results_dir, "GradientBoosting_study.pkl"))


# Analysis Without Study Files
xgboost_trials = non_overfitted[non_overfitted["model"] == "XGBoost"]
xgboost_trials.columns

# Scatterplot: max_depth vs robust_f1 (XGBoost)
sns.scatterplot(
    x="max_depth",
    y="robust_f1",
    size="mean_test_precision",
    sizes=(50, 500),
    data=xgboost_trials,
)
plt.xlabel("Max Depth")
plt.ylabel("Robust F1 Score")
plt.title("XGBoost: Max Depth vs Robust F1 (Size: Precision)")
plt.savefig(
    os.path.join(results_dir, "xgboost_max_depth_vs_robust_f1.png"), bbox_inches="tight"
)
plt.show()


# Scatterplot: learning_rate vs robust_f1 (XGBoost)
sns.scatterplot(
    x="learning_rate",
    y="robust_f1",
    size="mean_test_precision",
    sizes=(50, 500),
    data=xgboost_trials,
)
plt.xlabel("Learning Rate")
plt.ylabel("Robust F1 Score")
plt.title("XGBoost: Learning Rate vs Robust F1 (Size: Precision)")
plt.savefig(
    os.path.join(results_dir, "xgboost_learning_rate_vs_robust_f1.png"),
    bbox_inches="tight",
)
plt.show()

# Scatterplot: subsample vs robust_f1 (XGBoost)

sns.scatterplot(
    x="subsample",
    y="robust_f1",
    size="mean_test_precision",
    sizes=(50, 500),
    data=xgboost_trials,
)
plt.xlabel("Subsample")
plt.ylabel("Robust F1 Score")
plt.title("XGBoost: Subsample vs Robust F1 (Size: Precision)")
plt.savefig(
    os.path.join(results_dir, "xgboost_subsample_vs_robust_f1.png"), bbox_inches="tight"
)
plt.show()

# Scatterplot: gamma vs robust_f1 (XGBoost)

sns.scatterplot(
    x="gamma",
    y="robust_f1",
    size="mean_test_precision",
    sizes=(50, 500),
    data=xgboost_trials,
)
plt.xlabel("Gamma")
plt.ylabel("Robust F1 Score")
plt.title("XGBoost: Gamma vs Robust F1 (Size: Precision)")
plt.savefig(
    os.path.join(results_dir, "xgboost_gamma_vs_robust_f1.png"), bbox_inches="tight"
)
plt.show()
