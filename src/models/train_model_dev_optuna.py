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
from src.models.utils.common import save_experiment
from data.load_dataset import load_clean_adult_dataset  # Your dataset loader

# --------------------------------
# Configuration
# --------------------------------


def get_project_root():
    current_dir = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(current_dir, "README.md")):
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):  # Reached filesystem root
            raise FileNotFoundError("Could not find project root (README.md)")
    return current_dir


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

# Combine and save results
final_results = pd.concat(all_results, ignore_index=True)


# Save config and results
results_dir = save_experiment(all_results, config, results_root=config["results_dir"])


# --------------------------------
# Analysing Optuna results
# --------------------------------

non_overfitted = final_results[final_results["overfit_flg"] == False]
print(f"\nNumber of non-overfitted trials: {len(non_overfitted)}")
