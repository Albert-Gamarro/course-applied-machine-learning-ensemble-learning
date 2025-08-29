# src/utils/common.py
import os
import yaml
import pandas as pd
import joblib
import datetime
import ast
import json


# Get project root
def get_project_root():
    current_dir = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(current_dir, "README.md")):
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            raise FileNotFoundError("Could not find project root (README.md)")
    return current_dir


def save_experiment(all_results, studies, config, results_root="results"):
    """
    Save experiment results and configuration in a reproducible way.

    Args:
        all_results (list): List of DataFrames, each containing trial results for a model.
        studies (dict): Dictionary of model_name: study object.
        config (dict): Experiment configuration dictionary.
        results_root (str): Path to root results folder (relative to project root).

    Returns:
        str: Path to the experiment results directory.
    """

    # Resolve experiment name
    experiment_name = config.get(
        "experiment_name",
        f"experiment_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    )
    print(f"Resolved experiment_name: {experiment_name}")

    # Create experiment directory
    results_dir = os.path.join(get_project_root(), results_root, experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Combine and save all trial results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv(
            os.path.join(results_dir, "optuna_results.csv"), index=False
        )
    else:
        print("Warning: No results to save.")

    # Save study objects
    for model_name, study in studies.items():
        joblib.dump(study, os.path.join(results_dir, f"{model_name}_study.pkl"))

    print(f"✅ Results saved in {results_dir}")
    return results_dir


def parse_params(param_str):
    if isinstance(param_str, str):
        try:
            # first try ast.literal_eval
            return ast.literal_eval(param_str)
        except (ValueError, SyntaxError):
            # if it fails, json.loads (in case is a json)
            try:
                return json.loads(param_str)
            except (ValueError, json.JSONDecodeError):
                # both failed, return empty dict
                return {}
    # if not a string, return empty dict
    return {}
