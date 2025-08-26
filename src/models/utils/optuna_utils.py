# optuna_utils.py
import optuna
from sklearn.model_selection import cross_validate
import importlib


def create_objective(pipeline, X, y, model_name, param_ranges, scoring, cv_folds):
    """
    Creates an Optuna objective function for hyperparameter tuning of a specific model.

    Args:
        pipeline (Pipeline): Scikit-learn pipeline with 'preprocessor' and 'model' steps.
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        model_name (str): Name of the model (e.g., 'RandomForest').
        param_ranges (dict): Hyperparameter ranges (e.g., {'n_estimators': {'type': 'int', 'low': 50, 'high': 200}}).
        scoring (dict): Dictionary of metric names to scoring functions (e.g., {'f1': f1_score}).
        cv_folds (int): Number of cross-validation folds.

    Returns:
        callable: Objective function for Optuna to optimize.
    """

    print(f"Scoring keys received in create_objective: {list(scoring.keys())}")

    def objective(trial):
        """
        Objective function for Optuna to evaluate one set of hyperparameters.

        Args:
            trial (optuna.Trial): Optuna trial object that suggests hyperparameter values.

        Returns:
            float: Mean test F1 score to optimize (higher is better).
        """
        # Initialize dictionary for hyperparameters suggested by Optuna
        params = {}

        # Suggest values for each hyperparameter based on its type
        for param, param_config in param_ranges.items():
            param_key = param.replace("model__", "")  # Remove 'model__' prefix
            if param_config["type"] == "int":
                params[param_key] = trial.suggest_int(
                    param, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_key] = trial.suggest_float(
                    param, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_key] = trial.suggest_categorical(
                    param, param_config["choices"]
                )

        # Update the pipeline's 'model' step with suggested hyperparameters
        # Example: {'n_estimators': 100} becomes {'model__n_estimators': 100}
        pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

        # Run cross-validation: preprocessor transforms X, then model trains/predicts
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv_folds,
            scoring=list(scoring.keys()),
            return_train_score=True,
            n_jobs=-1,
        )

        # Calculate mean test and train scores for each metric
        mean_scores = {
            f"mean_test_{k}": scores[f"test_{k}"].mean() for k in scoring.keys()
        }
        mean_scores.update(
            {f"mean_train_{k}": scores[f"train_{k}"].mean() for k in scoring.keys()}
        )

        # Store scores and parameters for analysis
        trial.set_user_attr("scores", mean_scores)
        trial.set_user_attr("params", params)

        # Optimize for F1 score
        return mean_scores["mean_test_f1"]

    return objective


def load_class(class_path):
    """
    Load a Python class from a string import path (e.g., 'sklearn.ensemble.RandomForestClassifier').

    Args:
        class_path (str): Import path to the class (e.g., 'sklearn.ensemble.RandomForestClassifier').

    Returns:
        type: The Python class.
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
