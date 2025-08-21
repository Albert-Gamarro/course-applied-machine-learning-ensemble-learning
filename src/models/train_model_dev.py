from data.load_dataset import load_clean_adult_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


from sklearn.pipeline import Pipeline

# --------------------------------
# Load data
# --------------------------------


# Load data (X = features, y = target
X, y = load_clean_adult_dataset()

# Convert target to numeric (important for metrics like precision/recall/F1)
# '>50K' → 1 (positive class), '<=50K' → 0 (negative class)
y = y.map({">50K": 1, "<=50K": 0})

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# Preprocessor: only encode categorical variables
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
# Define pipeline
# --------------------------------
# 🚨 Why pipeline? During cross-validation, preprocessing is fit *only on training folds*
# This avoids "data leakage" (encoder accidentally learning from validation data).

rf = RandomForestClassifier(random_state=42)  # initialize the model

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])

# --------------------------------
# Hyperparameter grid
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

# --------------------------------
# Grid Search with multiple metrics
# --------------------------------

scoring = {
    "accuracy": make_scorer(accuracy_score),
    # Accuracy: proportion of correctly classified samples over all samples.
    # Good general measure but can be misleading if data is imbalanced.
    "precision": make_scorer(precision_score),
    # Precision: proportion of positive predictions that are actually positive.
    # Tells us how “trustworthy” the model’s positive predictions are.
    "recall": make_scorer(recall_score),
    # Recall (sensitivity): proportion of actual positives correctly identified.
    # Tells us how good the model is at catching all positives.
    "f1": make_scorer(f1_score),
    # F1 score: harmonic mean of precision and recall.
    # Balances false positives and false negatives; very useful for imbalanced data.
}


grid_search = GridSearchCV(
    pipeline,  # pipeline = preprocessing + model
    param_grid,  # parameters to try
    cv=5,  # 5-fold cross-validation
    scoring=scoring,  # evaluate multiple metrics
    refit="f1",  # refit final model using best F1 score
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


# Convert all CV results into a DataFrame for deeper analysis
results_df = pd.DataFrame(grid_search.cv_results_)

# Keep only the most useful columns
# mean_test_X → the average X metric score across CV folds for this parameter combo.
# std_test_X → how much the score varies across folds (i.e., stability of the model).

results_df = results_df[
    [
        "params",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "rank_test_f1",  # use rank based on main metric of interest
    ]
].sort_values(by="mean_test_f1", ascending=False)
top5 = results_df.head(5)



# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(grid_search.best_estimator_, "best_random_forest.pkl")





# --------------------------------
# Experiment: Stacking Classifier
# --------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.model_selection import cross_validate

# Define base learners
estimators = [
    ("dt", DecisionTreeClassifier(max_depth=3)),
    ("knn", KNeighborsClassifier(n_neighbors=3)),
    ("rf", RandomForestClassifier(random_state=42)),
    ("ada", AdaBoostClassifier(random_state=42)),
]

# Define stacking model
# stacking_clf = StackingClassifier(
#     estimators=estimators,
#     final_estimator=RandomForestClassifier(
#         n_estimators=50, random_state=42
#     ),  # meta-learner - defaults to LogisticRegression
#     n_jobs=-1,
# )
stacking_clf = StackingClassifier(
    estimators=estimators
)  

# Wrap inside pipeline (with preprocessing!)
stacking_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", stacking_clf)]
)

# Evaluate with cross-validation (using same metrics setup)
cv_results = cross_validate(
    stacking_pipeline, X, y, cv=3, scoring=scoring, return_train_score=True, n_jobs=-1
)

# Store results in dataframe (consistent with your winners_df logic)
stacking_results = pd.DataFrame(
    {
        "model": ["StackingClassifier"],
        "mean_test_accuracy": [cv_results["test_accuracy"].mean()],
        "mean_test_precision": [cv_results["test_precision"].mean()],
        "mean_test_recall": [cv_results["test_recall"].mean()],
        "mean_test_f1": [cv_results["test_f1"].mean()],
        "std_test_accuracy": [cv_results["test_accuracy"].std()],
        "std_test_precision": [cv_results["test_precision"].std()],
        "std_test_recall": [cv_results["test_recall"].std()],
        "std_test_f1": [cv_results["test_f1"].std()],
        "overfit": [cv_results["train_f1"].mean() - cv_results["test_f1"].mean()],
        "overfit_flg": [
            (cv_results["train_f1"].mean() - cv_results["test_f1"].mean()) > 0.05
        ],
    }
)

print(stacking_results)
