"""Logistic Regression training: 
    - Exp 1: LogisticRegression grid search logged in MLflow
    - Exp 2: Optuna + sklearn LogisticRegressionCV (smart hyperparameter search) logged in MLflow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import optuna
import mlflow
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.data_processing.data_processing import DataProcessor
from Config import KEY_COLUMN, TARGET_COLUMN

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress optuna logs

# Load training and validation sets from S3 storage
X_train = load_data_processed_from_S3("X_train")
y_train = load_data_processed_from_S3("y_train").values
X_test  = load_data_processed_from_S3("X_test")
y_test  = load_data_processed_from_S3("y_test").values

# ------------------------------------------------------------------
# Experiment 1 : LogisticRegression 
# ------------------------------------------------------------------
mlflow.set_experiment("logistic_regression_v1")

for penalty, class_weight in product(["l1", "l2"], ["balanced", None]):
    with mlflow.start_run():
        model = LogisticRegression(
            penalty=penalty,
            class_weight=class_weight,
            solver="saga", #default: lgbfs but doesn't work with l2
            max_iter=500,
            tol=1e-3,
        )
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        mlflow.log_params({"penalty": penalty, "class_weight": class_weight})
        mlflow.log_metric("roc_auc", auc)

# ------------------------------------------------------------------
# Experiment 2 : Optuna + sklearn LogisticRegressionCV — smart CV search
# ------------------------------------------------------------------
mlflow.set_experiment("logistic_regression_cv_optuna_v1")

def objective(trial):
    penalty      = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    solver_map = {
    "l1": "saga",
    "l2": "saga", 
    "elasticnet": "saga",
    None: "lbfgs"
    }

    model = LogisticRegressionCV(
        penalty=penalty,
        class_weight=class_weight,
        solver=solver_map[penalty],
        **({"l1_ratios": [trial.suggest_float("l1_ratio", 0.1, 0.9)]} if penalty == "elasticnet" else {}),
        cv=StratifiedKFold(n_splits=5),
        scoring="roc_auc",
        max_iter=500,
        tol=1e-3,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    with mlflow.start_run():
        mlflow.log_params({
            "penalty":      penalty,
            "class_weight": class_weight,
            "best_C":       model.C_[0],
            **({" l1_ratio": trial.params.get("l1_ratio")} if penalty == "elasticnet" else {})
        })
        mlflow.log_metric("roc_auc", auc)

    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Best run summary
print(f"\nBest AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

#Save result in csv
runs = mlflow.search_runs(experiment_names=["logistic_regression"])
runs.to_csv("results/logistic_regression_runs.csv", index=False)