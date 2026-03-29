"""File where the training of the logistic regression is performed"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import optuna
import mlflow
from itertools import product
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, KFold
from sklearn.metrics import roc_auc_score
from src.data_processing.data_processing import DataProcessor
from Config import KEY_COLUMN, TARGET_COLUMN

# Pre-process training data
train_processor = DataProcessor(mode="training")
df_train = train_processor.run(optional_normalisation=True)

# Pre-process test data reusing train encoder (no leakage)
test_processor = DataProcessor(mode="validation")
test_processor.global_means          = train_processor.global_means
test_processor.target_encoding_maps  = train_processor.target_encoding_maps
test_processor.ordinal_maps          = train_processor.ordinal_maps
test_processor.encoded_columns       = train_processor.encoded_columns
test_processor.scaler                = train_processor.scaler
test_processor.normalized_columns    = train_processor.normalized_columns
df_test = test_processor.run_transform(optional_normalisation=True)

# Supervised datasets
X_train = df_train.drop(columns=[TARGET_COLUMN, KEY_COLUMN])
y_train = df_train[TARGET_COLUMN]
X_test  = df_test.drop(columns=[TARGET_COLUMN, KEY_COLUMN])
y_test  = df_test[TARGET_COLUMN]

# Solver map (required by sklearn per penalty)
solver_map = {
    "l1": "liblinear",
    "l2": "lbfgs",
    "elasticnet": "saga",
    None: "lbfgs"
}

# Experiment 1: LogisticRegression
params_grid = {
    "penalty":      ["l1", "l2", "elasticnet", None],
    "class_weight": ["balanced", None]
}

mlflow.set_experiment("logistic_regression")
for penalty, class_weight in product(*params_grid.values()):
    with mlflow.start_run():
        model = LogisticRegression(
            penalty=penalty,
            class_weight=class_weight,
            solver=solver_map[penalty],
            l1_ratio=0.5 if penalty == "elasticnet" else None,
            max_iter=1000
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params({"penalty": penalty, "class_weight": class_weight})
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "logistic_regression")

#Grid search with optuna (library that helps to do faster and smarter grid)
def objective(trial):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    
    model = LogisticRegression(penalty=penalty, class_weight=class_weight, ...)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # 20 trials intelligents vs 100 bêtes

# Experiment 2: LogisticRegressionCV 
params_grid_cv = {
    "penalty":      ["l1", "l2", "elasticnet"],
    "class_weight": ["balanced", None],
    "cv":           [
        StratifiedKFold(n_splits=5),   # best for imbalance dataset
        KFold(n_splits=5),
        ShuffleSplit(n_splits=5)
    ]
}

mlflow.set_experiment("logistic_regression_cv")
for penalty, class_weight, cv in product(*params_grid_cv.values()):
    with mlflow.start_run():
        model = LogisticRegressionCV(
            penalty=penalty,
            class_weight=class_weight,
            solver=solver_map[penalty],
            **({"l1_ratios": [0.3, 0.5, 0.7]} if penalty == "elasticnet" else {}),
            cv=cv,
            scoring="roc_auc",
            max_iter=1000,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_params({
            "penalty":      penalty,
            "class_weight": class_weight,
            "cv":           type(cv).__name__,
            "best_C":       model.C_[0]        # parameter C (regulrisation term) chosen by CV given roc_auc
        })
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "logistic_regression_cv")