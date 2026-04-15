"""
Logistic Regression training script
MLOps : MLflow tracking · Optuna HPO · SHAP explainability
Imbalance: class_weight="balanced" / SMOTE
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
os.chdir("src/models")  # to avoid pyproject to be detected by mlflow

os.environ["MLFLOW_TRACKING_URI"]      = "https://projet-bdc-data-mlflow.lab.groupe-genes.fr"
os.environ["MLFLOW_S3_ENDPOINT_URL"]   = "https://minio-simple.lab.groupe-genes.fr"
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
os.environ["MLFLOW_S3_IGNORE_TLS"]     = "true"

import time
import warnings
import yaml
import numpy as np
import mlflow
import mlflow.sklearn
import optuna
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_processing.data_load import load_data_processed_from_S3
from Config import MLFLOW_EXPERIMENT_NAME

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_NAME = "logistic_regression"


# ── CONFIG ─────────────────────────────────────────────

def load_config(path="config/training_config.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = raw["training"].copy()
    cfg.update(raw["models"][MODEL_NAME])
    cfg["optuna"] = raw["optuna"]
    return cfg


# ── HELPERS ────────────────────────────────────────────

def evaluate(model, X, y, prefix=""):
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        f"{prefix}roc_auc": roc_auc_score(y, proba),
        f"{prefix}pr_auc": average_precision_score(y, proba),
        f"{prefix}f1": f1_score(y, pred),
        f"{prefix}precision": precision_score(y, pred),
        f"{prefix}recall": recall_score(y, pred),
    }


def cv_score(estimator, X, y, cfg, scoring="average_precision"):
    cv = StratifiedKFold(
        n_splits=cfg["cv_folds"],
        shuffle=True,
        random_state=cfg["random_state"]
    )
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


# ── MODEL ─────────────────────────────────────────────

def get_model(params, cfg):
    return LogisticRegression(
        **params,
        class_weight=cfg["class_weight"],
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        random_state=cfg["random_state"],
    )


def build_pipeline(estimator, cfg):
    steps = []

    # scaling is REQUIRED for logistic regression
    steps.append(("scaler", StandardScaler()))

    if cfg["use_smote"]:
        steps.append(("smote", SMOTE(random_state=cfg["random_state"])))

    steps.append(("model", estimator))

    return ImbPipeline(steps)


# ── OPTUNA ─────────────────────────────────────────────

def get_search_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)
    }


def run_optuna(X_train, y_train, cfg):

    def objective(trial):
        params = get_search_space(trial)
        model = get_model(params, cfg)
        pipe = build_pipeline(model, cfg)
        return cv_score(pipe, X_train, y_train, cfg)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg["n_optuna_trials"])

    return study.best_params, study.best_value

# ── SHAP (logistic-compatible) ─────────────────────────
def compute_shap(pipeline, X_sample):
    """
    Use LinearExplainer — the correct explainer for Logistic Regression.
    TreeExplainer would crash or give wrong results here.
    """
    scaler = pipeline.named_steps["scaler"]
    model  = pipeline.named_steps["model"]
    
    # SHAP expects scaled input (same as what the model sees)
    X_scaled = scaler.transform(X_sample)
    
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top20 = np.argsort(mean_abs)[::-1][:20]
    
    feature_names = (
        list(X_sample.columns)
        if hasattr(X_sample, "columns")
        else [str(i) for i in range(X_sample.shape[1])]
    )
    return {f"shap_top_feat_{i+1}": feature_names[top20[i]] for i in range(min(20, len(top20)))}


# ── COEFFICIENTS (top N) ───────────────────────────────
def log_top_coefficients(pipeline, feature_names, cfg, top_n=50):
    """
    Extract and log LR coefficients — the most direct feature importance
    for a linear model. Logs to MLflow and prints a ranked table.
    """
    scaler = pipeline.named_steps["scaler"]
    model  = pipeline.named_steps["model"]

    # Coefficients are on the scaled space → divide by std to get
    # the "per-unit-of-original-feature" importance ranking
    coefs = model.coef_[0]  # shape (n_features,)

    # Rank by absolute value
    top_idx = np.argsort(np.abs(coefs))[::-1][:top_n]

    print(f"\n{'Rank':<6} {'Feature':<45} {'Coefficient':>12} {'|Coef|':>10}")
    print("─" * 76)

    coef_dict = {}
    for rank, idx in enumerate(top_idx, 1):
        fname = feature_names[idx]
        coef  = coefs[idx]
        print(f"{rank:<6} {fname:<45} {coef:>12.4f} {abs(coef):>10.4f}")
        coef_dict[f"coef_{rank:02d}_{fname[:40]}"] = round(float(coef), 6)

    # Log to MLflow (param = name→value mapping)
    mlflow.log_params(coef_dict)

    return coef_dict

# ── MLflow ─────────────────────────────────────────────

def mlflow_run(cfg, best_params, best_cv, model, X_train, X_test, y_train, y_test, elapsed):

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=MODEL_NAME):

        mlflow.set_tags({"model": MODEL_NAME})

        mlflow.log_params(best_params)
        mlflow.log_metric("cv_pr_auc", best_cv)

        train_m = evaluate(model, X_train, y_train, "train_")
        test_m = evaluate(model, X_test, y_test, "test_")

        mlflow.log_metrics({**train_m, **test_m})
        mlflow.log_metric("train_time_s", elapsed)

        # SHAP (LinearExplainer)
        X_sample = (
            X_test.iloc[:cfg["shap_sample_size"]]
            if hasattr(X_test, "iloc")
            else X_test[:cfg["shap_sample_size"]]
        )
        shap_info = compute_shap(model, X_sample)
        if shap_info:
            mlflow.log_params(shap_info)

        # Coefficients
        feature_names = (
            list(X_train.columns)
            if hasattr(X_train, "columns")
            else [str(i) for i in range(X_train.shape[1])]
        )
        log_top_coefficients(model, feature_names, cfg, top_n=10)

        # SAVE MODEL
        mlflow.sklearn.log_model(
            model.named_steps["model"],
            name= MODEL_NAME,
        )

    return {**test_m, "cv_pr_auc": best_cv, "model_name": MODEL_NAME}


# ── MAIN ───────────────────────────────────────────────

def main():

    cfg = load_config()

    Path(cfg["output_dir"]).mkdir(exist_ok=True)

    print("\n" + "="*60)
    print(f"{MODEL_NAME} training ...")
    print("="*60 + "\n")

    X_train = load_data_processed_from_S3(f"X_train_{cfg['data_suffix']}")
    y_train = load_data_processed_from_S3(f"y_train_{cfg['data_suffix']}").values
    X_test = load_data_processed_from_S3(f"X_test_{cfg['data_suffix']}")
    y_test = load_data_processed_from_S3(f"y_test_{cfg['data_suffix']}").values

    #Load or create experiment with name and uri defined in .env
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=S3_BUCKET_ARTIFACT_TRAINING
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)  

    t0 = time.perf_counter()

    best_params, best_cv = run_optuna(X_train, y_train, cfg)

    model = get_model(best_params, cfg)
    pipe = build_pipeline(model, cfg)
    pipe.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0

    row = mlflow_run(
        cfg,
        best_params,
        best_cv,
        pipe,
        X_train,
        X_test,
        y_train,
        y_test,
        elapsed
    )

    print(row)


if __name__ == "__main__":
    main()