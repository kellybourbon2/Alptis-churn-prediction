"""
HistGradientBoosting training script
MLOps : MLflow tracking · Optuna HPO · SHAP explainability
Imbalance: class_weight='balanced' / SMOTE
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import time
import warnings
import s3fs
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
import shap

from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data_processing.data_processing import DataProcessor
from src.data_processing.data_load import load_data_processed_from_S3
from Config import KEY_COLUMN, TARGET_COLUMN

@dataclass
class Config:
    random_state: int       = 42
    cv_folds: int           = 5
    n_optuna_trials: int    = 20
    use_smote: bool         = False
    experiment_name: str    = "boosting"
    output_dir: Path        = Path("artifacts")
    primary_metric: str     = "pr_auc"

CFG = Config()
CFG.output_dir.mkdir(exist_ok=True)

MODEL_NAME = "histgb"


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_scale_pos_weight(y):
    y = y.ravel()  
    neg, pos = np.bincount(y)
    spw = neg / pos
    print(f"scale_pos_weight = {spw:.2f}  (neg={neg}, pos={pos})")
    return spw


def evaluate(model, X, y, prefix=""):
    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)
    return {
        f"{prefix}roc_auc":   roc_auc_score(y, proba),
        f"{prefix}pr_auc":    average_precision_score(y, proba),
        f"{prefix}f1":        f1_score(y, pred),
        f"{prefix}precision": precision_score(y, pred),
        f"{prefix}recall":    recall_score(y, pred),
    }


def cv_score(estimator, X, y, cfg: Config, scoring="average_precision"):
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


def get_model(params: dict):
    return HistGradientBoostingClassifier(
        **params,
        class_weight="balanced",
        random_state=CFG.random_state,
    )


def build_pipeline(estimator, cfg: Config):
    steps = [("smote", SMOTE(random_state=cfg.random_state)), ("model", estimator)]
    return ImbPipeline(steps)


# ── Optuna ────────────────────────────────────────────────────────────────────

SEARCH_SPACE = lambda trial: {
    "max_iter":          trial.suggest_int("max_iter", 200, 800),
    "max_depth":         trial.suggest_int("max_depth", 3, 10),
    "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 255),
    "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 5, 50),
    "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
}


def run_optuna(X_train, y_train, cfg: Config):
    def objective(trial):
        params = SEARCH_SPACE(trial)
        est    = get_model(params)
        pipe   = build_pipeline(est, cfg)
        return cv_score(pipe, X_train, y_train, cfg)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
        pruner=optuna.pruners.HyperbandPruner(min_resource=50, max_resource=400, reduction_factor=3),
    )
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(estimator, X_sample):
    raw = estimator.named_steps["model"]
    explainer = shap.TreeExplainer(raw)
    vals = explainer(X_sample)
    mean_abs = np.abs(vals.values).mean(axis=0)
    top20 = np.argsort(mean_abs)[::-1][:20]
    feature_names = list(X_sample.columns) if hasattr(X_sample, "columns") else [str(i) for i in range(X_sample.shape[1])]
    return {f"shap_top_feat_{i+1}": feature_names[top20[i]] for i in range(20)}


# ── MLflow ────────────────────────────────────────────────────────────────────

def mlflow_run(best_params, best_cv, estimator, X_train, X_test, y_train, y_test, elapsed):
    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.set_tags({"model": MODEL_NAME, "smote": CFG.use_smote})

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", CFG.cv_folds)
        mlflow.log_param("n_optuna_trials", CFG.n_optuna_trials)

        mlflow.log_metric(f"cv_{CFG.primary_metric}", best_cv)

        train_m = evaluate(estimator, X_train, y_train, prefix="train_")
        test_m  = evaluate(estimator, X_test,  y_test,  prefix="test_")
        mlflow.log_metrics({**train_m, **test_m})
        mlflow.log_metric("train_time_s", elapsed)

        X_sample = X_test[:200].toarray() if hasattr(X_test, "toarray") else X_test[:200]
        shap_info = compute_shap(estimator, X_sample)
        if shap_info:
            mlflow.log_params(shap_info)

        mlflow.sklearn.log_model(estimator.named_steps["model"], "model")

        pred  = estimator.predict(X_test)
        proba = estimator.predict_proba(X_test)[:, 1]
        report_path = CFG.output_dir / f"{MODEL_NAME}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"=== {MODEL_NAME.upper()} ===\n\n")
            f.write(f"Best CV PR-AUC : {best_cv:.4f}\n\n")
            f.write(classification_report(y_test, pred, digits=4))
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, pred)}\n")
            f.write(f"\nTest ROC-AUC : {roc_auc_score(y_test, proba):.4f}\n")
            f.write(f"\nTest PR-AUC  : {average_precision_score(y_test, proba):.4f}\n")
            f.write(f"\nTest precision : {precision_score(y_test, pred):.4f}\n")
            f.write(f"\nTest recall    : {recall_score(y_test, pred):.4f}\n")
        mlflow.log_artifact(str(report_path))

        return {**test_m, "cv_pr_auc": best_cv, "model_name": MODEL_NAME}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("HistGradientBoosting training ...")
    print("="*60 + "\n")

    # Load training and validation sets from S3 storage
    X_train = load_data_processed_from_S3("X_train")
    y_train = load_data_processed_from_S3("y_train").values
    X_test  = load_data_processed_from_S3("X_test")
    y_test  = load_data_processed_from_S3("y_test").values

    mlflow.set_experiment(CFG.experiment_name)

    print(f"  Optuna HPO ({CFG.n_optuna_trials} trials) …")
    t0 = time.perf_counter()
    best_params, best_cv = run_optuna(X_train, y_train, CFG)
    print(f"  Best CV {CFG.primary_metric}: {best_cv:.4f}")

    est  = get_model(best_params)
    pipe = build_pipeline(est, CFG)
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # Note: histgb does not use scale_pos_weight (uses class_weight='balanced' instead)
    row = mlflow_run(best_params, best_cv, pipe, X_train, X_test, y_train, y_test, elapsed)
    print(f"  Test ROC-AUC={row['test_roc_auc']:.4f} | PR-AUC={row['test_pr_auc']:.4f}")


if __name__ == "__main__":
    main()