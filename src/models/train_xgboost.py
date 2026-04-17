"""
XGBoost training script
MLOps : MLflow tracking · Optuna HPO · SHAP explainability
Imbalance: scale_pos_weight / SMOTE
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from Config import MLFLOW_EXPERIMENT_NAME, S3_BUCKET_ARTIFACT_TRAINING

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

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data_processing.data_load import load_data_processed_from_S3

MODEL_NAME = "xgboost"


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path="config/training_config.yaml") -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = raw["training"].copy()
    cfg.update(raw["models"][MODEL_NAME])
    cfg["optuna"] = raw["optuna"]
    return cfg

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def cv_score(estimator, X, y, cfg: dict, scoring="average_precision"):
    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["random_state"])
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()

def get_model(params: dict, cfg: dict):
    return XGBClassifier(
        **params,
        scale_pos_weight=cfg["scale_pos_weight"],
        eval_metric = cfg["eval_metric"],
        tree_method=cfg["tree_method"],
        device=cfg["device"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )

def build_pipeline(estimator, cfg: dict):
    if cfg["use_smote"]:
        steps = [("smote", SMOTE(random_state=cfg["random_state"])), ("model", estimator)]
    else:
        steps = [("model", estimator)]
    return ImbPipeline(steps)


# ── Optuna ────────────────────────────────────────────────────────────────────
def get_search_space(trial, o):
    return {
        # boosting rounds
        "n_estimators": trial.suggest_int("n_estimators", 200, 3000),

        # tree complexity
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),

        # learning rate
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),

        # row / feature sampling
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),

        # regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 50, log=True),

        # split control
        "gamma": trial.suggest_float("gamma", 1e-8, 10, log=True),

        # leaf step size / robustness
        "max_delta_step": trial.suggest_float("max_delta_step", 0, 10),

        # grow method
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }


def run_optuna(X_train, y_train, cfg: dict):
    o = cfg["optuna"]["xgboost"]

    def objective(trial):
        params = get_search_space(trial, o)
        est    = get_model(params, cfg)
        pipe   = build_pipeline(est, cfg)
        return cv_score(pipe, X_train, y_train, cfg)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["random_state"]),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=o["hyperband_min_resource"],
            max_resource=o["hyperband_max_resource"],
            reduction_factor=o["hyperband_reduction_factor"],
        ),
    )
    study.optimize(objective, n_trials=cfg["n_optuna_trials"], show_progress_bar=False)
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

def mlflow_run(cfg, best_params, best_cv, estimator, X_train, X_test, y_train, y_test, elapsed):
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(exist_ok=True)

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.set_tags({"model": MODEL_NAME, "smote": cfg["use_smote"]})

        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds",        cfg["cv_folds"])
        mlflow.log_param("n_optuna_trials", cfg["n_optuna_trials"])
        mlflow.log_param("use_smote",       cfg["use_smote"])
        mlflow.log_param("data_suffix",     cfg["data_suffix"])

        mlflow.log_metric(f"cv_{cfg['primary_metric']}", best_cv)

        train_m = evaluate(estimator, X_train, y_train, prefix="train_")
        test_m  = evaluate(estimator, X_test,  y_test,  prefix="test_")
        mlflow.log_metrics({**train_m, **test_m})
        mlflow.log_metric("train_time_s", elapsed)

        X_sample = X_test.iloc[:cfg["shap_sample_size"]] if hasattr(X_test, "iloc") else X_test[:cfg["shap_sample_size"]]
        shap_info = compute_shap(estimator, X_sample)
        if shap_info:
            mlflow.log_params(shap_info)

        mlflow.sklearn.log_model( estimator.named_steps["model"],  MODEL_NAME, pip_requirements=[]   )
        pred  = estimator.predict(X_test)
        proba = estimator.predict_proba(X_test)[:, 1]
        report_path = output_dir / f"{MODEL_NAME}_report.txt"
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
    cfg = load_config()
    Path(cfg["output_dir"]).mkdir(exist_ok=True)

    print("\n" + "="*60)
    print(f"{MODEL_NAME} training ...")
    print("="*60 + "\n")

    suffix  = cfg["data_suffix"]
    X_train = load_data_processed_from_S3(f"X_train_{suffix}")
    y_train = load_data_processed_from_S3(f"y_train_{suffix}").values
    X_test  = load_data_processed_from_S3(f"X_test_{suffix}")
    y_test  = load_data_processed_from_S3(f"y_test_{suffix}").values

    #Load or create experiment with name and uri defined in .env
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=S3_BUCKET_ARTIFACT_TRAINING
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)  

    print(f"  Optuna HPO ({cfg['n_optuna_trials']} trials) …")
    t0 = time.perf_counter()
    best_params, best_cv = run_optuna(X_train, y_train, cfg)
    print(f"  Best CV {cfg['primary_metric']}: {best_cv:.4f}")

    est = get_model(best_params, cfg)
    pipe = build_pipeline(est, cfg)
    pipe.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s")
    row = mlflow_run(cfg, best_params, best_cv, pipe, X_train, X_test, y_train, y_test, elapsed)
    print(f"  Test ROC-AUC={row['test_roc_auc']:.4f} | PR-AUC={row['test_pr_auc']:.4f}")


if __name__ == "__main__":
    main()