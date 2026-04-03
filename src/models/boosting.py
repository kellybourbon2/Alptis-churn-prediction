"""
Boosting models training
Models   : XGBoost - LightGBM - HistGradientBoosting
Data processing: encoding, no normalisation
MLOps    : MLflow tracking · Optuna HPO · SHAP explainability
Imbalance: scale_pos_weight / class_weight / SMOTE (optional)

--> augment the optuna_trial (5 right now --> better at 30)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so data_loading visible

import time
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna

from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from sklearn.pipeline import Pipeline

# Boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

#try shap

import shap
SHAP_AVAILABLE = True

# Optional: imbalanced-learn SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.data_processing.data_processing import DataProcessor
from Config import KEY_COLUMN, TARGET_COLUMN


@dataclass
class Config:
    #Reproductibility
    random_state: int     = 42

    # Training
    cv_folds: int         = 5
    n_optuna_trials: int  = 20            # ↑ for better HPO, ↓ for speed
    early_stopping_rounds: int = 20
    use_smote: bool       = False          # flip to True to apply smote

    # MLflow
    experiment_name: str = "boosting"

    # Output
    output_dir: Path      = Path("artifacts")
    primary_metric: str   = "pr_auc"       # optimise for PR-AUC (best for imbalance)
    device:             str = "cuda"  #to run on gpu

CFG = Config()
CFG.output_dir.mkdir(exist_ok=True)


def compute_scale_pos_weight(y):
    """Ratio neg/pos — used by XGBoost & LightGBM --> to handle imbalance"""
    neg, pos = np.bincount(y)
    spw = neg / pos
    print(f"scale_pos_weight = {spw:.2f}  (neg={neg}, pos={pos})")
    return spw


# ── Metrics ───────────────────────────────────────────────────────────────────
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
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                         random_state=cfg.random_state)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


def get_model(name: str, params: dict, spw: float, random_state: int):
    """Return a bare (non-pipeline) estimator."""
    if name == "xgboost":
        return XGBClassifier(
            **params, 
            #scale_pos_weight=spw
            use_label_encoder=False,
            eval_metric="aucpr",
            tree_method="hist",         # fast histogram
            device=CFG.device,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1,
        )
    elif name == "lightgbm":
        return LGBMClassifier(
            **params,
            scale_pos_weight=spw,
            objective="binary",
            metric="average_precision",
            boosting_type="gbdt",
            verbose=-1,
            random_state=random_state,
            n_jobs=-1,
            #device= "cuda"
        )
    elif name == "histgb":
        return HistGradientBoostingClassifier(
            **params,
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {name}")


# ── Optuna Search Spaces ──────────────────────────────────────────────────────
SEARCH_SPACES = {
    "xgboost": lambda trial: {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    },
    "lightgbm": lambda trial: {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", -1, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    },
    "histgb": lambda trial: {
        "max_iter":         trial.suggest_int("max_iter", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes":   trial.suggest_int("max_leaf_nodes", 15, 255),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization":trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
    },
}


# ── Optuna HPO ────────────────────────────────────────────────────────────────
def run_optuna(model_name: str, X_train, y_train, spw: float, cfg: Config):
    def objective(trial):
        params = SEARCH_SPACES[model_name](trial)
        est    = get_model(model_name, params, spw, cfg.random_state)
        pipe   = build_pipeline(est, cfg)
        return cv_score(pipe, X_train, y_train, cfg)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
        pruner=optuna.pruners.HyperbandPruner(min_resource=50, max_resource=400, reduction_factor=3),
    )
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ── Pipeline ──────────────────────────────────────────────────────────────────
def build_pipeline(estimator, cfg: Config):
    steps = []   # no scaler
    steps.insert(0, ("smote", SMOTE(random_state=cfg.random_state)))
    PipeClass = ImbPipeline
    steps.append(("model", estimator))
    return PipeClass(steps)


# ── SHAP Explainability ───────────────────────────────────────────────────────
def compute_shap(model_name: str, estimator, X_sample, cfg: Config):
    """Print top 20 shap values given model"""
    raw = estimator.named_steps["model"]
    explainer = shap.TreeExplainer(raw)
    vals = explainer(X_sample)
    mean_abs = np.abs(vals.values).mean(axis=0)
    top5 = np.argsort(mean_abs)[::-1][:5]
    feature_names = list(X_sample.columns) if hasattr(X_sample, "columns") else [str(i) for i in range(X_sample.shape[1])]
    return {f"shap_top_feat_{i+1}": feature_names[top5[i]] for i in range(20)}

# ── MLflow Run ────────────────────────────────────────────────────────────────
def mlflow_run(model_name, best_params, best_cv, estimator, X_train, X_test, y_train, y_test,
               spw, cfg, elapsed):
    """Save parameters, metrics computed on training and test datasets on mlflow 
    (from best model selected by optuna)"""

    with mlflow.start_run(run_name=model_name):
        # Tags
        mlflow.set_tags({
            "model":    model_name,
            "smote":    cfg.use_smote,
        })

        # Save Parameters of training 
        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", round(spw, 3))
        mlflow.log_param("cv_folds", cfg.cv_folds)
        mlflow.log_param("n_optuna_trials", cfg.n_optuna_trials)

        # CV metric
        mlflow.log_metric(f"cv_{cfg.primary_metric}", best_cv)

        # Train metrics
        train_m = evaluate(estimator, X_train, y_train, prefix="train_")
        test_m  = evaluate(estimator, X_test, y_test, prefix="test_")
        mlflow.log_metrics({**train_m, **test_m})
        mlflow.log_metric("train_time_s", elapsed)

        # SHAP
        shap_info = compute_shap(model_name, estimator,
                                 X_test[:200].toarray() if hasattr(X_test, "toarray") else X_test[:200],
                                 cfg)
        if shap_info:
            mlflow.log_params(shap_info)

        # Model artifact
        if model_name == "xgboost":
            mlflow.xgboost.log_model(estimator.named_steps["model"], "model")
        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(estimator.named_steps["model"], "model")
        elif model_name == "histgb":
            mlflow.lightgbm.log_model(estimator.named_steps["model"], "model")
        else: 
            logging.warning("Error - model not defined")

        # Classification report
        pred  = estimator.predict(X_test)
        proba = estimator.predict_proba(X_test)[:, 1]
        report_path = cfg.output_dir / f"{model_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"=== {model_name.upper()} ===\n\n")
            f.write(f"Best CV PR-AUC : {best_cv:.4f}\n\n")
            f.write(classification_report(y_test, pred, digits=4))
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_test, pred)}\n")
            f.write(f"\nTest ROC-AUC  : {roc_auc_score(y_test, proba):.4f}\n")
            f.write(f"\nTest PR-AUC   : {average_precision_score(y_test, proba):.4f}\n")
            f.write(f"\nTest precision  : {precision_score(y_test, pred):.4f}\n")
            f.write(f"\nTest recall  : {recall_score(y_test, pred):.4f}\n")
        mlflow.log_artifact(str(report_path))

        return {**test_m, "cv_pr_auc": best_cv, "model_name": model_name}


# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():

    print("\n" + "="*60)
    print("Starting boosting models mlflow experiment (xgboost - histgb - lightgbm) .... ")
    print("="*60 + "\n")

    # 1.Load the data: with no normalisation but encoded
    train_processor = DataProcessor(mode="training")
    df_train = train_processor.run(optional_encoding=True, optional_normalisation=False)
    
    test_processor = DataProcessor(mode="validation")

    #pass the arguments from training to test so no data-leakage
    test_processor.global_means = train_processor.global_means
    test_processor.target_encoding_maps = train_processor.target_encoding_maps
    test_processor.ordinal_maps         = train_processor.ordinal_maps
    test_processor.encoded_columns      = train_processor.encoded_columns
    test_processor.scaler               = train_processor.scaler
    test_processor.normalized_columns   = train_processor.normalized_columns

    #process test dataset with arguments computed on train
    df_test = test_processor.run_transform(optional_encoding=True, optional_normalisation=False)

    X_train = df_train.drop(columns=[TARGET_COLUMN, KEY_COLUMN])
    y_train = df_train[TARGET_COLUMN].values
    X_test  = df_test.drop(columns=[TARGET_COLUMN, KEY_COLUMN])
    y_test  = df_test[TARGET_COLUMN].values

    spw  = compute_scale_pos_weight(y_train) #Ratio: non_churners/churners

    # 2. MLflow experiment
    mlflow.set_experiment(CFG.experiment_name)

    # 3. Models to train
    models = ["lightgbm", "histgb", "xgboost"]
    leaderboard = []

    for model_name in models:
        print(f"{model_name.upper()}")

        #  Optuna HPO: to compute better hyperparameters
        print(f"  Optuna HPO ({CFG.n_optuna_trials} trials) …")
        t0 = time.perf_counter()
        best_params, best_cv = run_optuna(model_name, X_train, y_train, spw, CFG)
        print(f"  Best CV {CFG.primary_metric}: {best_cv:.4f}")

        # Retrain the model with best hyperparameters on full training dataset
        est  = get_model(model_name, best_params, spw, CFG.random_state)
        pipe = build_pipeline(est, CFG)
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        print(f"  Training done in {elapsed:.1f}s")

        # MLflow
        row = mlflow_run(
            model_name, best_params, best_cv, pipe,
            X_train, X_test, y_train, y_test, spw, CFG, elapsed
        )
        leaderboard.append(row)
        print(f"  Test ROC-AUC={row['test_roc_auc']:.4f} | PR-AUC={row['test_pr_auc']:.4f}")

    # 4. Leaderboard
    print("\n" + "="*60)
    print("  LEADERBOARD  (sorted by test PR-AUC)")
    print("="*60)
    lb = pd.DataFrame(leaderboard).sort_values("test_pr_auc", ascending=False)
    print(lb[["model_name", "test_roc_auc", "test_pr_auc", "test_f1","test_precision", "test_recall", "cv_pr_auc"]].to_string(index=False))

    lb.to_csv(CFG.output_dir / "leaderboard.csv", index=False)
    print(f"\n Leaderboard saved → {CFG.output_dir / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()