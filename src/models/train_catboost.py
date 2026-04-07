"""
CatBoost training 
- CatBoost handles raw categoricals columns (no encoding needed)
--> to complete: need to precise which columns are categoricals and which are not to catboost
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import time
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost
import optuna

from pathlib import Path
from dataclasses import dataclass
from catboost import CatBoostClassifier, Pool, cv as catboost_cv
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, matthews_corrcoef,
)

import shap

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    target_col:         str   = TARGET_COLUMN
    cat_features:       list  = None      # None → auto-detect object columns

    # ── Imbalance ─────────────────────────────────────────────────────────────
    # Computed from y_train automatically — do not set manually
    # imbalance_ratio only used for the synthetic dataset
    imbalance_ratio:    float = 0.08

    # ── Training ──────────────────────────────────────────────────────────────
    random_state:       int   = 42
    test_size:          float = 0.2
    cv_folds:           int   = 5
    n_optuna_trials:    int   = 40        # ↑ for production, ↓ for quick test
    early_stopping_rounds: int = 50
    task_type:          str   = "GPU"     

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str  = "mlruns"
    experiment_name:    str   = "catboost_imbalanced_clf"

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:         Path  = Path("artifacts")
    primary_metric:     str   = "pr_auc"   # PR-AUC is best for imbalanced data


CFG = Config()
CFG.output_dir.mkdir(exist_ok=True)



def compute_class_weights(y_train: pd.Series):
    """CatBoost uses [w_neg, w_pos] — we set w_pos = neg/pos, w_neg = 1."""
    y_train = y_train.ravel()
    neg, pos = np.bincount(y_train)
    spw = neg / pos
    print(f"class_weights = [1, {spw:.2f}]  (neg={neg}, pos={pos})")
    return [1.0, float(spw)]


# ── CatBoost Pool helpers ──────────────────────────────────────────────────────
def make_pool(X: pd.DataFrame, y, cat_features: list):
    """CatBoost Pool — efficient data container that avoids repeated conversion."""
    return Pool(data=X, label=y, cat_features=cat_features)


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate(model: CatBoostClassifier, pool: Pool, y_true, prefix=""):
    proba = model.predict_proba(pool)[:, 1]
    pred  = model.predict(pool)
    return {
        f"{prefix}roc_auc": roc_auc_score(y_true, proba),
        f"{prefix}pr_auc":  average_precision_score(y_true, proba),
        f"{prefix}f1":      f1_score(y_true, pred),
        f"{prefix}mcc":     matthews_corrcoef(y_true, pred),  # great for imbalance
    }


# ── Optuna Search Space ────────────────────────────────────────────────────────
def suggest_params(trial: optuna.Trial) -> dict:
    """
    Full CatBoost search space.
    Ranges are wide enough for exploration but bounded to keep training fast.
    """
    grow_policy = trial.suggest_categorical("grow_policy",
                                            ["SymmetricTree", "Depthwise", "Lossguide"])
    params = {
        "iterations":           trial.suggest_int("iterations", 300, 1000),
        "learning_rate":        trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth":                trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":          trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        "subsample":            trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bylevel":    trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "min_data_in_leaf":     trial.suggest_int("min_data_in_leaf", 1, 50),
        "grow_policy":          grow_policy,
        # Categorical encoding strategy
        "cat_features_strategy": trial.suggest_categorical(
            "cat_features_strategy",
            ["BinarizedTargetMeanValue", "Counter", "FeatureFrequency"]
        ),
        # Border count (discretisation bins)
        "border_count":         trial.suggest_int("border_count", 32, 254),
        # Bagging temperature (Bayesian bootstrap)
        "bagging_temperature":  trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    # Lossguide needs max_leaves instead of depth
    if grow_policy == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)
    return params


# ── CV with CatBoost native CV ─────────────────────────────────────────────────
def cv_pr_auc(params: dict, pool: Pool, cfg: Config) -> float:
    """
    Use CatBoost's built-in CV — faster than sklearn cross_val_score
    because it shares the histogram across folds.
    """
    full_params = {
        **params,
        "loss_function":    "Logloss",
        "eval_metric":      "PRAUC",
        "task_type":        cfg.task_type,
        "random_seed":      cfg.random_state,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "verbose":          False,
        "bootstrap_type":   "Bernoulli",
    }
    # cat_features_strategy is not a valid Pool param — pass via model params
    full_params.pop("cat_features_strategy", None)

    result = catboost_cv(
        pool=pool,
        params=full_params,
        fold_count=cfg.cv_folds,
        stratified=True,
        shuffle=True,
        seed=cfg.random_state,
        verbose=False,
    )
    # result is a DataFrame; last row = best iteration
    col = "test-PRAUC-mean"
    if col in result.columns:
        return result[col].iloc[-1]
    # Fallback: AUC
    col_auc = "test-AUC-mean"
    if col_auc in result.columns:
        return result[col_auc].iloc[-1]
    return 0.0


# ── Optuna HPO ────────────────────────────────────────────────────────────────
def run_optuna(pool_train: Pool, class_weights: list, cfg: Config):
    def objective(trial):
        params = suggest_params(trial)
        params["class_weights"] = class_weights
        return cv_pr_auc(params, pool_train, cfg)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=cfg.n_optuna_trials, show_progress_bar=True)
    print(f"\n  Best CV PR-AUC : {study.best_value:.4f}")
    print(f"  Best params    : {study.best_params}")
    return study.best_params, study.best_value


# ── Final model training ───────────────────────────────────────────────────────
def train_final(best_params: dict, pool_train: Pool, pool_test: Pool,
                class_weights: list, cfg: Config) -> CatBoostClassifier:
    model = CatBoostClassifier(
        **best_params,
        loss_function="Logloss",
        eval_metric="PRAUC",
        class_weights=class_weights,
        task_type=cfg.task_type,
        random_seed=cfg.random_state,
        early_stopping_rounds=cfg.early_stopping_rounds,
        bootstrap_type="Bernoulli",
        verbose=100,
    )
    model.fit(
        pool_train,
        eval_set=pool_test,
        use_best_model=True,        # keeps best iteration, not last
    )
    return model


# ── Feature Importance / SHAP ─────────────────────────────────────────────────
def get_feature_importance(model: CatBoostClassifier,
                           pool_test: Pool,
                           feature_names: list) -> dict:
    """
    Returns top-10 features by importance.
    Uses SHAP if available, otherwise CatBoost's built-in PredictionValuesChange.
    """
    result = {}

    if SHAP_AVAILABLE:
        try:
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(pool_test)
            mean_abs   = np.abs(shap_vals).mean(axis=0)
            top_idx    = np.argsort(mean_abs)[::-1][:10]
            for rank, idx in enumerate(top_idx, 1):
                result[f"shap_feat_{rank}"] = feature_names[idx]
                result[f"shap_importance_{rank}"] = round(float(mean_abs[idx]), 6)
            print(f"  SHAP top-3: {[feature_names[i] for i in top_idx[:3]]}")
            return result
        except Exception as e:
            print(f"  SHAP failed ({e}) — falling back to native importance")

    # Native CatBoost importance (no SHAP needed)
    imp    = model.get_feature_importance(pool_test, type="PredictionValuesChange")
    top_idx = np.argsort(imp)[::-1][:10]
    for rank, idx in enumerate(top_idx, 1):
        result[f"fi_feat_{rank}"] = feature_names[idx]
        result[f"fi_importance_{rank}"] = round(float(imp[idx]), 6)
    print(f"  Native top-3: {[feature_names[i] for i in top_idx[:3]]}")
    return result


# ── MLflow logging ────────────────────────────────────────────────────────────
def log_to_mlflow(model, best_params, best_cv, pool_tr, pool_te,
                  y_tr, y_te, class_weights, feat_names, cfg, elapsed):
    with mlflow.start_run(run_name="catboost"):
        mlflow.set_tags({
            "model":            "catboost",
            "task_type":        cfg.task_type,
            "imbalance_ratio":  cfg.imbalance_ratio,
            "best_iteration":   model.best_iteration_,
        })

        # Params
        mlflow.log_params(best_params)
        mlflow.log_param("class_weight_pos", round(class_weights[1], 3))
        mlflow.log_param("cv_folds",         cfg.cv_folds)
        mlflow.log_param("n_optuna_trials",  cfg.n_optuna_trials)
        mlflow.log_param("best_iteration",   model.best_iteration_)

        # CV metric
        mlflow.log_metric(f"cv_{cfg.primary_metric}", best_cv)

        # Train / test metrics
        train_m = evaluate(model, pool_tr, y_tr, prefix="train_")
        test_m  = evaluate(model, pool_te, y_te, prefix="test_")
        mlflow.log_metrics({**train_m, **test_m, "train_time_s": elapsed})

        # Feature importance
        fi = get_feature_importance(model, pool_te, feat_names)
        mlflow.log_params(fi)

        # Model artifact
        mlflow.catboost.log_model(model, "catboost_model")

        # Classification report
        pred  = model.predict(pool_te)
        proba = model.predict_proba(pool_te)[:, 1]
        report_path = cfg.output_dir / "catboost_report.txt"
        with open(report_path, "w") as f:
            f.write("=== CATBOOST ===\n\n")
            f.write(f"Best CV PR-AUC   : {best_cv:.4f}\n")
            f.write(f"Best iteration   : {model.best_iteration_}\n\n")
            f.write(classification_report(y_te, pred, digits=4))
            f.write(f"\nConfusion Matrix:\n{confusion_matrix(y_te, pred)}\n")
            f.write(f"\nTest ROC-AUC     : {roc_auc_score(y_te, proba):.4f}\n")
            f.write(f"Test PR-AUC      : {average_precision_score(y_te, proba):.4f}\n")
            f.write(f"Test MCC         : {matthews_corrcoef(y_te, pred):.4f}\n")
        mlflow.log_artifact(str(report_path))

        print(f"\n  Test ROC-AUC = {test_m['test_roc_auc']:.4f}")
        print(f"  Test PR-AUC  = {test_m['test_pr_auc']:.4f}")
        print(f"  Test MCC     = {test_m['test_mcc']:.4f}")
        print(f"  Test F1      = {test_m['test_f1']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  CatBoost MLOps — Imbalanced Binary Classification")
    print("="*60 + "\n")

    # 1. Load data
    X_train = load_data_processed_from_S3("X_train")
    y_train = load_data_processed_from_S3("y_train").values
    X_test  = load_data_processed_from_S3("X_test")
    y_test  = load_data_processed_from_S3("y_test").values
   
    class_weights = compute_class_weights(y_train)

    # 3. CatBoost Pools
    pool_tr = make_pool(X_train, y_train, cat_features)
    pool_te = make_pool(X_test, y_test, cat_features)

    # 4. Optuna HPO
    print(f"\nOptuna HPO — {CFG.n_optuna_trials} trials …\n")
    t0 = time.perf_counter()
    best_params, best_cv = run_optuna(pool_tr, class_weights, CFG)

    # 5. Final training with early stopping on test set
    print("\nTraining final model …")
    model   = train_final(best_params, pool_tr, pool_te, class_weights, CFG)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s  |  best iteration = {model.best_iteration_}")

    # 6. MLflow
    mlflow.set_tracking_uri(CFG.mlflow_tracking_uri)
    mlflow.set_experiment(CFG.experiment_name)
    feat_names = list(X_train.columns)
    log_to_mlflow(model, best_params, best_cv, pool_tr, pool_te,
                  y_train, y_test, class_weights, feat_names, CFG, elapsed)

    # 7. Save model locally as well
    model_path = CFG.output_dir / "catboost_model.cbm"
    model.save_model(str(model_path))
    print(f"\nModel saved   → {model_path}")
    print(f"MLflow UI     → mlflow ui --backend-store-uri {CFG.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()