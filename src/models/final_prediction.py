"""File where the final prediction on evaluation is made
Input: model_ensemble saved in S3_FINAL_MODEL bucket
Output: csv with predictions and probas """

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.data_processing.data_load import load_data_processed_from_S3
from src.models.model_saving import load_model_from_s3
from src.models.model_baseline import SoftVoteEnsemble
from src.models.train_catboost import CatBoostAutoCat
import Config

if __name__ == "__main__":

    # ── 0. Load data ──────────────────────────────────────────────────────────
    X_eval_noenc_nonorm = load_data_processed_from_S3("X_eval_noenc_nonorm")
    X_eval_enc_norm     = load_data_processed_from_S3("X_eval_enc_norm")
    X_eval_enc_nonorm   = load_data_processed_from_S3("X_eval_enc_nonorm")
    client_codes        = X_eval_noenc_nonorm[Config.KEY_COLUMN].values

    # ── 1. Load ensemble ──────────────────────────────────────────────────────
    ensemble_model = load_model_from_s3(bucket=Config.S3_BUCKET_FINAL_MODELS, model_name="ensemble_model")

    # ── 2. Fix CatBoost internal state ────────────────────────────────────────
    cat_model    = ensemble_model.models[ensemble_model.names.index("CAT")]
    cat_features = X_eval_noenc_nonorm.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_model._cat_features_fitted = cat_features

    # ── 3. Extract expected features per model ────────────────────────────────
    xgb_model    = ensemble_model.models[ensemble_model.names.index("XGB")]
    lr_model     = ensemble_model.models[ensemble_model.names.index("LR")]
    features_xgb = xgb_model.get_booster().feature_names
    features_cat = cat_model.feature_names_
    features_lr  = lr_model.feature_names_in_

    # ── 4. Align inputs ───────────────────────────────────────────────────────
    base = lambda df: df.drop(columns=[Config.KEY_COLUMN], errors="ignore")
    X_ensemble = {
        "XGB": base(X_eval_enc_nonorm)[features_xgb],
        "CAT": base(X_eval_noenc_nonorm)[features_cat],
        "LR":  base(X_eval_enc_norm)[features_lr],
    }

    # ── 5. Predict ────────────────────────────────────────────────────────────
    churn_proba = ensemble_model.predict_proba(X_ensemble)[:, 1]
    churn_pred  = (churn_proba >= Config.GLOBAL_THRESHOLD).astype(int)

    print(f"Churn rate: {churn_pred.mean()*100:.1f}%  ({churn_pred.sum()} / {len(churn_pred)} clients)")
    print(f"Avg proba:  {churn_proba.mean():.4f}")

    # ── 6. Save CSV ───────────────────────────────────────────────────────────
    output = pd.DataFrame({
        Config.KEY_COLUMN: client_codes,
        "churn_proba":     churn_proba,
        "churn_pred":      churn_pred,
    })

    output_path = "churn_predictions.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}  ({len(output)} rows)")