"""File where the final prediction on evaluation is made
Output: csv with predictions """
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so src and Config visible

import pandas as pd
from src.data_processing.data_load import load_data_processed_from_S3
from src.models.model_saving import load_model_from_s3
from src.models.model_baseline_majority_vote import MajorityVoteEnsemble
from src.models.train_catboost import CatBoostAutoCat
from Config import S3_BUCKET_FINAL_MODELS
import Config

if __name__ == "__main__":  # fix: = not ==
    # ── 0. Load data ──────────────────────────────────────────────────────────
    X_eval_noenc_nonorm = load_data_processed_from_S3("X_eval_noenc_nonorm")
    X_eval_enc_norm     = load_data_processed_from_S3("X_eval_enc_norm")
    X_eval_enc_nonorm   = load_data_processed_from_S3("X_eval_enc_nonorm")
    client_codes        = X_eval_noenc_nonorm[Config.KEY_COLUMN]

    # ── 1. Load ensemble ──────────────────────────────────────────────────────
    ensemble_model = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="final_model")

    # ── 2. Fix CatBoost internal state ────────────────────────────────────────
    cat_model    = ensemble_model.models[ensemble_model.names.index("CAT")]
    cat_features = X_eval_noenc_nonorm.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_model._cat_features_fitted = cat_features

    # ── 3. Extract expected features per model ────────────────────────────────
    xgb_model = ensemble_model.models[ensemble_model.names.index("XGB")]
    lr_model  = ensemble_model.models[ensemble_model.names.index("LR")]

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

    # ── 5. Predictions ────────────────────────────────────────────────────────
    def get_proba(model, X): return model.predict_proba(X)[:, 1]
    def get_pred(model, X):  return model.predict(X)

    results = pd.DataFrame({
        "client_code": client_codes.values,
        "churn_proba": get_proba(ensemble_model, X_ensemble),
        "churn_pred":  get_pred(ensemble_model, X_ensemble),
    })

    # ── 6. Global threshold ───────────────────────────────────────────────────
    GLOBAL_THRESHOLD = 0.65
    results["churn_final"] = (results["churn_proba"] >= GLOBAL_THRESHOLD).astype(int)
    print(f"[0] Churn rate initial:        {results['churn_final'].mean()*100:.1f}%  ({results['churn_final'].sum()} clients)")

    # ── 7. Clip 1 — NPS churn mention ─────────────────────────────────────────
    nps_clients = X_eval_noenc_nonorm[X_eval_noenc_nonorm["client_nps_churn_mention_n"] > 0][Config.KEY_COLUMN].tolist()
    before = results["churn_final"].sum()
    results.loc[results["client_code"].isin(nps_clients), "churn_final"] = 1
    print(f"[1] After NPS clip (+1):       {results['churn_final'].mean()*100:.1f}%  (+{results['churn_final'].sum() - before} clients)")

    # ── 8. Clip 1 — Mail churn mention ────────────────────────────────────────
    mail_clients = X_eval_noenc_nonorm[X_eval_noenc_nonorm["interaction_mail_churn_mention"] > 0][Config.KEY_COLUMN].tolist()
    before = results["churn_final"].sum()
    results.loc[results["client_code"].isin(mail_clients), "churn_final"] = 1
    print(f"[2] After mail clip (+1):      {results['churn_final'].mean()*100:.1f}%  (+{results['churn_final'].sum() - before} clients)")

    # ── 9. Clip 0 — Toujours engagé ───────────────────────────────────────────
    engaged_clients = X_eval_noenc_nonorm[X_eval_noenc_nonorm["client_toujours_engage"] == 1][Config.KEY_COLUMN].tolist()
    before = results["churn_final"].sum()
    results.loc[results["client_code"].isin(engaged_clients), "churn_final"] = 0
    print(f"[3] After engaged clip (→0):   {results['churn_final'].mean()*100:.1f}%  (-{before - results['churn_final'].sum()} clients)")

    # ── 10. Final summary ─────────────────────────────────────────────────────
    print(f"\n✅ Churn rate final:           {results['churn_final'].mean()*100:.1f}%  ({results['churn_final'].sum()} / {len(results)} clients)")

    # ── 11. Save CSV ──────────────────────────────────────────────────────────
    output = results[[Config.KEY_COLUMN, "churn_final"]].rename(columns={Config.KEY_COLUMN: Config.KEY_COLUMN, "churn_final": "churn_pred"})
    output_path = "churn_predictions.csv"
    output.to_csv(output_path, index=False)
    print(f"Final prediction saved: {output_path}  ({len(output)} rows)")