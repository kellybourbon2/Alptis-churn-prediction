import joblib
import numpy as np
from scipy import stats
from model_saving import save_stack_model_to_s3, load_model_from_s3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so src visible

from src.data_processing.data_load import load_data_processed_from_S3
from Config import S3_BUCKET_FINAL_MODELS


#Setting so catboost load succeed 
from src.models.train_catboost import CatBoostAutoCat #import catboost class created
import __main__
__main__.CatBoostAutoCat = CatBoostAutoCat  # inject class catboost in __main__


class MajorityVoteEnsemble:
    def __init__(self, models, thresholds, names=None):
        self.models = models
        self.thresholds = thresholds
        self.names = names

    def predict_proba(self, X_list):
        probas = []
        for model, X in zip(self.models, X_list):
            p = model.predict_proba(X)[:,1]
            probas.append(p)

        return np.mean(np.column_stack(probas), axis=1)

    def predict(self, X_list):
        preds = []

        for model, X, t in zip(self.models, X_list, self.thresholds):
            p = model.predict_proba(X)[:,1]
            preds.append((p >= t).astype(int))

        preds = np.column_stack(preds)
        vote = stats.mode(preds, axis=1).mode.ravel()
        return vote

def best_threshold_f1(y, proba, name):
    """Trouve le seuil qui maximise le F1"""
    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_t   = thresholds[best_idx]
    print(f"{name:<20} Best threshold: {best_t:.2f} | F1: {f1s[best_idx]:.4f} | Precision: {precisions[best_idx]:.4f} | Recall: {recalls[best_idx]:.4f}")
    return best_t

if __name__== "__main__":
    #1- We load the data on which the models were trained
    #for logistic regression
    X_train_enc_norm=load_data_processed_from_S3("X_train_enc_norm")
    y_train_enc_norm=load_data_processed_from_S3("y_train_enc_norm").values
    X_val_enc_norm=load_data_processed_from_S3("X_test_enc_norm")
    y_val_enc_norm=load_data_processed_from_S3("y_test_enc_norm").values

    #for xgboost 
    X_train_enc_nonorm=load_data_processed_from_S3("X_train_enc_nonorm")
    y_train_enc_nonorm=load_data_processed_from_S3("y_train_enc_nonorm").values
    X_val_enc_nonorm=load_data_processed_from_S3("X_test_enc_nonorm")
    y_val_enc_nonorm=load_data_processed_from_S3("y_test_enc_nonorm").values

    #for catboost
    X_train_noenc_nonorm =load_data_processed_from_S3("X_train_noenc_nonorm")
    y_train_noenc_nonorm =load_data_processed_from_S3("y_train_noenc_nonorm").values
    X_val_noenc_nonorm =load_data_processed_from_S3("X_test_noenc_nonorm")
    y_val_noenc_nonorm =load_data_processed_from_S3("y_test_noenc_nonorm").values

    #2- We load the models pipelines trained previously with argoworkflow 
    xgb_pipeline  = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="xgboost")
    cat_pipeline   = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="catboost")
    lr_pipeline = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="logistic_regression" )
  
    #We extract the models
    xgb_model= xgb_pipeline["model"]
    cat_model= cat_pipeline["model"]
    lr_model= lr_pipeline #we keep the full pipeline for regression logistique (scaler + model)

    #for catboost: setting of intern parameters
    cat_features = X_val_noenc_nonorm.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    cat_model._cat_features_fitted = cat_features


    #4- We define the best threshold for each of the three models based on their metrics 
    # (see notebooks/stacking_exploration.ipynb to understand how these threshold were computed)
    t_xgb   = 0.49
    t_cat   = 0.51
    t_lr    = 0.67
    t_stack = 0.57

    #5-Finally, we build the MajorityVoteEnsemble model based on the threshold 
    ensemble = MajorityVoteEnsemble(
        models=[lr_model, xgb_model, cat_model],
        thresholds=[t_lr, t_xgb, t_cat],
        names=["LR","XGB","CAT"])
    
    #6-We dump the model in S3
    save_stack_model_to_s3(ensemble, "final_model")

