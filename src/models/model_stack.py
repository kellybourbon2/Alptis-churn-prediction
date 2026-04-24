"""Here is the file to build the stack SoftVoteEnsemble model based on LogisticReg, Catboost and Xgboost
to understand the choice of threshold per model, see notebooks/Stacking_models_exploration
"""

import joblib
import numpy as np
from scipy import stats
from model_saving import save_stack_model_to_s3, load_model_from_s3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so src visible

from src.data_processing.data_load import load_data_processed_from_S3
from Config import S3_BUCKET_FINAL_MODELS
import Config


#Setting of baseline
from src.models.train_catboost import CatBoostAutoCat
from src.models.model_baseline import SoftVoteEnsemble
import __main__
__main__.CatBoostAutoCat = CatBoostAutoCat
__main__.SoftVoteEnsemble = SoftVoteEnsemble

if __name__== "__main__":
    #1- We load the models pipelines trained previously with argoworkflow 
    xgb_pipeline  = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="xgboost")
    cat_pipeline   = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="catboost")
    lr_pipeline = load_model_from_s3(bucket=S3_BUCKET_FINAL_MODELS, model_name="logistic_regression" )
  
    #2-We extract the models
    xgb_model= xgb_pipeline["model"]
    cat_model= cat_pipeline["model"]
    lr_model= lr_pipeline #we keep the full pipeline for regression logistique (scaler + model)

    #3-for catboost: setting of intern parameters
    X_val_noenc_nonorm = load_data_processed_from_S3("X_test_noenc_nonorm")
    cat_features = X_val_noenc_nonorm.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    cat_model._cat_features_fitted = cat_features

    #4- We define the best threshold for each of the three models based on their metrics 
    # (see notebooks/stacking_exploration.ipynb to understand how these threshold were computed)
    t_xgb   = Config.XGBOOST_THRESHOLD
    t_cat   = Config.CATBOOST_THRESHOLD
    t_lr    = Config.LOGREG_THRESHOLD

    #5-Finally, we build the MajorityVoteEnsemble model based on the threshold 
    ensemble = SoftVoteEnsemble(
        models=[lr_model, xgb_model, cat_model],
        thresholds=[t_lr, t_xgb, t_cat],
        names=["LR","XGB","CAT"])
    
    #6-We dump the ensemble model in S3
    save_stack_model_to_s3(ensemble, "ensemble_model")

