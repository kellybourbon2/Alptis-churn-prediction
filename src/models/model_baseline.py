"""Here is the file where the Baseline of SoftVote and FinalModel with clipping are defined, 
both are used to build the final model """

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2])) #so config is visible

import Config

class SoftVoteEnsemble:
    def __init__(self, models, thresholds, names=None):
        self.models     = models
        self.thresholds = thresholds
        self.names      = names

    def predict_proba(self, X_dict):
        probas = []
        for model, name, t in zip(self.models, self.names, self.thresholds):
            p = model.predict_proba(X_dict[name])[:, 1]
            # shift proba relative to each model's threshold
            p_calibrated = p / (2 * t)  # to amplify or diminish the importance of model based on their threshold
            probas.append(np.clip(p_calibrated, 0, 1))
        mean_p = np.mean(np.column_stack(probas), axis=1)
        return np.column_stack([1 - mean_p, mean_p])

    def predict(self, X_dict, threshold=Config.GLOBAL_THRESHOLD):
        mean_p = self.predict_proba(X_dict)[:, 1]
        return (mean_p >= threshold).astype(int)
