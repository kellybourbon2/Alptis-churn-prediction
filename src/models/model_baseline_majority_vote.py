# src/models/model_baseline_majority_vote.py

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityVoteEnsemble:
    def __init__(self, models, thresholds, names=None):
        self.models     = models
        self.thresholds = thresholds
        self.names      = names

    def predict_proba(self, X_dict):
        probas = []
        for model, name in zip(self.models, self.names):
            p = model.predict_proba(X_dict[name])[:, 1]
            probas.append(p)
        mean_p = np.mean(np.column_stack(probas), axis=1)
        return np.column_stack([1 - mean_p, mean_p])

    def predict(self, X_dict):
        preds = []
        for model, name, t in zip(self.models, self.names, self.thresholds):
            p = model.predict_proba(X_dict[name])[:, 1]
            preds.append((p >= t).astype(int))
        preds = np.column_stack(preds)
        vote  = stats.mode(preds, axis=1).mode.ravel()
        return vote