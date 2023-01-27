import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score, r2_score, \
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def get_metrics_binary(y_true, y_pred):
    scores = {}
    try:
        scores["roc_auc"] = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        scores["roc_auc"] = -np.inf
    y_pred = torch.round(torch.sigmoid(y_pred))
    scores["accuracy"] = float(accuracy_score(y_true, y_pred))
    scores["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    scores["precision"] = float(precision_score(y_true, y_pred,
                                                zero_division=0))
    scores["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return scores


def get_metrics_regression(y_true, y_pred):
    scores = {}
    if np.isfinite(y_pred).all():
        scores["mse"] = float(mean_squared_error(y_true, y_pred))
        scores["mae"] = float(mean_absolute_error(y_true, y_pred))
        scores["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
        scores["r2"] = float(r2_score(y_true, y_pred))
    else:
        scores["mse"], scores["mae"], scores["mape"], scores["r2"] \
            = [-np.inf] * 4
    return scores
