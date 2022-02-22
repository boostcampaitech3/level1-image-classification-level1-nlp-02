from sklearn.metrics import recall_score, precision_score, f1_score

import numpy as np


def get_metrics(y: np.ndarray, y_hat: np.ndarray) -> float:
    y_hat = np.where(y_hat > 0.5, 1, 0)

    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)
    f1 = f1_score(y, y_hat)

    return precision, recall, f1

