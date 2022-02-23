from sklearn.metrics import recall_score, precision_score, f1_score

import numpy as np


def get_metrics(y: np.ndarray, y_hat: np.ndarray, is_binary: bool = True, average: str = None) -> float:
    
    if is_binary:
        y_hat = np.where(y_hat > 0.5, 1, 0)
        precision = precision_score(y, y_hat)
        recall = recall_score(y, y_hat)
        f1 = f1_score(y, y_hat)
    else:
        y_hat = y_hat.argmax(axis=1)
        precision = precision_score(y, y_hat, average=average)
        recall = recall_score(y, y_hat, average=average)
        f1 = f1_score(y, y_hat, average=average)

    return precision, recall, f1

