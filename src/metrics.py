import numpy as np


def calculate_confusion_matrix(y_true, y_pred):
    """Retorna a matriz de confusão em formato [[VN, FP], [FN, VP]]."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    matrix = np.array([[tn, fp], [fn, tp]], dtype=int)
    return {
        "matrix": matrix,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def calculate_metrics(y_true, y_pred):
    """Calcula acurácia, precisão, recall e F1-score manualmente."""
    conf = calculate_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf["tn"], conf["fp"], conf["fn"], conf["tp"]

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "matrix": conf["matrix"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


__all__ = ["calculate_confusion_matrix", "calculate_metrics"]