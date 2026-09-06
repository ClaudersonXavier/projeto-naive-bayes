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


def metrics_at_threshold(y_true, p1, threshold):
    """Calcula a matriz de confusão e as métricas para um limiar de decisão
    sobre P(Y=1|x), reaproveitando calculate_metrics."""
    p1 = np.asarray(p1, dtype=float)
    y_pred = (p1 >= threshold).astype(int)
    return calculate_metrics(y_true, y_pred)


def roc_curve_auc(y_true, p1):
    """Calcula a curva ROC (FPR, TPR) e a AUC a partir de P(Y=1|x).

    Retorna (fpr, tpr, auc), com fpr/tpr ordenados por limiar decrescente
    (do ponto (0,0) ao (1,1)).
    """
    y_true = np.asarray(y_true, dtype=int)
    p1 = np.asarray(p1, dtype=float)

    order = np.argsort(-p1)
    y_sorted = y_true[order]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve_auc requer ao menos uma observação de cada classe.")

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


__all__ = [
    "calculate_confusion_matrix",
    "calculate_metrics",
    "metrics_at_threshold",
    "roc_curve_auc",
]