import numpy as np


class ClassificationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        classes = np.unique(y_true)
        precision_scores = []
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            precision_scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return np.mean(precision_scores)

    @staticmethod
    def recall(y_true, y_pred):
        classes = np.unique(y_true)
        recall_scores = []
        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))
            recall_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return np.mean(recall_scores)

    @staticmethod
    def f1_score(y_true, y_pred, average="macro"):
        precision = ClassificationMetrics.precision(y_true, y_pred)
        recall = ClassificationMetrics.recall(y_true, y_pred)
        if precision + recall == 0:
            return 0
        f1 = 2 * (precision * recall) / (precision + recall)
        if average == "macro":
            return f1
        elif average == "micro":
            return ClassificationMetrics._f1_micro(y_true, y_pred)
        else:
            raise ValueError("Unsupported average method")

    @staticmethod
    def _f1_micro(y_true, y_pred):
        tp = np.sum((y_pred == y_true) & (y_true == 1))
        fp = np.sum((y_pred != y_true) & (y_pred == 1))
        fn = np.sum((y_pred != y_true) & (y_true == 1))
        if tp + fp + fn == 0:
            return 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
