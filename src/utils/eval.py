from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def generate_report(y_true, y_pred, set=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        f"{set}_acc": acc,
        f"{set}_f1": f1,
        f"{set}_precision": precision,
        f"{set}_recall": recall
    }