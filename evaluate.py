from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_per_label(y_true, y_probs, thresholds):
    results = {}
    n_labels = y_true.shape[1]

    for i in range(n_labels):
        y_pred = (y_probs[:, i] >= thresholds[i]).astype(int)
        y_true_col = y_true[:, i]
        y_pred_col = y_pred

        results[f'Label_{i}'] = {
            'Accuracy': accuracy_score(y_true_col, y_pred_col),
            'Precision': precision_score(y_true_col, y_pred_col, zero_division=0),
            'Recall': recall_score(y_true_col, y_pred_col, zero_division=0),
            'F1 Score': f1_score(y_true_col, y_pred_col, zero_division=0),
            'AUC': roc_auc_score(y_true_col, y_probs[:, i]) if len(np.unique(y_true_col)) > 1 else float('nan')
        }

    return results


def evaluate_model(y_true, y_probs, thresholds):
    """
    y_true: numpy array of shape (num_samples, num_labels)
    y_probs: numpy array of shape (num_samples, num_labels)
    thresholds: list of threshold values for each label
    """
    y_pred = np.zeros_like(y_probs)
    for i in range(y_probs.shape[1]):
        y_pred[:, i] = (y_probs[:, i] >= thresholds[i]).astype(int)

    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (micro)': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'Recall (micro)': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'F1 Score (micro)': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'Precision (macro)': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Recall (macro)': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 Score (macro)': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

  
    aucs = []
    for i in range(y_test_true.shape[1]):
        if len(np.unique(y_test_true[:, i])) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_test_true[:, i], y_test_pred[:, i]))
    results['AUC'] = np.nanmean(aucs)
    return results
