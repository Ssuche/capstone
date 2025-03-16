
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import numpy as np

def accuracy(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y, 
        y_proba,
        pos_label=1)

    average_accuracies  = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        acc_0 = np.mean(y_pred[y == 0] == 0) 
        acc_1 = np.mean(y_pred[y == 1] == 1)  
        avg = (acc_0 + acc_1) / 2
        average_accuracies.append(avg)

    best_index = np.argmax(average_accuracies)
    optimal_threshold = thresholds[best_index]
    return optimal_threshold


def roc_auc(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    tnr = 1 - fpr  
    fnr = 1 - tpr  
    
    youden_index_positive = tpr - fpr  
    youden_index_negative = tnr - fnr 
    
    combined_youden_index = youden_index_positive + youden_index_negative
    
    best_index = np.argmax(combined_youden_index)
    optimal_threshold = thresholds[best_index]
    return optimal_threshold