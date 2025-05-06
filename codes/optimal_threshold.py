
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score
import numpy as np

def f1score(model, X, y):
    y_proba = model.predict_proba(X)[:, 0] 
    thresholds = np.unique(y_proba)

    macro_f1_scores = []

    for threshold in thresholds:
        y_pred = (y_proba < threshold).astype(int) 
        f1_0 = f1_score(y, y_pred, pos_label=0)
        f1_1 = f1_score(y, y_pred, pos_label=1)
        macro_f1 = (f1_0 + f1_1) / 2
        macro_f1_scores.append(macro_f1)

    best_index = np.argmax(macro_f1_scores)
    optimal_threshold = thresholds[best_index]
    return optimal_threshold

    
def youden(model, X, y):
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