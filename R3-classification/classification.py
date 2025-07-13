import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

#==========================
#         LOAD DATA
#==========================
def load_data():
    data = pd.read_csv('data/mnist_784.csv')

    y = data['class']
    X = data.drop(columns=['class'])

    return X.to_numpy(), y.to_numpy()

X, y = load_data()

#==========================
#    DATA VISUALIZATION
#==========================

def make_digit_plot(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.savefig('R3-classification/plots/digit.png')

make_digit_plot(X[0])

#==========================
#       5 OR NOT 5
#      PREPARE DATA
#==========================
X_train, X_test, y_train, y_test  = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#==========================
#    TRAIN MODEL
#==========================

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)

# print(sgd_classifier.predict([X[0]]))

#==========================
#   PERFORMANCE MEASURES
#==========================

# y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)

def print_metrics(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f'Confussion Matrix:\n {cm}\n',
        f'Precision: {round(precision, 4)}\n',
        f'Recall: {round(recall, 4)}\n',
        f'f1: {round(f1, 4)}')

# print('METRICS FOR THRESHOLD = 0')
# print_metrics(y_train_5, y_train_pred)

# SET THRESHOLDS
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3,
                             method='decision_function', n_jobs=-1)

threshold = 2000
y_pred_5 = (y_scores > threshold)

print(f'METRICS FOR THRESHOLD = {threshold}')
print_metrics(y_train_5, y_pred_5)

def plots(y, y_scores, threshold, name):
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g--', label='Recall', linewidth=2)
    plt.vlines(threshold, 0, 1, 'k', 'dotted', label='Threshold')
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig(f'R3-classification/plots/precision-recall-threshold-curve-{name}.png')


    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, linewidth=2, label='Precision/Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'R3-classification/plots/precision-recall-curve-{name}.png')

    # ROC CURVE
    fpr, recall, thresholds = roc_curve(y, y_scores)
    
    #IF WE NEED CUSTOM PRECISSION
    custom_precision = 0.90
    custom_threshold = (precisions >= custom_precision).argmax()
    idx = (thresholds <= custom_threshold).argmax()
    custom_threshold_x, custom_threshold_y = fpr[idx], recall[idx] 

    plt.figure(figsize=(8,6))
    plt.plot(fpr, recall, linewidth=2, label='ROC Curve')
    plt.plot([custom_threshold_x], [custom_threshold_y], 'ko', label=f'Threshold for {custom_precision} precision')
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('Recall (TPR)')
    plt.savefig(f'R3-classification/plots/ROC-curve-{name}.png')

    auc = roc_auc_score(y, y_scores)
    print(f'AUC: {round(auc, 2)}')

# plots(y_train_5, y_scores, threshold, 'sgd')

#==========================
#   RandomForestClassifier
#==========================

forest_classifier = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3, method='predict_proba', n_jobs=-1)

y_scores_forest = y_probas_forest[:, 1]
# plots(y_train_5, y_scores_forest, threshold=0.5, name='rand-forest')

#AFTER THE ANALYSIS CHOOSE THE APPROPRIATE THRESHOLD
threshold = 0.5
print(f'METRICS FOR THRESHOLD = {threshold}')
y_pred_forest  = (y_scores_forest >= threshold)
print_metrics(y_train_5,  y_pred_forest)