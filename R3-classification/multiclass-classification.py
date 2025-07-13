import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib

def save_model(model):
    joblib.dump(model, 'MODELS/mnist-784.pkl')

#==========================
#         LOAD DATA
#==========================
def load_data():
    data = pd.read_csv('data/mnist_784.csv')

    y = data['class']
    X = data.drop(columns=['class'])

    return X.to_numpy(), y.to_numpy()

X, y = load_data()

digit = X[0]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


#=================================
#   MULTICLASS CLASSIFICATION
#=================================


def tests():

    sgd_clf = SGDClassifier()
    sgd_clf.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype('float64'))

    dec_fun = sgd_clf.decision_function([digit]).round(2)
    print(dec_fun)

    CVS = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    print(CVS)

    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize='true', values_format='.0%')
    plt.savefig('R3-classification/plots/multiclass-confusion-matrix.png')

# tests()


#=================================
#   MULTICLASS CLASSIFICATION
#       KNeighborsClassifier
#=================================

def to_float64(X):
    return X.astype('float64')

model = Pipeline(steps=[
    ('to_float', FunctionTransformer(to_float64)),
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

def grid_search():
    param_grid = {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("BEST PARAMS:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ACCURACY OF BEST MODEL: {accuracy:.4f}")

    save_model(best_model)
    

grid_search()
