import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict

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
X_train, X_test, y_train, y_test  = X[:6000], X[6000:], y[:6000], y[6000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#==========================
#    TRAIN MODEL
#==========================

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)

print(sgd_classifier.predict([X[0]]))

#==========================
#   PERFORMANCE MEASURES
#==========================
#   ...