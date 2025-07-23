import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOAD TITANIC DATA
def load_data():
    train_set = pd.read_csv('data/titanic/train.csv')
    test_set = pd.read_csv('data/titanic/test.csv')
    y_test = pd.read_csv('data/titanic/gender_submission.csv')

    y_test.drop(columns=['PassengerId'], inplace=True)
    y_train = train_set['Survived']
    train_set.drop(columns=['Survived'], inplace=True)

    # print(train_set.info())
    # print(y_train.info())
    # print(test_set.info())
    # print(y_test.info())

    return train_set, y_train, test_set, y_test

x_train, y_train, x_test, y_test = load_data()
