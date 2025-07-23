import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


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

x_train.drop(columns=['Name', 'PassengerId', 'Cabin', 'Ticket'], inplace=True)

print(x_train.head())
print(x_train.info())
# print(x_train['Parch'].unique())

numeric = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorial = ['Sex', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categories_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric),
        ('cat', categories_transformer, categorial)
    ],
    remainder='drop'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Confussion Matrix:\n {cm}\n',
    f'Precision: {round(precision, 4)}\n',
    f'Recall: {round(recall, 4)}\n',
    f'f1: {round(f1, 4)}')

# DEFAULT RESULTS:
# Confussion Matrix:
# [[229  37]
# [ 37 115]]
#  Precision: 0.7566
#  Recall: 0.7566
#  f1: 0.7566

