import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel

def save_corr_matrix(data):
    corr_matrix = data.corr(numeric_only=True)
    print(corr_matrix['Addiction_Level'].sort_values(ascending=False))

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, fmt='.3f', annot=True, cmap='coolwarm', square=True)
    plt.savefig('R2/content/corr_plot_phones.png')

def load_data():

    data = pd.read_csv('R2/content/data/teen_phone_addiction_dataset.csv')
    data = data.drop(columns=['ID', 'Name', 'Location', 'School_Grade', 'Anxiety_Level', 'Depression_Level']) 

    print(data.head())
    print(data.info())
    # print(data['Phone_Usage_Purpose'].value_counts())
    # print(data['Apps_Used_Daily'].value_counts())

    # save_corr_matrix(data)
    return data

# PREPARING DATA
data = load_data()
X = data.drop(columns=['Addiction_Level'])
y = data['Addiction_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# PREPROCESSING
numeric_features = [
    'Age',
    'Daily_Usage_Hours',
    'Sleep_Hours',
    'Academic_Performance',
    'Social_Interactions',
    'Exercise_Hours',
    'Self_Esteem',
    'Parental_Control',
    'Screen_Time_Before_Bed',
    'Phone_Checks_Per_Day',
    'Apps_Used_Daily',
    'Time_on_Social_Media',
    'Time_on_Gaming',
    'Time_on_Education',
    'Family_Communication',
    'Weekend_Usage_Hours'
    ]

categorical_features = ['Gender', 'Phone_Usage_Purpose']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categories_transformer = Pipeline(steps=[
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categories_transformer, categorical_features)
    ],
    remainder='drop')

# DEFINE AND TRAIN MODEL

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('features_selection', SelectFromModel(RandomForestRegressor(random_state=42))),
    ('regressor', RandomForestRegressor(random_state=42))
])

def train_model(): 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse}')

    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': abs(y_test - y_pred)
    })
    print(results.head())

# train_model()


def randomized_search():
    random_search_params = {
        'features_selection__threshold': ['mean', 'median', 0.01, 0.1],
        'features_selection__estimator__n_estimators': randint(20, 60),
        'regressor__n_estimators': randint(350, 600),
        'regressor__max_depth': [15, 20, None],
        'regressor__max_features': randint(10, 17)
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=random_search_params,
        n_iter = 5,
        cv = 5,
        scoring = 'neg_mean_squared_error',
        # random_state = 42,
        n_jobs = -1
    )

    random_search.fit(X_train, y_train)

    print("BEST PARAMS:", random_search.best_params_)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE OF BEST MODEL: {rmse.round(4)}")

# randomized_search()

def grid_search():
    param_grid = {
        'features_selection__threshold': ['mean'],
        'features_selection__estimator__n_estimators': [30, 50],
        'regressor__n_estimators': [240, 350],
        'regressor__max_depth': [15, 20, None],
        'regressor__max_features': [10, 13]
    }

    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("BEST PARAMS:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE OF BEST MODEL: {rmse.round(4)}")

grid_search()



#======================================
#               NOTES
#======================================
#
#   LinearRegression RMSE: 0.8154
#   DecissionTreeRegressor RMSE: 0.9998
#   RandomForestRegressor RMSE: 0.5735 (default settings)
#
#   RandomForestRegressor (RandomizedSearch)
#
#   BEST PARAMS: {'regressor__max_depth': 15, 'regressor__max_features': 15, 'regressor__n_estimators': 237}
#   RMSE OF BEST MODEL: 0.5647
#
#   BEST PARAMS: {'regressor__max_depth': 15, 'regressor__max_features': 15, 'regressor__n_estimators': 212}
#   RMSE OF BEST MODEL: 0.564
#
#   BEST PARAMS: {'regressor__max_depth': 15, 'regressor__max_features': 15, 'regressor__n_estimators': 264}
#   RMSE OF BEST MODEL: 0.5638
#
#   ================
#   WITH SELECT_FROM_MODEL
#   ================
#
#   BEST PARAMS: {'features_selection__estimator__n_estimators': 33, 'features_selection__threshold': 0.01, 'regressor__max_depth': None, 'regressor__max_features': 14, 'regressor__n_estimators': 237}
#   RMSE OF BEST MODEL: 0.4942
#
#   BEST PARAMS: {'features_selection__estimator__n_estimators': 47, 'features_selection__threshold': 'mean', 'regressor__max_depth': 20, 'regressor__max_features': 16, 'regressor__n_estimators': 238}
#   RMSE OF BEST MODEL: 0.4948
#
#   BEST PARAMS: {'features_selection__estimator__n_estimators': 48, 'features_selection__threshold': 'mean', 'regressor__max_depth': 20, 'regressor__max_features': 14, 'regressor__n_estimators': 298}
#   RMSE OF BEST MODEL: 0.4935
#
#   BEST PARAMS: {'features_selection__estimator__n_estimators': 33, 'features_selection__threshold': 'mean', 'regressor__max_depth': None, 'regressor__max_features': 13, 'regressor__n_estimators': 471}
#   RMSE OF BEST MODEL: 0.4928
