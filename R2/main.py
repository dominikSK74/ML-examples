import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import joblib
from sklearn.feature_selection import SelectFromModel

def save_model(model):
    joblib.dump(model, 'MODELS/california_housing_model.pkl')

def load_housing_data():
    return pd.read_csv('R2/content/data/housing.csv')

data = load_housing_data()

#=================================
# PEARSON CORELATION
# WARNING: ONLY NUMERICAL VARIABLES
#=================================

corr_matrix = data.corr(numeric_only=True)
# print(corr_matrix['median_house_value'].sort_values(ascending=False))
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.savefig('R2/content/corrplot.png')

#=======================================
#   ALL THE ABOVE OPERATIONS BUT SHORTER
#=======================================

# 0. PREPARING DATA
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']

# 1. DATA BREAKDOWN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. PREPROCESSING
numeric_features = ['longitude', 'latitude', 'housing_median_age', 'total_bedrooms', 'population', 'total_rooms', 'households', 'median_income'];
categorical_features = ['ocean_proximity']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categories_transformer = Pipeline(steps=[
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

# COLUMN TRANSFORMER HAS ONLY ONE OPERATIONS FOR ONE COLUMN
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categories_transformer, categorical_features)
    ],
    remainder='drop')  # reminder='passthrough'
                       # drop - delete unlisted columns
                       # passthrough - leaves unchanged unlisted columns

# VIEW DATA AFTER TRANSFORMATIONS
def view_data():
    transformed_train_data = preprocessor.fit_transform(X_train)
    columns_names = preprocessor.get_feature_names_out()
    df_train_data = pd.DataFrame(transformed_train_data, columns=columns_names)
    print('========= BEFORE =========')
    print(X_train.info())
    print(X_train.head())
    print('========= AFTER =========')
    print(df_train_data.info())
    print(df_train_data.head())

# view_data()

# 3. DEFINE AND TRAIN MODEL

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('features_selection', SelectFromModel(RandomForestRegressor(random_state=42)) ),
    ('regressor', RandomForestRegressor(random_state=42))
])

# RandomForestRegressor params:
#   n_estimators=100,     # tree counts: default: 100-500
#   max_depth=10,         # default: None 
#   max_features,         # default: sqrt - sqrt of count feautures
#   n_jobs=-1,            # CPU counts default: 1 (-1 = all CPUs)

#=======================
# CROSS_VAL_SCORE USE TO CHECK MODEL ONLY
# WARNING: UNNECESSERY IF USE GRIDSEARCH
#=======================
def cross_val():
    scores = cross_val_score(model, X_train, y_train, cv = 5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())

# cross_val()

#===========================
# GRIDSEARCH USE TO SEARCH THE BEST PARAMS FOR MODEL
#===========================
def grid_search():
    param_grid = {
        'features_selection__threshold': ['mean', 'median', 0.01],
        'features_selection__estimator__n_estimators': [10, 30, 50],
        'regressor__n_estimators': [100, 184, 200],
        'regressor__max_depth': [15, None],
        'regressor__max_features': [7, 10] # 6 = 60% of features, 8 = 80% ...
    }

    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("BEST PARAMS:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE OF BEST MODEL: {rmse.round(2)}")

    # save_model(best_model)

# grid_search()

#==========================
# RandomizedSearchCV USE TO  SEARCH THE BEST PARAMS
# THIS METHOD USING RANDOM COMBINATIONS (GRID SEARCH USE ALL COMBINATIONS)
# GRID SEARCH IS MORE ACCURATE THEN RANDOMIZED SEARCH
# BUT RANDOMIZED IS FASTER
#==========================
def randomized_search():
    random_search_params = {
        'regressor__n_estimators': randint(100, 200),
        'regressor__max_depth': [10, 15, None],
        'regressor__max_features': randint(4, 11)
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=random_search_params,
        n_iter=5, # COUNT OF RANDOM COMBINATIONS
        cv = 5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("BEST PARAMS:", random_search.best_params_)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE OF BEST MODEL: {rmse.round(2)}")

# randomized_search()

#=======================
# TRAIN MODEL AND CHECK PREDICTIONS
# WARNING: IF YOU TRAIN MODEL USING GRID SEARCH DONT TRAIN AGAIN MODEL!
#=======================
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



