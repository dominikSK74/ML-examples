import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

data = load_data()