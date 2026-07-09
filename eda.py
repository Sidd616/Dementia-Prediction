"""Exploratory data analysis for data.csv: summary stats and a few plots.

This is a standalone analysis script, not part of the training/GUI pipeline
(model.py has its own preprocessing with different, more careful missing-value
handling — see preprocessing.py). Run directly: `python eda.py`.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = 'data.csv'


def run(data_path=DATA_PATH):
    data = pd.read_csv(data_path, encoding='utf-8')

    print("First 5 rows of the dataset:")
    print(data.head())

    print("\nShape of the dataset:")
    print(data.shape)

    print("\nMissing values before fill:")
    print(data.isnull().sum())

    data.ffill(inplace=True)

    print("\nMissing values after fill:")
    print(data.isnull().sum())

    # Gender distribution pie chart
    gender_counts = data['Gender'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Gender Counts')
    plt.show()

    # Dementia vs Age distribution count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Age', hue='Dementia', palette='viridis')
    plt.title('Age Distribution by Dementia Status')
    plt.show()

    # Dementia vs Gender count plot
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x='Dementia', hue='Gender', palette='viridis')
    plt.title('Dementia Status by Gender')
    plt.show()

    print("\nSmoking Status counts:")
    print(data['Smoking_Status'].value_counts())


if __name__ == "__main__":
    run()
