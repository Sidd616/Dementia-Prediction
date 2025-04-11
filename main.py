import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt  # for visualization
import seaborn as sns  # for advanced visualizations

# Load the data from CSV with UTF-8 encoding
data = pd.read_csv('data.csv', encoding='utf-8')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display the shape of the dataset (rows, columns)
print("\nShape of the dataset:")
print(data.shape)

# Check for missing values before forward fill
print("\nMissing values before fill:")
print(data.isnull().sum())

# Fill missing values using forward fill (ffill) without the fillna method
data.ffill(inplace=True)

# Check if there are any remaining missing values
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

# Smoking status value counts
print("\nSmoking Status counts:")
print(data['Smoking_Status'].value_counts())
