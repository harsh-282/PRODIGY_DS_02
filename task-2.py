# Import necessary libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set Matplotlib backend for better handling of plots (especially in headless environments)
matplotlib.use('Agg')  # Use 'Agg' backend to avoid GUI-related issues

# Load the Titanic dataset
file_path = "F:/HARSHINI/Prodigy Infotech/train.csv"  # Make sure the path is correct
df = pd.read_csv(file_path)

# Display dataset head and info
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill missing Age with median value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing Embarked with mode value

# Verify that missing values are filled
print("\nMissing Values after filling:")
print(df.isnull().sum())

# Display a statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Visualize missing values using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Exploratory Data Analysis (EDA) - Visualizing relationships between variables

# 1. Distribution of 'Age' and 'Fare'
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=30, color='blue')
plt.title("Distribution of Age")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], kde=True, bins=30, color='green')
plt.title("Distribution of Fare")
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# 2. Relationship between 'Survived' and 'Pclass'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set1')
plt.title("Survival Count by Class (Pclass)")
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# 3. Relationship between 'Survived' and 'Sex'
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2')
plt.title("Survival Count by Sex")
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()

# 4. Age distribution based on 'Survived'
plt.figure(figsize=(10, 6))
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True, color='blue', label='Survived', bins=30)
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True, color='red', label='Not Survived', bins=30)
plt.title("Age Distribution by Survival Status")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Optional: Save cleaned data to a new CSV file
df.to_csv('Titanic_cleaned_data.csv', index=False)

print("\nData cleaning and EDA completed. Cleaned data saved as 'Titanic_cleaned_data.csv'.")