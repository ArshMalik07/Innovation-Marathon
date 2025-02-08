import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
data = pd.read_csv(r"D:\Projects\Innovation Marathon Project\large_soil_plant_data.csv")

# Basic Information
print("Dataset Information:")
print(data.info())

# Summary Statistics (Numerical Columns)
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values[missing_values > 0])  # Show only columns with missing values

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()
print("\nTotal Duplicate Rows:", duplicate_rows)

# Check for mixed data types in each column
mixed_columns = data.columns[data.applymap(type).nunique() > 1]
print("\nColumns with mixed data types:", mixed_columns.tolist())

# Unique values in categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nUnique values in categorical columns:")
for col in categorical_columns:
    print(f"{col}: {data[col].nunique()} unique values")

# Correlation Analysis
print("\nCorrelation Matrix:")
print(data.corr(numeric_only=True))

# Value Counts for Categorical Features
for col in categorical_columns:
    print(f"\nValue counts for {col}:")
    print(data[col].value_counts().head(10))  # Show top 10 values

# Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Histogram for numerical features
data.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplot for detecting outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.select_dtypes(include=['number']))
plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Features")
plt.show()

# Count plot for categorical variables
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=data[col], order=data[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()

print("EDA completed successfully!")
