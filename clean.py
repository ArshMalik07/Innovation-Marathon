import numpy as np
import pandas as pd

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
