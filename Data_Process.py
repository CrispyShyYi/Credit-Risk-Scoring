#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 10:40:04 2025

@author: Jialiang Wu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Read the CSV file
df = pd.read_csv("data_devsample.csv")
df_pred = pd.read_csv("data_to_score.csv")

# Check for missing values in the 'TARGET' column
missing_target_rows = df[df['TARGET'].isna()]

# Print how many rows have missing TARGET
print(f"Number of rows with missing TARGET: {missing_target_rows.shape[0]}")

# Display the first 5 rows (show the structure of the data frame)
# print(df.head())

#############################################################################


# Define columns to drop from features
drop_cols = ['TARGET', 'SK_ID_CURR', 'TIME', 'BASE', 'DAY']

# Split features and target
X = df.drop(columns=drop_cols)
y = df['TARGET']

# Create a summary DataFrame
summary = pd.DataFrame({
    'Data Type': X.dtypes,
    'Unique Values': X.nunique()
})

# Show all unique data types and how many columns fall into each type
type_counts = X.dtypes.value_counts()

print("Column types and their counts:")
print(type_counts)


# Calculate proportion of NAs per column
na_proportion = X.isna().mean().sort_values(ascending=False)

# keep columns with < 50% NAs
X = X.loc[:, X.isna().mean() < 0.5]                                           # Adjust as needed

# Detect categorical columns (dtype == 'object' or 'category')
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# Fill missing values in those columns with "Missing"
X[cat_cols] = X[cat_cols].fillna("Missing")

# One-hot encode the categorical variables
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Save the column names
kept_columns = X.columns

# Identify numerical columns
num_cols = X.select_dtypes(include=['number']).columns

# Optional: Check how many NAs remain in categorical columns
# print(X[cat_cols].isna().sum())

# Replace inf/-inf with NaN in numeric columns only
X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

# Standardize numerical columns for KNN imputation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X[num_cols]), 
                        columns=num_cols, index=X.index)

# Apply KNN imputation
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X_scaled), 
                         columns=num_cols, index=X.index)

# Inverse transform to return numerical columns to original scale
X[num_cols] = scaler.inverse_transform(X_imputed)

# Check after dealing with missing values
X.isna().sum().sum()


# Do the same process for date_to_score file
drop_cols_test = ['SK_ID_CURR', 'TIME', 'BASE', 'DAY']
X_pred = df_pred.drop(columns=drop_cols_test)


# Fill missing values in categorical columns with "Missing"
X_pred[cat_cols] = X_pred[cat_cols].fillna("Missing")

# One-hot encode the categorical variables
X_pred = pd.get_dummies(X_pred, drop_first=True)

# Align columns to training
X_pred = X_pred.reindex(columns=kept_columns, fill_value=0)

# Replace inf/-inf with NaN in numeric columns only
X_pred[num_cols] = X_pred[num_cols].replace([np.inf, -np.inf], np.nan)

# Standardize numerical columns for KNN imputation
X_pred_scaled = pd.DataFrame(scaler.transform(X_pred[num_cols]), 
                             columns=num_cols, index=X_pred.index)
X_pred_imputed = pd.DataFrame(imputer.transform(X_pred_scaled), 
                              columns=num_cols, index=X_pred.index)

# Inverse transform back to original scale
X_pred[num_cols] = scaler.inverse_transform(X_pred_imputed)

# Add target back for training set
X['TARGET'] = y

# Save processed files
X.to_csv("processed_train.csv", index=False)
X_pred.to_csv("processed_test.csv", index=False)