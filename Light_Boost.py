#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 14:30:16 2025

@author: Jialiang Wu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import optuna
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

# Clean feature names to be LightGBM-safe
def clean_feature_names(df):
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    return df

# Read the processed data
df_train = pd.read_csv("processed_train.csv")
X_pred = pd.read_csv("processed_test.csv")

# Reload original df_pred to get SK_ID_CURR column for final output
df_pred = pd.read_csv("data_to_score.csv")

# Split features and target
X = df_train.drop(columns=["TARGET"])
y = df_train["TARGET"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Define fixed parameters
fixed_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'scale_pos_weight': scale_pos_weight
}

# Apply cleaning
X_train = clean_feature_names(X_train)
X_test = clean_feature_names(X_test)
X = clean_feature_names(X)
X_pred = clean_feature_names(X_pred)

# Clean Output
warnings.filterwarnings("ignore")

# Optuna objective
def objective(trial):
    tuned_params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'num_leaves': trial.suggest_int("num_leaves", 15, 256),
        'max_depth': trial.suggest_int("max_depth", 3, 15),
        'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    }
    params = {**fixed_params, **tuned_params}
    model = lgb.LGBMClassifier(**params, n_estimators=1000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)]
    )
    preds = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, preds)

# Auto optimize parameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print the best params
print("Best AUC: ", study.best_value)
print("Best params: ", study.best_params)

# Train final model
best_params = {**fixed_params, **study.best_params}
best_model = lgb.LGBMClassifier(**best_params, n_estimators=1000)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(stopping_rounds=100), log_evaluation(0)]
)

# Prediction on validation set
y_valid_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Compute optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_valid_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")

# Predict on test data
y_pred_proba_final = best_model.predict_proba(X_pred)[:, 1]
y_pred_final = (y_pred_proba_final >= optimal_threshold).astype(int)

# Save submission
submission = df_pred[['SK_ID_CURR']].copy()
submission["TARGET_PROB"] = y_pred_proba_final
submission["TARGET"] = y_pred_final
submission.to_csv("predicted_results_LightGBM.csv", index=False)
print("Saved results to predicted_results_LightGBM.csv")