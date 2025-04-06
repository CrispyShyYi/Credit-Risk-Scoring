#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:10:10 2025

@author: Jialiang Wu
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.combine import SMOTEENN  # For SMOTEENN
from sklearn.decomposition import PCA  # For PCA
import warnings

# Load data
df_train = pd.read_csv("processed_train.csv")
X_pred = pd.read_csv("processed_test.csv")
df_pred = pd.read_csv("data_to_score.csv")

# Split features and target
X = df_train.drop(columns=["TARGET"])
y = df_train["TARGET"]

# Apply SMOTEENN
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X, y)
print(f"Original dataset shape: {X.shape}")
print(f"Resampled dataset shape: {X_resampled.shape}")

# Apply PCA (e.g., retain 95% of variance)
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_resampled)
X_pred_pca = pca.transform(X_pred)  # Transform test data with the same PCA
print(f"PCA reduced dataset shape: {X_pca.shape}")
print(f"Number of components: {pca.n_components_}")

# Train/validation split on resampled data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
dfull = xgb.DMatrix(X_pca, label=y_resampled)
dfinal_pred = xgb.DMatrix(X_pred_pca)

# Clean Output
warnings.filterwarnings("ignore")

# Since SMOTEENN balances the classes, scale_pos_weight may not be needed
scale_pos_weight = 1  # Adjust if imbalance persists
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Optuna objective function (unchanged except for data)
def objective(trial):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'predictor': 'cpu_predictor',
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 1.0),
        'gamma': trial.suggest_float("gamma", 0, 5),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'auc',
        'verbosity': 0
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_pca, y_resampled):
        X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
        y_tr, y_val = y_resampled[train_idx], y_resampled[val_idx]
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        model = xgb.train(
            params,
            dtr,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        preds = model.predict(dval)
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
    return np.mean(aucs)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best params
print("Best AUC: ", study.best_value)
print("Best params: ", study.best_params)

# Retrain on full dataset
final_params = study.best_params
final_params.update({
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'predictor': 'cpu_predictor',
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'auc',
    'verbosity': 0
})

X_t, X_val, y_t, y_val = train_test_split(
    X_pca, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)
dtrain_final = xgb.DMatrix(X_t, label=y_t)
dval_final = xgb.DMatrix(X_val, label=y_val)

final_model = xgb.train(
    final_params,
    dtrain_final,
    num_boost_round=1000,
    evals=[(dval_final, "val")],
    early_stopping_rounds=50,
    verbose_eval=False
)

# Compute optimal threshold
y_val_pred = final_model.predict(dval_final)
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.4f}")

# Predict test set
y_pred_proba_final = final_model.predict(dfinal_pred)
y_pred_final = (y_pred_proba_final >= optimal_threshold).astype(int)

# Save submission
submission = df_pred[['SK_ID_CURR']].copy()
submission["TARGET_PROB"] = y_pred_proba_final
submission["TARGET"] = y_pred_final
submission.to_csv("predicted_results_XGBoost.csv", index=False)
print("Saved the results to predicted_results_XGBoost.csv")