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
import warnings
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Define functions (these can stay outside the guard)
def objective(trial, X, y, scale_pos_weight):
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
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
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

def get_objective(X_train, y_train, scale_pos_weight):
    def obj(trial):
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
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            model = xgb.train(
                params,
                dtr,
                num_boost_round=1000,
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            preds = model.predict(dval)
            auc = roc_auc_score(y_val, preds)
            aucs.append(auc)
        return np.mean(aucs)
    return obj

def train_with_optuna(n, X, y, X_pred, df_pred, importances_df, scale_pos_weight):
    try:
        logger.info(f"Starting train_with_optuna for n={n}")
        top_features = importances_df.head(n)["feature"].tolist()
        X_top = X[top_features]
        X_pred_top = X_pred[top_features]

        assert not X_top.isnull().any().any(), f"NaN values in X_top for n={n}"
        assert np.isfinite(X_top.select_dtypes(include=[np.number]).to_numpy()).all(), f"Infinite values in X_top for n={n}"

        X_t, X_val, y_t, y_val = train_test_split(X_top, y, test_size=0.2, stratify=y, random_state=42)
        dfinal_pred_top = xgb.DMatrix(X_pred_top)
        dtrain_top = xgb.DMatrix(X_t, label=y_t)
        dval_top = xgb.DMatrix(X_val, label=y_val)

        study = optuna.create_study(direction="maximize")
        study.optimize(get_objective(X_top, y, scale_pos_weight), n_trials=10)
        best_params = study.best_params
        best_params.update({
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'predictor': 'cpu_predictor',
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'auc',
            'verbosity': 0
        })

        model_top = xgb.train(
            best_params,
            dtrain_top,
            num_boost_round=1000,
            evals=[(dval_top, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_val_pred = model_top.predict(dval_top)
        auc = roc_auc_score(y_val, y_val_pred)

        fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred_proba = model_top.predict(dfinal_pred_top)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        submission = df_pred[['SK_ID_CURR']].copy()
        submission["TARGET_PROB"] = y_pred_proba
        submission["TARGET"] = y_pred
        filename = f"predicted_results_XGBoost_top_{n}.csv"
        submission.to_csv(filename, index=False)

        logger.info(f"[Top {n}] AUC: {auc:.4f} | Saved to {filename}")
        return (n, auc)
    except Exception as e:
        logger.error(f"Error in train_with_optuna for n={n}: {str(e)}")
        raise

# Main execution block
if __name__ == "__main__":
    # Load data
    df_train = pd.read_csv("processed_train.csv")
    X_pred = pd.read_csv("processed_test.csv")
    df_pred = pd.read_csv("data_to_score.csv")

    # Split features and target
    X = df_train.drop(columns=["TARGET"])
    y = df_train["TARGET"]

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    dfull = xgb.DMatrix(X, label=y)
    dfinal_pred = xgb.DMatrix(X_pred)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Run Optuna for initial model
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, scale_pos_weight), n_trials=50)

    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

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

    X_t, X_val, y_t, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
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

    y_val_pred = final_model.predict(dval_final)
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")

    y_pred_proba_final = final_model.predict(dfinal_pred)
    y_pred_final = (y_pred_proba_final >= optimal_threshold).astype(int)

    submission = df_pred[['SK_ID_CURR']].copy()
    submission["TARGET_PROB"] = y_pred_proba_final
    submission["TARGET"] = y_pred_final
    submission.to_csv("predicted_results_XGBoost.csv", index=False)
    logger.info("Saved results to predicted_results_XGBoost.csv")

    # Feature importances
    importances = final_model.get_score(importance_type='gain')
    importances_df = pd.DataFrame(list(importances.items()), columns=["feature", "importance"])
    importances_df.sort_values(by="importance", ascending=False, inplace=True)

    # Parallel training for simpler models
    top_n_list = [25, 50, 75, 100]
    results = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(train_with_optuna, n, X, y, X_pred, df_pred, importances_df, scale_pos_weight)
                   for n in top_n_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Future failed: {str(e)}")

    # Show results
    results.sort()
    for n, auc in results:
        logger.info(f"Top {n} features: AUC = {auc:.4f}")
