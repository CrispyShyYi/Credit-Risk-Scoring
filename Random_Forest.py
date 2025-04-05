import pandas as pd
import numpy as np
import optuna
import optuna.visualization as vis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import warnings

# Read the processed data
df_train = pd.read_csv("processed_train.csv")
X_pred = pd.read_csv("processed_test.csv")

# Reload original df_pred to get SK_ID_CURR column for final output
df_pred = pd.read_csv("data_to_score.csv")

# Split features and target
X = df_train.drop(columns=["TARGET"])
y = df_train["TARGET"]

# Optionally split into train/test sets (for validation purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# Clean Output
warnings.filterwarnings("ignore")


# Random Forest

# Define the objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    # Initialize model with trial parameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Use 5-fold cross-validation with ROC AUC scoring
    auc_scores = cross_val_score(clf, X_train, y_train, cv=5, 
                                 scoring="roc_auc", error_score='raise')
    return auc_scores.mean()

# Create study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=600)  # 50 trials or max 10 mins

# Show best trial results
print("Best trial:")
print(f"  Value (ROC AUC): {study.best_value:.4f}")
print("  Params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
    
# Train the final model on full training set with best params
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

# Evaluate on test set
y_test_pred = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred)

print(f"Test ROC AUC: {test_auc:.4f}")

# Get TPR, FPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

# Find the index of the point closest to (0, 1)
optimal_idx = np.argmax(tpr - fpr)  # Youden’s J = TPR - FPR
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold based on ROC (Youden's J): {optimal_threshold:.4f}")

# shows the progress of optimization over time
vis.plot_optimization_history(study).show(renderer="browser")

# shows how important each hyperparameter was to the model’s performance
vis.plot_param_importances(study).show(renderer="browser")

# Predict:
# probability of being in class 1
y_pred_proba = best_model.predict_proba(X_pred)[:, 1]

# predicted class (e.g., 0 or 1)
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

results = pd.DataFrame({
    "SK_ID_CURR": df_pred["SK_ID_CURR"],  # restore the ID
    "TARGET": y_pred,
    "TARGET_PROB": y_pred_proba
})

# Save to CSV
results.to_csv("predicted_results_RF.csv", index=False)


##############################################################################

# Simpler Random Forest Model

# Get top 50 important features
importances = best_model.feature_importances_
indices = importances.argsort()[-50:][::-1]  # Top 50 features

# Feature names corresponding to top 50 importances
top_features = X_train.columns[indices]

# Filter training, test, and prediction sets ===
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
X_pred_top = X_pred[top_features]

# Define a simpler Optuna-tuned Random Forest ===
def simple_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    auc_scores = cross_val_score(clf, X_train_top, y_train, cv=5, 
                                 scoring="roc_auc", error_score='raise')
    return auc_scores.mean()


# Run optimization
simple_study = optuna.create_study(direction="maximize")
simple_study.optimize(simple_objective, n_trials=30, timeout=300)  # Shorter search

# Train best model on top 50 features
simple_best_model = RandomForestClassifier(**simple_study.best_params, 
                                           random_state=42, n_jobs=-1)
simple_best_model.fit(X_train_top, y_train)

# Evaluate
y_test_pred_simple = simple_best_model.predict_proba(X_test_top)[:, 1]
test_auc_simple = roc_auc_score(y_test, y_test_pred_simple)
print(f"[Top 50 features] Test ROC AUC: {test_auc_simple:.4f}")

# Prediction
y_pred_proba_simple = simple_best_model.predict_proba(X_pred_top)[:, 1]

# Thresholding (optional: reuse threshold logic from before)
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_simple)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold_simple = thresholds[optimal_idx]

y_pred_simple = (y_pred_proba_simple >= optimal_threshold_simple).astype(int)

# Final Results
results_simple = pd.DataFrame({
    "SK_ID_CURR": df_pred["SK_ID_CURR"],
    "TARGET": y_pred_simple,
    "TARGET_PROB": y_pred_proba_simple
})

results_simple.to_csv("predicted_results_top50_RF.csv", index=False)
print("Saved: predicted_results_top50_RF.csv")



