import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import sys
import os

# Add project root to path to import utils
sys.path.append(os.getcwd())

from utils.stage3_utils import fit_stage3, transform_stage3, prepare_X_y

# ========================================
# CONFIG
# ========================================
DATA_PATH = "data/pre_processed/pre_processed_data_for_kaggle.csv"
TEST_SIZE = 0.15
RANDOM_STATE = 42

# ========================================
# 1) Load data
# ========================================
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# ========================================
# 2) Split BEFORE Stage 3
# ========================================
print("Splitting data...")
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ========================================
# 3) FIT Stage 3 only on training
# ========================================
print("Fitting Stage 3...")
fitted = fit_stage3(df_train)

# ========================================
# 4) TRANSFORM train + test using training statistics
# ========================================
print("Transforming data...")
df_train_s3 = transform_stage3(df_train, fitted)
df_test_s3  = transform_stage3(df_test, fitted)

# ========================================
# 5) Final X and y
# ========================================
X_train, y_train = prepare_X_y(df_train_s3)
X_test,  y_test  = prepare_X_y(df_test_s3)

print(f"Initial features ({len(X_train.columns)}): {list(X_train.columns)}")

# ========================================
# 6) Feature Selection Loop
# ========================================
current_features = list(X_train.columns)
results = []

print("\nStarting Recursive Feature Elimination...")
print(f"{'# Feats':<10} | {'Avg Val MAE':<15} | {'Removed Feature':<20}")
print("-" * 50)

while len(current_features) > 0:
    # Use only current features
    X_train_curr = X_train[current_features]
    
    # KFold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []
    
    for train_index, val_index in kf.split(X_train_curr):
        X_tr, X_va = X_train_curr.iloc[train_index], X_train_curr.iloc[val_index]
        y_tr, y_va = y_train.iloc[train_index], y_train.iloc[val_index]

        model = CatBoostRegressor(
            iterations=1369,
            depth=9,
            learning_rate=0.12920462180825767,
            l2_leaf_reg=0.8242326515788326,
            random_strength=2.2847008203805235,
            bagging_temperature=1.1151326396033874,
            border_count=166,
            loss_function="RMSE",
            random_state=42,
            verbose=False,
            task_type="CPU",
            allow_writing_files=False
        )

        model.fit(X_tr, y_tr)
        preds_val = model.predict(X_va)
        mae_val = mean_absolute_error(y_va, preds_val)
        mae_scores.append(mae_val)
    
    avg_mae = np.mean(mae_scores)
    
    # Train on full train set to get feature importance for next step
    # Only if we still have features to remove
    removed_feature = "None"
    if len(current_features) > 1:
        model_full = CatBoostRegressor(
            iterations=1369,
            depth=9,
            learning_rate=0.12920462180825767,
            l2_leaf_reg=0.8242326515788326,
            random_strength=2.2847008203805235,
            bagging_temperature=1.1151326396033874,
            border_count=166,
            loss_function="RMSE",
            random_state=42,
            verbose=False,
            task_type="CPU",
            allow_writing_files=False
        )
        model_full.fit(X_train_curr, y_train)
        
        # Get feature importances
        importances = model_full.get_feature_importance()
        feat_imp = pd.Series(importances, index=current_features)
        
        # Identify least important feature
        removed_feature = feat_imp.idxmin()
        
    results.append({
        "num_features": len(current_features),
        "avg_val_mae": avg_mae,
        "features": list(current_features),
        "removed_feature": removed_feature
    })
    
    print(f"{len(current_features):<10} | {avg_mae:<15.2f} | {removed_feature:<20}")
    
    if len(current_features) > 1:
        current_features.remove(removed_feature)
    else:
        break

# ========================================
# 7) Save and Display Results
# ========================================
results_df = pd.DataFrame(results)
results_df.to_csv("feature_selection_results.csv", index=False)
print("\nResults saved to feature_selection_results.csv")

# Find best MAE
best_run = results_df.loc[results_df["avg_val_mae"].idxmin()]
print("\nBest Configuration:")
print(f"Number of features: {best_run['num_features']}")
print(f"MAE: {best_run['avg_val_mae']:.2f}")
print(f"Features: {best_run['features']}")
