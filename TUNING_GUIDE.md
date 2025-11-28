# Advanced Model Tuning Guide

Based on your dataset (21,472 rows, 73 features after preprocessing), here are optimized hyperparameter recommendations and strategies to achieve R² > 0.9 with minimal overfitting.

## Dataset Characteristics

- **Size**: 21,472 samples
- **Features**: 73 (after Stage 3 processing)
- **Target**: Property prices (continuous regression)
- **Key Features**: bathrooms, area, locality_te_price, postal_code_te_price

## Recommended Hyperparameter Ranges

### XGBoost (Best for: Structured data, interpretability)

**Optimal ranges for your dataset:**
```python
{
    'n_estimators': 300-800,        # More trees = better, but watch overfitting
    'max_depth': 4-8,               # Shallow trees reduce overfitting
    'learning_rate': 0.02-0.1,      # Lower = better generalization
    'min_child_weight': 3-10,       # Higher = more conservative
    'subsample': 0.7-0.9,           # Row sampling for regularization
    'colsample_bytree': 0.7-0.9,    # Column sampling for regularization
    'gamma': 0-3,                   # Minimum loss reduction
    'reg_alpha': 1-5,               # L1 regularization
    'reg_lambda': 1-5,              # L2 regularization
}
```

**Why these ranges:**
- Your dataset has ~21k samples, so moderate tree depth (4-8) prevents overfitting
- Strong regularization (alpha/lambda 1-5) needed for R² stability
- Subsample 0.7-0.9 provides good variance reduction

### LightGBM (Best for: Speed, large datasets)

**Optimal ranges for your dataset:**
```python
{
    'n_estimators': 300-800,
    'max_depth': 4-8,
    'learning_rate': 0.02-0.1,
    'num_leaves': 31-100,           # 2^max_depth is typical
    'min_child_samples': 20-50,     # Higher for your dataset size
    'subsample': 0.7-0.9,
    'colsample_bytree': 0.7-0.9,
    'reg_alpha': 1-5,
    'reg_lambda': 1-5,
    'min_split_gain': 0-2,
}
```

**Why these ranges:**
- `num_leaves` should be < 2^max_depth to prevent overfitting
- `min_child_samples` 20-50 is appropriate for 21k samples
- LightGBM is faster than XGBoost, good for many Optuna trials

### CatBoost (Best for: Categorical features, robustness)

**Optimal ranges for your dataset:**
```python
{
    'iterations': 300-800,
    'depth': 4-8,
    'learning_rate': 0.02-0.1,
    'l2_leaf_reg': 3-10,            # Strong regularization
    'bagging_temperature': 0-1,     # Bayesian bootstrap
    'random_strength': 1-5,         # Randomness in splits
    'border_count': 128-254,        # Feature discretization
}
```

**Why these ranges:**
- CatBoost handles your target-encoded features well
- Higher `l2_leaf_reg` (3-10) reduces overfitting
- `border_count` 128-254 balances speed and accuracy

## Ensemble Strategy

### Stacking (Recommended)

**Why stacking works for your data:**
1. **Diversity**: XGBoost, LightGBM, CatBoost have different strengths
2. **Variance reduction**: Averaging predictions reduces overfitting
3. **Meta-learner**: Ridge regression combines predictions optimally

**Configuration:**
```python
StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(...)),
        ('lgb', LGBMRegressor(...)),
        ('cat', CatBoostRegressor(...))
    ],
    final_estimator=Ridge(alpha=10),  # Strong regularization
    cv=5,                              # Cross-validation for meta-features
)
```

### Alternative: Weighted Average

If stacking doesn't improve results:
```python
# Based on validation performance
predictions = (
    0.35 * xgb_pred +
    0.35 * lgb_pred +
    0.30 * cat_pred
)
```

## Strategies to Achieve R² > 0.9, Overfitting < 0.1

### 1. **Early Stopping**
```python
# XGBoost
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50)

# LightGBM
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50)])
```

### 2. **Cross-Validation During Tuning**
```python
def objective(trial):
    params = {...}
    model = XGBRegressor(**params)
    
    # Use CV instead of single validation set
    scores = cross_val_score(model, X_train, y_train, 
                             cv=5, scoring='r2')
    return scores.mean()
```

### 3. **Feature Selection**
Your top features are already strong. Consider:
- Removing features with importance < 0.01
- Removing highly correlated features (correlation > 0.95)

### 4. **Regularization Tuning**
For your dataset, prioritize:
- **L2 regularization** (reg_lambda, l2_leaf_reg): 3-10
- **Subsampling**: 0.7-0.85 (not too low, you have limited data)
- **Learning rate**: 0.03-0.08 (lower = less overfitting)

### 5. **Data Split Strategy**
```python
# Recommended for your dataset
train: 70-75%  (15,000-16,000 samples)
val:   10-15%  (2,000-3,000 samples)
test:  15-20%  (3,000-4,000 samples)
```

## Expected Performance

Based on your current results and dataset:

| Model | Expected Train R² | Expected Test R² | Overfitting |
|-------|------------------|------------------|-------------|
| XGBoost (tuned) | 0.93-0.95 | 0.88-0.91 | 0.03-0.05 |
| LightGBM (tuned) | 0.93-0.95 | 0.88-0.91 | 0.03-0.05 |
| CatBoost (tuned) | 0.93-0.95 | 0.88-0.91 | 0.03-0.05 |
| **Ensemble** | **0.94-0.96** | **0.90-0.93** | **0.02-0.04** |

## Optuna Configuration

### Recommended Settings
```python
# Number of trials
N_TRIALS = 100  # Good balance of time/performance
# For production: 200-300 trials

# Sampler
sampler = TPESampler(
    seed=42,
    n_startup_trials=20,  # Random search first
    multivariate=True,     # Consider parameter interactions
)

# Pruner (optional, for faster tuning)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=10,
    n_warmup_steps=50,
)
```

### Objective Function Tips
```python
def objective(trial):
    params = {...}
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Validation score
    val_r2 = r2_score(y_val, model.predict(X_val))
    
    # Penalize overfitting
    train_r2 = r2_score(y_train, model.predict(X_train))
    overfitting = max(0, train_r2 - val_r2 - 0.05)
    
    # Return penalized score
    return val_r2 - (overfitting * 2)
```

## Usage

1. **Install dependencies:**
```bash
pip install optuna xgboost lightgbm catboost scikit-learn
```

2. **Run tuning:**
```bash
python optuna_tuning.py
```

3. **Monitor progress:**
- Optuna shows progress bar
- Best parameters printed at end
- Models saved automatically

4. **Adjust if needed:**
- If overfitting > 0.1: Increase regularization, reduce max_depth
- If R² < 0.9: Increase n_estimators, try different learning rates
- If training too slow: Reduce N_TRIALS or use MedianPruner

## Pro Tips

1. **Start with LightGBM** - fastest to tune, often best results
2. **Use GPU** if available - set `tree_method='gpu_hist'` for XGBoost
3. **Save study** - `study.trials_dataframe().to_csv('optuna_results.csv')`
4. **Visualize** - `optuna.visualization.plot_optimization_history(study)`
5. **Parallel tuning** - Run multiple studies with different seeds

## Troubleshooting

**If overfitting is high (> 0.1):**
- Increase `reg_alpha`, `reg_lambda` to 5-15
- Decrease `max_depth` to 3-5
- Increase `min_child_weight` / `min_child_samples`
- Lower `subsample` to 0.6-0.7

**If R² is low (< 0.85):**
- Check for data leakage removal (you've already done this ✓)
- Increase `n_estimators` to 1000-1500
- Try `learning_rate` 0.01-0.03 with more trees
- Add more features or feature interactions

**If training is too slow:**
- Use `tree_method='hist'` for XGBoost
- Reduce `N_TRIALS` to 50
- Use `MedianPruner` to stop bad trials early
- Run on GPU if available
