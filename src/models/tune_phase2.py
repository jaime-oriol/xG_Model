"""
Hyperparameter tuning for Phase 2 using RandomizedSearchCV
Pure xG model (no post-shot features)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score, make_scorer
from pathlib import Path

from src.data.loader import load_shot_events
from src.features.geometric import calculate_basic_features
from src.features.freeze_frame import calculate_freeze_frame_features

def tune_phase2():
    """Hyperparameter tuning for Phase 2 (pure xG)"""
    print("Loading and preparing data...")
    df = load_shot_events()

    df = calculate_basic_features(df)
    df = calculate_freeze_frame_features(df)

    # One-hot encoding (NO shot_height_category - that's post-shot)
    df = pd.get_dummies(df, columns=['body_part', 'technique', 'shot_type'], drop_first=False)

    geometric_cols = ['distance_to_goal', 'angle_to_goal', 'x_coordinate', 'y_deviation', 'distance_to_goal_line']
    freeze_frame_cols = ['keeper_distance_from_line', 'keeper_lateral_deviation', 'keeper_cone_blocked',
                        'defenders_in_triangle', 'closest_defender_distance', 'defenders_within_5m', 'defenders_in_shooting_lane']
    one_hot_cols = [c for c in df.columns if c.startswith(('body_part_', 'technique_', 'shot_type_'))]

    feature_cols = geometric_cols + freeze_frame_cols + one_hot_cols

    # Fill NaN in freeze frame features
    for col in freeze_frame_cols:
        if col in ['keeper_distance_from_line', 'keeper_lateral_deviation']:
            df[col] = df[col].fillna(0.0)
        elif col == 'closest_defender_distance':
            df[col] = df[col].fillna(999.0)
        else:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]
    y = df['is_goal']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"Features: {len(feature_cols)}")

    # Parameter grid
    param_distributions = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'n_estimators': [200, 300, 400, 500],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_alpha': [0.1, 0.3, 0.5, 1.0],
        'reg_lambda': [1.0, 1.5, 2.0, 3.0],
        'min_child_weight': [1, 3, 5],
    }

    # Monotonicity constraints
    constraints = []
    for col in feature_cols:
        if col == 'distance_to_goal':
            constraints.append(-1)  # More distance → lower xG
        elif col == 'angle_to_goal':
            constraints.append(1)   # Better angle → higher xG
        elif col == 'y_deviation':
            constraints.append(-1)  # More lateral → lower xG
        elif col == 'distance_to_goal_line':
            constraints.append(-1)  # Further from line → lower xG
        elif col == 'keeper_lateral_deviation':
            constraints.append(1)   # Keeper more lateral → higher xG
        elif col == 'keeper_cone_blocked':
            constraints.append(-1)  # More blocked → lower xG
        elif col == 'defenders_in_triangle':
            constraints.append(-1)  # More defenders → lower xG
        elif col == 'closest_defender_distance':
            constraints.append(1)   # Defender further → higher xG
        elif col == 'defenders_within_5m':
            constraints.append(-1)  # More defenders close → lower xG
        elif col == 'defenders_in_shooting_lane':
            constraints.append(-1)  # More defenders blocking → lower xG
        else:
            constraints.append(0)   # No constraint (one-hot features)

    # XGBoost classifier with monotonicity constraints
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        monotone_constraints=tuple(constraints),
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # RandomizedSearch
    print("\nSearching best hyperparameters (5-fold CV, 50 iterations)...")
    print("This may take ~10-15 minutes...")
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions,
        n_iter=50,
        scoring='neg_brier_score',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("\n=== BEST HYPERPARAMETERS ===")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest CV Brier Score: {-random_search.best_score_:.4f}")

    # Evaluate best model on test set
    best_model = random_search.best_estimator_

    y_pred_train = best_model.predict_proba(X_train)[:, 1]
    y_pred_test = best_model.predict_proba(X_test)[:, 1]

    brier_train = brier_score_loss(y_train, y_pred_train)
    brier_test = brier_score_loss(y_test, y_pred_test)
    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_test = roc_auc_score(y_test, y_pred_test)

    print("\n=== TRAIN SET ===")
    print(f"Brier: {brier_train:.4f}")
    print(f"AUC: {auc_train:.4f}")

    print("\n=== TEST SET ===")
    print(f"Brier: {brier_test:.4f}")
    print(f"AUC: {auc_test:.4f}")

    print("\n=== OVERFITTING CHECK ===")
    print(f"Brier diff: {brier_test - brier_train:+.4f}")
    print(f"AUC diff: {auc_train - auc_test:+.4f}")

    if brier_test - brier_train < 0.005 and auc_train - auc_test < 0.02:
        print("✓ NO overfitting")
    elif brier_test - brier_train < 0.01 and auc_train - auc_test < 0.05:
        print("⚠ Slight overfitting - acceptable")
    else:
        print("✗ Significant overfitting")

    # Save best model
    model_path = Path("models/phase2_tuned.json")
    best_model.get_booster().save_model(str(model_path))
    print(f"\n✓ Best model saved to {model_path}")

    # Save feature list
    features_path = Path("models/phase2_tuned_features.txt")
    with open(features_path, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"✓ Features saved to {features_path}")

    # Save best params
    import json
    params_path = Path("models/phase2_best_params.json")
    with open(params_path, 'w') as f:
        json.dump(random_search.best_params_, f, indent=2)
    print(f"✓ Best params saved to {params_path}")

if __name__ == "__main__":
    tune_phase2()
