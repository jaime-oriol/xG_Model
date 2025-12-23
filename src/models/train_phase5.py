"""
Train Phase 5: Transfer Learning with 360 Data

Approach:
- Start from Phase 3 pre-trained model
- Fine-tune on shots with 360 data
- Low learning rate to preserve learned patterns
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path

from src.data.loader import load_shot_events_with_360
from src.features.geometric import calculate_basic_features
from src.features.freeze_frame import calculate_freeze_frame_features
from src.features.shot_height import calculate_shot_height_features
from src.features.three_sixty import calculate_three_sixty_features

def train_phase5_model():
    """Train Phase 5: Transfer learning with 360 data"""
    print("Loading shot events with 360 data...")
    df = load_shot_events_with_360()

    print(f"Total shots: {len(df):,}")

    # Calculate features
    print("Calculating geometric features...")
    df = calculate_basic_features(df)

    print("Calculating freeze frame features...")
    df = calculate_freeze_frame_features(df)

    print("Calculating shot height features...")
    df = calculate_shot_height_features(df)

    print("Calculating 360 features...")
    df = calculate_three_sixty_features(df)

    # Filter to only shots with 360 data
    df_360 = df[df['visible_area_360'].notna()].copy()
    print(f"\nShots with 360 data: {len(df_360):,} ({len(df_360)/len(df)*100:.1f}%)")
    print(f"Goals: {df_360['is_goal'].sum():,} ({df_360['is_goal'].mean()*100:.1f}%)")

    if len(df_360) < 100:
        print("ERROR: Not enough 360 data for training")
        return

    # Encode categoricals
    print("\nEncoding categoricals...")
    df_360 = pd.get_dummies(df_360, columns=['body_part', 'technique', 'shot_type', 'shot_height_category'], drop_first=False)

    # Features Phase 3 + 360
    geometric_cols = [
        'distance_to_goal',
        'angle_to_goal',
        'x_coordinate',
        'y_deviation',
        'distance_to_goal_line',
    ]

    freeze_frame_cols = [
        'keeper_distance_from_line',
        'keeper_lateral_deviation',
        'keeper_cone_blocked',
        'defenders_in_triangle',
        'closest_defender_distance',
        'defenders_within_5m',
        'defenders_in_shooting_lane',
    ]

    shot_height_cols = [
        'shot_height',
        'is_header_aerial_won',
        'keeper_forward_high_shot',
    ]

    three_sixty_cols = [
        'visible_area_size',
        'goal_visibility_score',
        'pressure_density_360',
    ]

    one_hot_cols = [c for c in df_360.columns if c.startswith((
        'body_part_', 'technique_', 'shot_type_', 'shot_height_category_'
    ))]

    feature_cols = geometric_cols + freeze_frame_cols + shot_height_cols + three_sixty_cols + one_hot_cols

    # Handle NaN
    for col in freeze_frame_cols:
        if col in ['keeper_distance_from_line', 'keeper_lateral_deviation']:
            df_360[col] = df_360[col].fillna(0.0)
        elif col == 'closest_defender_distance':
            df_360[col] = df_360[col].fillna(999.0)
        else:
            df_360[col] = df_360[col].fillna(0)

    X = df_360[feature_cols]
    y = df_360['is_goal']
    statsbomb_xg = df_360['statsbomb_xg'].values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")

    # Train/Test split
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test, sb_xg_train, sb_xg_test = train_test_split(
        X, y, statsbomb_xg, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,} shots ({y_train.sum()} goals)")
    print(f"Test: {len(X_test):,} shots ({y_test.sum()} goals)")

    # Fine-tuning params (LOW learning rate for transfer learning)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'learning_rate': 0.01,  # LOW for fine-tuning
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'random_state': 42,
    }

    # Monotonicity constraints
    constraints = []
    for col in feature_cols:
        if col == 'distance_to_goal':
            constraints.append(-1)
        elif col == 'angle_to_goal':
            constraints.append(1)
        elif col == 'y_deviation':
            constraints.append(-1)
        elif col == 'distance_to_goal_line':
            constraints.append(-1)
        elif col == 'keeper_lateral_deviation':
            constraints.append(1)
        elif col == 'keeper_cone_blocked':
            constraints.append(-1)
        elif col == 'defenders_in_triangle':
            constraints.append(-1)
        elif col == 'closest_defender_distance':
            constraints.append(1)
        elif col == 'defenders_within_5m':
            constraints.append(-1)
        elif col == 'defenders_in_shooting_lane':
            constraints.append(-1)
        elif col == 'goal_visibility_score':
            constraints.append(1)
        elif col == 'pressure_density_360':
            constraints.append(-1)
        else:
            constraints.append(0)

    params['monotone_constraints'] = tuple(constraints)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Check if Phase 3 model exists for transfer learning
    phase3_model_path = Path("models/phase3_tuned.json")
    if phase3_model_path.exists():
        print(f"\n=== TRANSFER LEARNING FROM PHASE 3 ===")
        print(f"Loading pre-trained model: {phase3_model_path}")

        # Load Phase 3 model
        base_model = xgb.Booster()
        base_model.load_model(str(phase3_model_path))

        # Continue training (fine-tune)
        print("Fine-tuning with 360 data...")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,  # Fewer rounds for fine-tuning
            xgb_model=base_model,  # Continue from Phase 3
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=False
        )
    else:
        print("\n=== TRAINING FROM SCRATCH (no Phase 3 model found) ===")
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=400,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

    # Evaluate on TRAIN
    print("\n=== TRAIN SET ===")
    y_pred_train = model.predict(dtrain)
    brier_train = brier_score_loss(y_train, y_pred_train)
    auc_train = roc_auc_score(y_train, y_pred_train)

    valid_idx_train = ~np.isnan(sb_xg_train)
    corr_train = np.corrcoef(y_pred_train[valid_idx_train], sb_xg_train[valid_idx_train])[0, 1]
    sb_brier_train = brier_score_loss(y_train[valid_idx_train], sb_xg_train[valid_idx_train])

    print(f"Brier Score: {brier_train:.4f}")
    print(f"AUC-ROC: {auc_train:.4f}")
    print(f"Correlation vs StatsBomb: {corr_train:.4f}")
    print(f"StatsBomb Brier: {sb_brier_train:.4f}")

    # Evaluate on TEST
    print("\n=== TEST SET (HOLDOUT) ===")
    y_pred_test = model.predict(dtest)
    brier_test = brier_score_loss(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_pred_test)

    valid_idx_test = ~np.isnan(sb_xg_test)
    corr_test = np.corrcoef(y_pred_test[valid_idx_test], sb_xg_test[valid_idx_test])[0, 1]
    sb_brier_test = brier_score_loss(y_test[valid_idx_test], sb_xg_test[valid_idx_test])

    print(f"Brier Score: {brier_test:.4f}")
    print(f"AUC-ROC: {auc_test:.4f}")
    print(f"Correlation vs StatsBomb: {corr_test:.4f}")
    print(f"StatsBomb Brier: {sb_brier_test:.4f}")

    # Overfitting check
    print("\n=== OVERFITTING CHECK ===")
    brier_diff = brier_test - brier_train
    auc_diff = auc_train - auc_test

    print(f"Brier difference (test - train): {brier_diff:+.4f}")
    print(f"AUC difference (train - test): {auc_diff:+.4f}")

    if brier_diff < 0.005 and auc_diff < 0.02:
        print("NO overfitting - model generalizes well")
    elif brier_diff < 0.01 and auc_diff < 0.05:
        print("Slight overfitting - acceptable")
    else:
        print("Significant overfitting - model may not generalize")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Test Brier: {brier_test:.4f}")
    print(f"Test beats StatsBomb: {brier_test < sb_brier_test}")
    print(f"Improvement vs StatsBomb: {sb_brier_test - brier_test:.4f}")

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "phase5_360.json"
    model.save_model(str(model_path))

    with open(model_dir / "phase5_features.txt", "w") as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")

    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_phase5_model()
