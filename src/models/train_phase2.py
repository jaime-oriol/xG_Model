"""
Train Phase 2: Geometric + Freeze Frame features
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from pathlib import Path

from src.data.loader import load_shot_events
from src.features.geometric import calculate_basic_features
from src.features.freeze_frame import calculate_freeze_frame_features

def train_phase2_model():
    """Train Phase 2: Phase 1 + Freeze Frames"""
    print("Loading shot events...")
    df = load_shot_events()

    print(f"Total shots: {len(df):,}")
    print(f"Valid coordinates: {df['x'].notna().sum():,}")

    # Phase 1: Geometric features
    print("Calculating geometric features...")
    df = calculate_basic_features(df)

    # Phase 2: Freeze frame features
    print("Calculating freeze frame features...")
    df = calculate_freeze_frame_features(df)

    # Encode categoricals
    print("Encoding categoricals...")
    df = pd.get_dummies(df, columns=['body_part', 'technique', 'shot_type'], drop_first=False)

    # Prepare features
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

    one_hot_cols = [c for c in df.columns if c.startswith(('body_part_', 'technique_', 'shot_type_'))]

    feature_cols = geometric_cols + freeze_frame_cols + one_hot_cols

    # Handle NaN in freeze frame features (shots without freeze_frame)
    for col in freeze_frame_cols:
        if col in ['keeper_distance_from_line', 'keeper_lateral_deviation']:
            df[col] = df[col].fillna(0.0)
        elif col == 'closest_defender_distance':
            df[col] = df[col].fillna(999.0)
        else:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]
    y = df['is_goal']

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")
    print(f"Goals: {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"Freeze frames available: {df['has_freeze_frame'].sum():,} ({df['has_freeze_frame'].mean()*100:.1f}%)")

    # XGBoost params
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
    }
    num_rounds = 500

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
        else:
            constraints.append(0)

    params['monotone_constraints'] = tuple(constraints)

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)

    print("\nTraining XGBoost...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        verbose_eval=False
    )

    # Evaluate
    print("\nEvaluating...")
    y_pred = model.predict(dtrain)

    brier = brier_score_loss(y, y_pred)
    auc = roc_auc_score(y, y_pred)

    print(f"\nMetrics vs Actual:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    # Compare vs StatsBomb
    statsbomb_xg = df['statsbomb_xg'].values
    valid_idx = ~np.isnan(statsbomb_xg)

    correlation = np.corrcoef(y_pred[valid_idx], statsbomb_xg[valid_idx])[0, 1]
    mean_abs_diff = np.mean(np.abs(y_pred[valid_idx] - statsbomb_xg[valid_idx]))
    sb_brier = brier_score_loss(y[valid_idx], statsbomb_xg[valid_idx])

    print(f"\nComparison vs StatsBomb:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Mean Abs Diff: {mean_abs_diff:.4f}")
    print(f"  Our Brier: {brier:.4f}")
    print(f"  StatsBomb Brier: {sb_brier:.4f}")

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "phase2_freeze_frames.json"
    model.save_model(str(model_path))

    # Save feature list
    with open(model_dir / "phase2_features.txt", "w") as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")

    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_phase2_model()
