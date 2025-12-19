"""
Train baseline xG model (Phase 1)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss, roc_auc_score
import pickle
from pathlib import Path

from src.data.loader import load_shot_events
from src.features.geometric import calculate_basic_features

def train_baseline_model():
    """Train Phase 1 baseline model"""

    print("Loading shot events...")
    df = load_shot_events()
    print(f"Total shots: {len(df):,}")

    # Remove invalid coordinates
    df = df.dropna(subset=['x', 'y'])
    print(f"Valid coordinates: {len(df):,}")

    # Calculate geometric features
    print("Calculating geometric features...")
    df = calculate_basic_features(df)

    # One-hot encode categoricals
    print("Encoding categoricals...")
    df = _encode_categoricals(df)

    # Prepare features
    feature_cols = [
        'distance_to_goal',
        'angle_to_goal',
        'x_coordinate',
        'y_deviation',
        'distance_to_goal_line',
    ]

    # Add one-hot columns
    one_hot_cols = [c for c in df.columns if c.startswith(('body_part_', 'technique_', 'shot_type_'))]
    feature_cols.extend(one_hot_cols)

    X = df[feature_cols]
    y = df['is_goal']

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")
    print(f"Goals: {y.sum():,} ({y.mean()*100:.1f}%)")

    # Train XGBoost
    print("\nTraining XGBoost...")

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

    # Monotonicity constraints
    constraints = []
    for col in feature_cols:
        if col == 'distance_to_goal':
            constraints.append(-1)
        elif col == 'angle_to_goal':
            constraints.append(1)
        elif col == 'distance_to_goal_line':
            constraints.append(-1)
        else:
            constraints.append(0)

    params['monotone_constraints'] = tuple(constraints)

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        verbose_eval=50
    )

    # Predict
    print("\nEvaluating...")
    y_pred = model.predict(dtrain)

    # Metrics vs actual outcomes
    brier = brier_score_loss(y, y_pred)
    auc = roc_auc_score(y, y_pred)

    print(f"\nMetrics vs Actual:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    # Metrics vs StatsBomb
    statsbomb_xg = df['statsbomb_xg'].values
    valid_idx = ~np.isnan(statsbomb_xg)

    correlation = np.corrcoef(y_pred[valid_idx], statsbomb_xg[valid_idx])[0, 1]
    mean_diff = np.mean(np.abs(y_pred[valid_idx] - statsbomb_xg[valid_idx]))

    brier_sb = brier_score_loss(y[valid_idx], statsbomb_xg[valid_idx])

    print(f"\nComparison vs StatsBomb:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  Mean Abs Diff: {mean_diff:.4f}")
    print(f"  Our Brier: {brier:.4f}")
    print(f"  StatsBomb Brier: {brier_sb:.4f}")

    # Save model
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    model.save_model(str(output_dir / "phase1_baseline.json"))
    print(f"\nModel saved to models/phase1_baseline.json")

    # Save feature list
    with open(output_dir / "phase1_features.txt", 'w') as f:
        f.write("\n".join(feature_cols))

    return model, df, y_pred

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features"""
    df = df.copy()

    # Body part
    body_dummies = pd.get_dummies(df['body_part'], prefix='body_part')
    df = pd.concat([df, body_dummies], axis=1)

    # Technique
    tech_dummies = pd.get_dummies(df['technique'], prefix='technique')
    df = pd.concat([df, tech_dummies], axis=1)

    # Shot type
    type_dummies = pd.get_dummies(df['shot_type'], prefix='shot_type')
    df = pd.concat([df, type_dummies], axis=1)

    return df

if __name__ == "__main__":
    train_baseline_model()
