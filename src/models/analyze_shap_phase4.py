"""
SHAP analysis: Why contextual features don't help
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path

from src.data.loader import load_shot_events
from src.features.geometric import calculate_basic_features
from src.features.freeze_frame import calculate_freeze_frame_features
from src.features.shot_height import calculate_shot_height_features
from src.features.contextual import calculate_contextual_features

def analyze_shap_phase4():
    """Analyze SHAP values for Phase 4"""
    print("Loading shot events...")
    df = load_shot_events()

    print("Calculating all features...")
    df = calculate_basic_features(df)
    df = calculate_freeze_frame_features(df)
    df = calculate_shot_height_features(df)
    df = calculate_contextual_features(df)

    df = pd.get_dummies(df, columns=['body_part', 'technique', 'shot_type', 'shot_height_category'], drop_first=False)

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

    contextual_cols = [
        'first_time',
        'under_pressure',
        'one_on_one',
    ]

    one_hot_cols = [c for c in df.columns if c.startswith((
        'body_part_', 'technique_', 'shot_type_', 'shot_height_category_'
    ))]

    feature_cols = geometric_cols + freeze_frame_cols + shot_height_cols + contextual_cols + one_hot_cols

    for col in freeze_frame_cols:
        if col in ['keeper_distance_from_line', 'keeper_lateral_deviation']:
            df[col] = df[col].fillna(0.0)
        elif col == 'closest_defender_distance':
            df[col] = df[col].fillna(999.0)
        else:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]
    y = df['is_goal']

    # Load Phase 4 model
    model_path = Path("models/phase4_contextual.json")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = xgb.Booster()
    model.load_model(str(model_path))

    # XGBoost native feature importance
    print("\n=== XGBOOST FEATURE IMPORTANCE (Gain) ===")
    importance_dict = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame([
        {'feature': k, 'gain': v}
        for k, v in importance_dict.items()
    ]).sort_values('gain', ascending=False)

    print("\nTop 20 features:")
    print(importance_df.head(20).to_string(index=False))

    print("\nContextual features importance:")
    contextual_importance = importance_df[importance_df['feature'].isin(contextual_cols)]
    if len(contextual_importance) > 0:
        print(contextual_importance.to_string(index=False))
    else:
        print("NO contextual features in top importance!")

    # SHAP analysis (sample 2000 shots for speed)
    print("\n=== SHAP ANALYSIS ===")
    print("Sampling 2000 shots for SHAP analysis...")
    sample_idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    print("Computing SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP values
    shap_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 20 features by SHAP:")
    print(shap_importance.head(20).to_string(index=False))

    print("\nContextual features SHAP:")
    contextual_shap = shap_importance[shap_importance['feature'].isin(contextual_cols)]
    print(contextual_shap.to_string(index=False))

    # Rank contextual features
    for feat in contextual_cols:
        rank = shap_importance[shap_importance['feature'] == feat].index[0] + 1
        total = len(shap_importance)
        print(f"{feat}: rank {rank}/{total}")

    # Distribution analysis
    print("\n=== CONTEXTUAL FEATURES DISTRIBUTION ===")
    for feat in contextual_cols:
        count = df[feat].sum()
        pct = df[feat].mean() * 100
        print(f"{feat}: {count:,} / {len(df):,} ({pct:.1f}%)")

    # Goal rate by contextual feature
    print("\n=== GOAL RATE BY CONTEXTUAL FEATURE ===")
    for feat in contextual_cols:
        goal_rate_yes = df[df[feat] == 1]['is_goal'].mean() * 100
        goal_rate_no = df[df[feat] == 0]['is_goal'].mean() * 100
        diff = goal_rate_yes - goal_rate_no
        print(f"{feat}:")
        print(f"  Yes: {goal_rate_yes:.1f}%")
        print(f"  No: {goal_rate_no:.1f}%")
        print(f"  Diff: {diff:+.1f}%")

if __name__ == "__main__":
    analyze_shap_phase4()
