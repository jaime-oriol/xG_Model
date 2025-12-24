"""
Prepare predictions once for all visualizations
Evita reprocesar features/predictions en cada viz
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

from src.data.loader import load_shot_events
from src.features.geometric import calculate_basic_features
from src.features.freeze_frame import calculate_freeze_frame_features
from src.features.shot_height import calculate_shot_height_features

def prepare_predictions(model_path='models/phase2_tuned.json',
                       features_path='models/phase2_tuned_features.txt',
                       output_path='data/processed/predictions_cache.pkl'):
    """
    Process features and predictions once, save for visualizations

    Returns DataFrame with:
    - Original shot data (x, y, match_id, etc.)
    - All features
    - Model predictions (my_xg)
    - StatsBomb xG
    - is_goal
    """
    print("Loading shots...")
    df = load_shot_events()

    print("Calculating features...")
    df = calculate_basic_features(df)
    df = calculate_freeze_frame_features(df)
    df = calculate_shot_height_features(df)

    print("Encoding categoricals...")
    df = pd.get_dummies(df, columns=['body_part', 'technique', 'shot_type', 'shot_height_category'], drop_first=False)

    # Load feature list
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f]

    # Fill NaN in freeze_frame features
    freeze_frame_cols = [
        'keeper_distance_from_line', 'keeper_lateral_deviation', 'keeper_cone_blocked',
        'defenders_in_triangle', 'closest_defender_distance', 'defenders_within_5m',
        'defenders_in_shooting_lane'
    ]

    for col in freeze_frame_cols:
        if col in ['keeper_distance_from_line', 'keeper_lateral_deviation']:
            df[col] = df[col].fillna(0.0)
        elif col == 'closest_defender_distance':
            df[col] = df[col].fillna(999.0)
        else:
            df[col] = df[col].fillna(0)

    X = df[feature_cols]

    print("Loading model and predicting...")
    model = xgb.Booster()
    model.load_model(str(model_path))

    dmatrix = xgb.DMatrix(X, feature_names=feature_cols)
    my_xg = model.predict(dmatrix)

    # Add predictions to dataframe
    df['my_xg'] = my_xg

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_pickle(output_path)

    print(f"\n✓ Saved predictions for {len(df):,} shots to {output_path}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return df

if __name__ == '__main__':
    prepare_predictions()
