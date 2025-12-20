"""
Shot height feature engineering
"""
import numpy as np
import pandas as pd

def calculate_shot_height_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate shot height features from end_z coordinate

    Args:
        df: DataFrame with end_z, body_part, aerial_won, keeper features

    Returns:
        DataFrame with shot height features
    """
    df = df.copy()

    # Shot height (end_z)
    df['shot_height'] = df['end_z'].fillna(0.0)

    # Height categories
    df['shot_height_category'] = pd.cut(
        df['shot_height'],
        bins=[-np.inf, 0.5, 1.2, 1.8, np.inf],
        labels=['ground', 'low', 'medium', 'high']
    )

    # Header that won aerial duel
    df['is_header_aerial_won'] = (
        (df['body_part'] == 'Head') & (df['aerial_won'] == True)
    ).astype(int)

    # Keeper forward + high shot (vulnerable)
    if 'keeper_distance_from_line' in df.columns:
        df['keeper_forward_high_shot'] = (
            (df['keeper_distance_from_line'] < -2.0) &
            (df['shot_height'] > 1.8)
        ).astype(int)
    else:
        df['keeper_forward_high_shot'] = 0

    return df
