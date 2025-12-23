"""
Contextual feature engineering
"""
import pandas as pd

def calculate_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract contextual features already in data

    Args:
        df: DataFrame with first_time, under_pressure, one_on_one

    Returns:
        DataFrame with contextual features as int
    """
    df = df.copy()

    # Convert boolean to int
    df['first_time'] = df['first_time'].astype(int)
    df['under_pressure'] = df['under_pressure'].astype(int)
    df['one_on_one'] = df['one_on_one'].astype(int)

    return df
