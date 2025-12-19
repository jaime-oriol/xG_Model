"""
Geometric feature engineering for xG model
"""
import numpy as np
import pandas as pd

# Field constants
GOAL_CENTER_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_POST_LEFT_Y = 36.16
GOAL_POST_RIGHT_Y = 43.84

def calculate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate geometric features from shot location

    Args:
        df: DataFrame with 'x' and 'y' columns

    Returns:
        DataFrame with added features
    """
    df = df.copy()

    # Distance to goal center
    df['distance_to_goal'] = np.sqrt(
        (GOAL_CENTER_X - df['x'])**2 + (GOAL_CENTER_Y - df['y'])**2
    )

    # Angle to goal
    df['angle_to_goal'] = _calculate_angle_to_goal(df['x'], df['y'])

    # Direct coordinates
    df['x_coordinate'] = df['x']
    df['y_deviation'] = np.abs(df['y'] - GOAL_CENTER_Y)

    # Distance to goal line
    df['distance_to_goal_line'] = GOAL_CENTER_X - df['x']

    return df

def _calculate_angle_to_goal(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Calculate angle between shot position and goal posts

    Returns:
        Angle in degrees
    """
    # Vectors to posts
    angle_left = np.arctan2(GOAL_POST_LEFT_Y - y, GOAL_CENTER_X - x)
    angle_right = np.arctan2(GOAL_POST_RIGHT_Y - y, GOAL_CENTER_X - x)

    # Absolute angle between vectors
    angle = np.abs(angle_left - angle_right)

    # Convert to degrees
    return angle * 180 / np.pi
