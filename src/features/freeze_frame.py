"""
Freeze frame feature engineering
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString

GOAL_CENTER_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_POST_LEFT_Y = 36.16
GOAL_POST_RIGHT_Y = 43.84

def calculate_freeze_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate freeze frame features: keeper, defenders

    Args:
        df: DataFrame with x, y, has_freeze_frame, freeze_frame

    Returns:
        DataFrame with freeze frame features
    """
    df = df.copy()

    df['keeper_distance_from_line'] = np.nan
    df['keeper_lateral_deviation'] = np.nan
    df['keeper_cone_blocked'] = 0.0
    df['defenders_in_triangle'] = 0
    df['closest_defender_distance'] = 999.0
    df['defenders_within_5m'] = 0
    df['defenders_in_shooting_lane'] = 0

    for idx, row in df.iterrows():
        if not row['has_freeze_frame'] or row['freeze_frame'] is None:
            continue

        freeze_frame = row['freeze_frame']
        shot_x, shot_y = row['x'], row['y']

        # Keeper
        keeper = next((p for p in freeze_frame
                      if p.get('position', {}).get('name') == 'Goalkeeper'
                      and not p.get('teammate', True)), None)

        if keeper:
            keeper_x, keeper_y = keeper['location']
            df.at[idx, 'keeper_distance_from_line'] = keeper_x - GOAL_CENTER_X
            df.at[idx, 'keeper_lateral_deviation'] = abs(keeper_y - GOAL_CENTER_Y)

            shot_angle = _calculate_angle(shot_x, shot_y)
            keeper_angle = _calculate_angle(keeper_x, keeper_y)
            df.at[idx, 'keeper_cone_blocked'] = min(keeper_angle, shot_angle) / max(shot_angle, 0.01)

        # Defenders
        defenders = [p for p in freeze_frame
                    if not p.get('teammate', True)
                    and p.get('position', {}).get('name') != 'Goalkeeper']

        if defenders:
            triangle = Polygon([
                [shot_x, shot_y],
                [GOAL_CENTER_X, GOAL_POST_LEFT_Y],
                [GOAL_CENTER_X, GOAL_POST_RIGHT_Y]
            ])

            shot_line = LineString([[shot_x, shot_y], [GOAL_CENTER_X, GOAL_CENTER_Y]])

            distances = []
            in_triangle = 0
            within_5m = 0
            in_lane = 0

            for defender in defenders:
                d_x, d_y = defender['location']
                d_point = Point(d_x, d_y)
                dist = np.sqrt((shot_x - d_x)**2 + (shot_y - d_y)**2)
                distances.append(dist)

                if triangle.contains(d_point):
                    in_triangle += 1
                if dist < 5.0:
                    within_5m += 1
                if shot_line.distance(d_point) < 2.0 and d_x > shot_x:
                    in_lane += 1

            df.at[idx, 'defenders_in_triangle'] = in_triangle
            df.at[idx, 'closest_defender_distance'] = min(distances) if distances else 999.0
            df.at[idx, 'defenders_within_5m'] = within_5m
            df.at[idx, 'defenders_in_shooting_lane'] = in_lane

    return df

def _calculate_angle(x: float, y: float) -> float:
    """Angle to goal"""
    angle_left = np.arctan2(GOAL_POST_LEFT_Y - y, GOAL_CENTER_X - x)
    angle_right = np.arctan2(GOAL_POST_RIGHT_Y - y, GOAL_CENTER_X - x)
    return abs(angle_left - angle_right) * 180 / np.pi
