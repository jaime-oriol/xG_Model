"""
360 degree vision features from StatsBomb 360 data
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from typing import List, Dict, Optional

GOAL_LEFT_POST = np.array([120.0, 36.16])
GOAL_RIGHT_POST = np.array([120.0, 43.84])
GOAL_CENTER = np.array([120.0, 40.0])

def calculate_visible_area_size(visible_area: List[float]) -> float:
    """
    Calculate area of visible polygon in square yards

    Args:
        visible_area: List of [x1, y1, x2, y2, ...] coordinates

    Returns:
        Area in square yards
    """
    if not visible_area or len(visible_area) < 6:
        return 0.0

    # Convert flat list to list of points
    points = [(visible_area[i], visible_area[i+1])
              for i in range(0, len(visible_area), 2)]

    try:
        polygon = Polygon(points)
        return polygon.area
    except:
        return 0.0

def calculate_goal_visibility_score(visible_area: List[float],
                                    shot_x: float,
                                    shot_y: float) -> float:
    """
    Calculate what % of goal is visible from shot position

    Args:
        visible_area: Visible polygon coordinates
        shot_x, shot_y: Shot position

    Returns:
        Score 0-1 indicating goal visibility
    """
    if not visible_area or len(visible_area) < 6:
        return 1.0  # Assume full visibility if no data

    points = [(visible_area[i], visible_area[i+1])
              for i in range(0, len(visible_area), 2)]

    try:
        visible_polygon = Polygon(points)

        # Sample 20 points along goal line
        goal_samples = []
        for i in range(21):
            y = 36.16 + (43.84 - 36.16) * i / 20
            goal_samples.append(Point(120.0, y))

        # Count how many goal points are visible
        visible_count = sum(1 for p in goal_samples
                           if visible_polygon.contains(p))

        return visible_count / len(goal_samples)
    except:
        return 1.0

def calculate_pressure_density_360(visible_area: List[float],
                                   freeze_frame: List[Dict]) -> float:
    """
    Calculate density of opponent players in visible area

    Args:
        visible_area: Visible polygon coordinates
        freeze_frame: List of player positions from 360 data

    Returns:
        Opponents per 100 square yards in visible area
    """
    if not visible_area or len(visible_area) < 6 or not freeze_frame:
        return 0.0

    points = [(visible_area[i], visible_area[i+1])
              for i in range(0, len(visible_area), 2)]

    try:
        visible_polygon = Polygon(points)
        area = visible_polygon.area

        if area < 1.0:
            return 0.0

        # Count opponent players in visible area
        opponents_in_area = 0
        for player in freeze_frame:
            if not player.get('teammate', False) and not player.get('keeper', False):
                player_pos = Point(player['location'])
                if visible_polygon.contains(player_pos):
                    opponents_in_area += 1

        # Density per 100 sq yards
        return (opponents_in_area / area) * 100.0
    except:
        return 0.0

def calculate_three_sixty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 360 features

    Args:
        df: DataFrame with visible_area_360 and freeze_frame_360 columns

    Returns:
        DataFrame with 360 features added
    """
    df = df.copy()

    # Initialize features
    df['visible_area_size'] = 0.0
    df['goal_visibility_score'] = 1.0
    df['pressure_density_360'] = 0.0

    # Calculate features for rows with 360 data
    has_360 = df['visible_area_360'].notna()

    for idx in df[has_360].index:
        visible_area = df.loc[idx, 'visible_area_360']
        freeze_frame = df.loc[idx, 'freeze_frame_360']
        shot_x = df.loc[idx, 'shot_x']
        shot_y = df.loc[idx, 'shot_y']

        df.loc[idx, 'visible_area_size'] = calculate_visible_area_size(visible_area)
        df.loc[idx, 'goal_visibility_score'] = calculate_goal_visibility_score(
            visible_area, shot_x, shot_y
        )

        if freeze_frame is not None:
            df.loc[idx, 'pressure_density_360'] = calculate_pressure_density_360(
                visible_area, freeze_frame
            )

    return df
