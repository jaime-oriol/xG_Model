"""
xG Heatmap Visualization for XGBoost Model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mplsoccer import VerticalPitch
import xgboost as xgb
from pathlib import Path

# Field constants (StatsBomb coordinates)
PITCH_LENGTH_X = 120.0
PITCH_WIDTH_Y = 80.0
GOAL_CENTER_X = 120.0
GOAL_CENTER_Y = 40.0
GOAL_POST_LEFT_Y = 36.16
GOAL_POST_RIGHT_Y = 43.84
GOAL_WIDTH_Y = GOAL_POST_RIGHT_Y - GOAL_POST_LEFT_Y

def calculate_features(x, y):
    """Calculate features for given x, y position"""
    distance = np.sqrt((GOAL_CENTER_X - x)**2 + (GOAL_CENTER_Y - y)**2)

    angle_left = np.arctan2(GOAL_POST_LEFT_Y - y, GOAL_CENTER_X - x)
    angle_right = np.arctan2(GOAL_POST_RIGHT_Y - y, GOAL_CENTER_X - x)
    angle = np.abs(angle_left - angle_right) * 180 / np.pi

    y_deviation = np.abs(y - GOAL_CENTER_Y)
    distance_to_goal_line = GOAL_CENTER_X - x

    return distance, angle, x, y_deviation, distance_to_goal_line

def plot_xg_heatmap(model_path='models/phase1_baseline.json',
                    features_path='models/phase1_features.txt'):
    """Plot xG heatmap for foot shots and headers"""

    # Load model
    model = xgb.Booster()
    model.load_model(str(model_path))

    # Load feature list
    with open(features_path) as f:
        feature_cols = [line.strip() for line in f]

    # Create grid
    X_test_grnd = pd.DataFrame(columns=feature_cols, dtype='float64')
    X_test_head = pd.DataFrame(columns=feature_cols, dtype='float64')

    i = 0
    for x_pos in range(int(PITCH_LENGTH_X//2), int(PITCH_LENGTH_X + 1)):
        for y_pos in range(0, int(PITCH_WIDTH_Y + 1)):
            distance, angle, x_coord, y_dev, dist_line = calculate_features(x_pos, y_pos)

            row_grnd = {
                'distance_to_goal': distance,
                'angle_to_goal': angle,
                'x_coordinate': x_coord,
                'y_deviation': y_dev,
                'distance_to_goal_line': dist_line,
            }

            # Set defaults for one-hot
            for col in feature_cols:
                if col.startswith('body_part_'):
                    row_grnd[col] = 1.0 if col == 'body_part_Right Foot' else 0.0
                elif col.startswith('technique_'):
                    row_grnd[col] = 1.0 if col == 'technique_Normal' else 0.0
                elif col.startswith('shot_type_'):
                    row_grnd[col] = 1.0 if col == 'shot_type_Open Play' else 0.0

            row_head = row_grnd.copy()
            for col in feature_cols:
                if col.startswith('body_part_'):
                    row_head[col] = 1.0 if col == 'body_part_Head' else 0.0

            X_test_grnd.loc[i] = row_grnd
            X_test_head.loc[i] = row_head
            i += 1

    # Predict
    dtest_grnd = xgb.DMatrix(X_test_grnd, feature_names=feature_cols)
    dtest_head = xgb.DMatrix(X_test_head, feature_names=feature_cols)

    prob_goal_grnd = model.predict(dtest_grnd).reshape(int(1 + PITCH_LENGTH_X//2), int(1 + PITCH_WIDTH_Y))
    prob_goal_head = model.predict(dtest_head).reshape(int(1 + PITCH_LENGTH_X//2), int(1 + PITCH_WIDTH_Y))

    # Plot
    mpl.rcParams['xtick.color'] = "white"
    mpl.rcParams['ytick.color'] = "white"
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10

    pitch = VerticalPitch(half=True, pitch_color='#313332', line_color='white', linewidth=1, stripe=False)
    fig, ax = pitch.grid(nrows=1, ncols=2, grid_height=0.75, space=0.1, axis=False)
    fig.set_size_inches(10, 5.5)
    fig.set_facecolor('#313332')

    pos1 = ax['pitch'][0].imshow(prob_goal_grnd, extent=(0, 80, 60, 120), aspect='equal',
                                  vmin=-0.04, vmax=0.4, cmap=plt.cm.inferno)
    pos2 = ax['pitch'][1].imshow(prob_goal_head, extent=(0, 80, 60, 120), aspect='equal',
                                  vmin=-0.04, vmax=0.4, cmap=plt.cm.inferno)

    cs1 = ax['pitch'][0].contour(prob_goal_grnd, extent=(0, 80, 60, 120),
                                  levels=[0.01, 0.05, 0.2, 0.5],
                                  colors=['darkgrey','darkgrey','darkgrey','k'], linestyles='dotted')
    cs2 = ax['pitch'][1].contour(prob_goal_head, extent=(0, 80, 60, 120),
                                  levels=[0.01, 0.05, 0.2, 0.5],
                                  colors=['darkgrey','darkgrey','darkgrey','k'], linestyles='dotted')
    ax['pitch'][0].clabel(cs1)
    ax['pitch'][1].clabel(cs2)

    fig.text(0.045, 0.9, "Expected Goals - XGBoost Phase 1", fontsize=16, color="white", fontweight="bold")
    fig.text(0.045, 0.85, "StatsBomb Open Data - 88,023 shots", fontsize=14, color="white", fontweight="regular")
    fig.text(0.12, 0.76, "Shot Type: Foot", fontsize=12, color="white", fontweight="bold")
    fig.text(0.66, 0.76, "Shot Type: Header", fontsize=12, color="white", fontweight="bold")

    cbar = fig.colorbar(pos2, ax=ax['pitch'][1], location="bottom", fraction=0.04, pad=0.0335)
    cbar.ax.set_ylabel('xG', loc="bottom", color="white", fontweight="bold", rotation=0, labelpad=20)

    fig.text(0.255, 0.09, "Created by Jaime Oriol", fontstyle="italic", ha="center", fontsize=9, color="white")

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = plot_xg_heatmap()
    plt.show()
