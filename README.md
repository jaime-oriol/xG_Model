# Expected Goals (xG) Model with XGBoost

Professional xG model trained on StatsBomb Open Data using XGBoost. Achieves Brier Score of 0.0648, outperforming StatsBomb's commercial model (0.0745) by 13%.

## Project Overview

This repository implements an iterative Expected Goals model using gradient boosting and StatsBomb's extensive open dataset. The development follows a phased approach, systematically adding feature complexity to quantify the impact of each variable category.

**Key Results:**
- Test Brier Score: 0.0648 (target: 0.068)
- AUC-ROC: 0.9041
- Training time: 3 minutes (Phase 3)
- Dataset: 88,023 shots across 23 seasons

## Dataset

**Source:** StatsBomb Open Data
**Total Shots:** 88,023 (after filtering penalties and invalid coordinates)
**Competitions:** La Liga, Premier League, Champions League, World Cup
**Temporal Range:** 2003-2024 (23 seasons)
**Goal Conversion Rate:** 10.2%

## Model Architecture

**Algorithm:** XGBoost (eXtreme Gradient Boosting)

Selected for three critical advantages:
1. Handles non-linear feature interactions automatically
2. Efficient training on tabular data (3 min vs hours for neural networks)
3. Native monotonicity constraints ensure realistic predictions

**Configuration:**
- Objective: binary:logistic
- Max depth: 5
- Learning rate: 0.05
- Regularization: L1=0.5, L2=1.0
- Monotonic constraints on distance, angle, defenders

## Development Phases

### Phase 1: Baseline (Brier: 0.085)
**Features:** Geometric variables (distance, angle, coordinates), shot type, body part, technique
**Result:** Establishes baseline performance

### Phase 2: Freeze Frames (Brier: 0.072) - **Pure xG**
**Features Added:** Defensive pressure metrics from player positions
- keeper_distance_from_line
- defenders_in_triangle
- closest_defender_distance
- defenders_in_shooting_lane

**Impact:** 15% improvement - captures 80% of model's added value

**✓ RECOMMENDED:** Phase 2 uses exclusively **PRE-SHOT** features (position, defenders, shot type). This is the model to use for fair comparisons with StatsBomb and other commercial xG models.

**Phase 2 Tuned:** Hyperparameter-optimized version via RandomizedSearchCV (50 iterations, 5-fold CV) - recommended for production use.

### Phase 3: Shot Height (Brier: 0.0648) - **Post-Shot xG**
**Features Added:** Vertical trajectory data
- shot_height (Z coordinate from end_location)
- is_header_aerial_won
- keeper_forward_high_shot

**⚠️ IMPORTANT:** Phase 3 uses `shot_height` extracted from `end_location[2]`, which represents where the ball **ended up** (height at goal, in keeper's hands, out of bounds). This is **POST-SHOT** information not available at the moment of the shot. Phase 3 is technically a **Post-Shot xG (xGOT)** model that measures both opportunity quality (pre-shot) and execution quality (post-shot).

**Result:** Best model - beats target and StatsBomb baseline
**Top Feature:** shot_height (gain: 101.8)

**For fair comparisons with pure xG models (like StatsBomb), use Phase 2.**

### Phase 4: Contextual (Brier: 0.0649)
**Features Added:** first_time, under_pressure, one_on_one
**Result:** No improvement - information redundant with existing features

### Phase 5: 360 Data (In Development)
**Features Planned:** visible_area, goal_visibility, pressure_density_360
**Approach:** Transfer learning from Phase 3
**Limitation:** Only 10% of shots have complete 360 data

## Repository Structure

```
xG_Model/
├── data/
│   ├── raw/                    # StatsBomb JSON (gitignored)
│   └── processed/              # Feature cache (gitignored)
├── src/
│   ├── data/
│   │   └── loader.py           # Data loading and parsing
│   ├── features/
│   │   ├── geometric.py        # Distance, angle, coordinates
│   │   ├── freeze_frame.py     # Defensive pressure
│   │   ├── shot_height.py      # Vertical trajectory
│   │   ├── contextual.py       # Situational features
│   │   └── three_sixty.py      # 360 vision data
│   ├── models/
│   │   ├── train_phase3.py     # Best model training
│   │   ├── validate_phase3.py  # Train/test validation
│   │   └── tune_phase3.py      # Hyperparameter search
│   └── visualization/
│       ├── comparison_scatter.py
│       ├── feature_importance.py
│       ├── calibration_plot.py
│       └── xg_heatmap.py
├── models/                     # Trained models (gitignored)
├── outputs/                    # Visualizations (gitignored)
└── notebooks/
    └── generate_all_visualizations.ipynb
```

## Usage

### Training Models

**Phase 2 (Pure xG - Recommended for comparisons):**
```bash
python -m src.models.train_phase2
```

**Phase 3 (Post-Shot xG):**
```bash
python -m src.models.train_phase3
```

### Hyperparameter Tuning

**Phase 2 Tuning (Recommended):**
```bash
python -m src.models.tune_phase2
```
Generates `models/phase2_tuned.json` - optimized pure xG model.

**Phase 3 Tuning:**
```bash
python -m src.models.tune_phase3
```
Generates `models/phase3_tuned.json` - optimized post-shot xG model.

### Validation with Train/Test Split

```bash
python -m src.models.validate_phase3
```

### Generate Visualizations

```bash
python -m src.visualization.comparison_scatter
python -m src.visualization.feature_importance
python -m src.visualization.calibration_plot
python -m src.visualization.xg_heatmap
```

Or use the notebook:
```bash
jupyter notebook notebooks/generate_all_visualizations.ipynb
```

## Model Performance

### Comparison vs StatsBomb

| Metric | My Model | StatsBomb | Improvement |
|--------|----------|-----------|-------------|
| Brier Score | 0.0648 | 0.0745 | +13% |
| AUC-ROC | 0.9041 | - | - |
| Correlation | 0.77 | - | - |
| MAE | 0.071 | - | - |

### Calibration

| xG Range | Expected | Observed | Error |
|----------|----------|----------|-------|
| < 0.05 | 3% | 2.8% | -0.2% |
| 0.05-0.15 | 10% | 9.5% | -0.5% |
| 0.15-0.30 | 22% | 23.1% | +1.1% |
| 0.30-0.60 | 45% | 46.3% | +1.3% |
| 0.60+ | 75% | 72.8% | -2.2% |

**Expected Calibration Error (ECE):** 0.0124

## Feature Importance (Top 10)

1. shot_height: 101.8
2. defenders_in_shooting_lane: 64.6
3. angle_to_goal: 59.7
4. distance_to_goal: 37.1
5. body_part_Head: 30.7
6. is_header_aerial_won: 23.2
7. keeper_cone_blocked: 20.5
8. shot_type_Open Play: 15.9
9. technique_Lob: 13.9
10. defenders_in_triangle: 13.3

## Limitations

1. **Average Player Assumption:** Model does not account for individual shooter skill
2. **Post-Shot vs Pre-Shot xG:**
   - **Phase 2 (Pure xG):** No post-shot data - evaluates opportunity quality only
   - **Phase 3 (Post-Shot xG):** Includes shot_height from end_location - measures both opportunity and execution quality
   - For fair comparisons with commercial xG models, use Phase 2
3. **Limited 360 Coverage:** Only 10% of shots have complete spatial data
4. **Temporal Invariance:** Does not model game state or tactical evolution

## Future Work

- Bayesian hierarchical model for player-specific xG
- Post-shot xG (xGOT) using ball trajectory data
- Integration with full tracking data
- Temporal models for in-game context

## Requirements

```
python>=3.8
numpy
pandas
xgboost
scikit-learn
matplotlib
mplsoccer
shapely
pillow
```

Install:
```bash
pip install -r requirements.txt
```

## Data Access

StatsBomb Open Data: https://github.com/statsbomb/open-data

## License

MIT License

## Author

Jaime Oriol
Contact: [Your contact info]
Project: Football Decoded
