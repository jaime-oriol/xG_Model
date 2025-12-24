# Expected Goals (xG) Model with XGBoost

Professional xG model trained on StatsBomb Open Data using XGBoost. Phase 2 Tuned achieves Brier Score of 0.0755, comparable to StatsBomb's commercial model (0.0745).

## Project Overview

This repository implements an iterative Expected Goals model using gradient boosting and StatsBomb's extensive open dataset. The development follows a phased approach, systematically adding feature complexity to quantify the impact of each variable category.

**Key Results (Phase 2 Tuned - Pure xG):**
- Test Brier Score: 0.0755 (StatsBomb baseline: 0.0745)
- AUC-ROC: 0.8132
- Correlation vs StatsBomb: 0.77
- Dataset: 88,023 shots across 23 seasons
- Training time: 2 minutes with hyperparameter optimization

## Dataset

**Source:** StatsBomb Open Data
**Total Shots:** 88,023 (after filtering penalties and invalid coordinates)
**Competitions:** La Liga, Premier League, Champions League, World Cup, Bundesliga, MLS, UEFA Euro...
**Temporal Range:** 2003-2024 (23 seasons)
**Goal Conversion Rate:** 10.2%

## Model Architecture

**Algorithm:** XGBoost (eXtreme Gradient Boosting)

Selected for three critical advantages:
1. Handles non-linear feature interactions automatically
2. Efficient training on tabular data (minutes vs hours for neural networks)
3. Native monotonicity constraints ensure realistic predictions

**Phase 2 Tuned Configuration:**
- Objective: binary:logistic
- Max depth: 4
- Learning rate: 0.05
- N estimators: 400
- Subsample: 0.8, colsample_bytree: 0.8
- Regularization: L1=0.3, L2=3.0
- Min child weight: 5
- Monotonic constraints on distance, angle, defenders

## Development Phases

### Phase 1: Baseline (Brier: 0.085)
**Features:** Geometric variables (distance, angle, coordinates), shot type, body part, technique
**Result:** Establishes baseline performance using only basic geometry

### Phase 2: Freeze Frames (Brier: 0.072)
**Features Added:** Defensive pressure metrics from player positions
- keeper_distance_from_line
- keeper_lateral_deviation
- keeper_cone_blocked
- defenders_in_triangle
- closest_defender_distance
- defenders_within_5m
- defenders_in_shooting_lane

**Impact:** 15% improvement - captures 80% of model's added value

### Phase 2 Tuned: Hyperparameter Optimization (Brier: 0.0755) - Pure xG
**Optimization:** RandomizedSearchCV with 50 iterations and 5-fold cross-validation
**Result:** Achieves comparable performance to StatsBomb using exclusively PRE-SHOT features
**RECOMMENDED:** Phase 2 Tuned is the reference model for fair comparisons with commercial xG models. Uses only information available at the moment of the shot (position, defenders, shot type). No post-shot data.

### Phase 3: Shot Height (Brier: 0.0648) - Post-Shot xG
**Features Added:** Vertical trajectory data
- shot_height (Z coordinate from end_location)
- is_header_aerial_won
- keeper_forward_high_shot

**WARNING:** Phase 3 uses shot_height extracted from end_location[2], which represents where the ball ended up (height at goal, in keeper's hands, out of bounds). This is POST-SHOT information not available at the moment of the shot. Phase 3 is technically a Post-Shot xG (xGOT) model that measures both opportunity quality (pre-shot) and execution quality (post-shot).

**Result:** Best Brier Score (0.0648) but not comparable to pure xG models
**Top Feature:** shot_height (gain: 101.8)

**For fair comparisons with commercial xG models, use Phase 2 Tuned.**

### Phase 4: Contextual (Brier: 0.0649)
**Features Added:** first_time, under_pressure, one_on_one
**Result:** No improvement - information redundant with existing freeze frame features

### Phase 5: 360 Data (Transfer Learning)
**Features Added:** visible_area_size, goal_visibility_score, pressure_density_360
**Approach:** Fine-tuning from Phase 3 with low learning rate (0.01)
**Data Coverage:** 10% of shots have complete 360 data
**Method:** Transfer learning leverages Phase 3 pre-trained model, then fine-tunes on 360 subset
**Requirement:** Shapely library for polygon calculations

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
│   │   ├── train_phase2.py     # Pure xG training
│   │   ├── tune_phase2.py      # Phase 2 hyperparameter optimization
│   │   ├── train_phase3.py     # Post-shot xG training
│   │   ├── train_phase5.py     # 360 transfer learning
│   │   └── validate_phase3.py  # Train/test validation
│   └── visualization/
│       ├── comparison_scatter.py  # Model vs StatsBomb scatter
│       ├── feature_importance.py  # SHAP values
│       ├── calibration_plot.py    # Calibration curve
│       ├── xg_heatmap.py          # Spatial xG heatmap
│       ├── timeline_phases.py     # Phase evolution
│       └── shot_xg.py             # Shot map visualization
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

**Phase 2 Tuned (Optimized Pure xG - Production):**
```bash
python -m src.models.tune_phase2
```
Generates `models/phase2_tuned.json` - optimized pure xG model.

**Phase 3 (Post-Shot xG):**
```bash
python -m src.models.train_phase3
```

**Phase 5 (360 Transfer Learning):**
```bash
python -m src.models.train_phase5
```
Requires: Shapely library, Phase 3 pre-trained model

### Hyperparameter Tuning

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
python -m src.visualization.timeline_phases
```

Or use the notebook:
```bash
jupyter notebook notebooks/generate_all_visualizations.ipynb
```

## Model Performance

### Phase 2 Tuned vs StatsBomb (Fair Comparison - Pure xG)

| Metric | Phase 2 Tuned | StatsBomb | Difference |
|--------|---------------|-----------|------------|
| Brier Score | 0.0755 | 0.0745 | +0.001 (comparable) |
| AUC-ROC | 0.8132 | - | - |
| Correlation | 0.77 | - | High consistency |
| MAE | 0.071 | - | 7.1 pp average difference |

**Conclusion:** Phase 2 Tuned achieves comparable performance to StatsBomb's commercial model using exclusively open data and pre-shot features.

### Phase 3 vs StatsBomb (Post-Shot xG - Not Directly Comparable)

| Metric | Phase 3 | StatsBomb | Improvement |
|--------|---------|-----------|-------------|
| Brier Score | 0.0648 | 0.0745 | +13% |
| AUC-ROC | 0.9041 | - | - |

**Note:** Phase 3 includes post-shot information (shot_height). Not a fair comparison with pure xG models.

### Calibration (Phase 2 Tuned)

Model demonstrates strong calibration with low Expected Calibration Error (ECE).

**Calibration Analysis:**
- Low Brier Score (0.0755) indicates both good discrimination and calibration
- Regularization (L1=0.3, L2=3.0) prevents overconfidence
- Monotonicity constraints ensure physically realistic probabilities

## Feature Importance (Phase 2 Tuned - Top 10)

1. defenders_in_shooting_lane: 64.6
2. angle_to_goal: 59.7
3. distance_to_goal: 37.1
4. body_part_Head: 30.7
5. keeper_cone_blocked: 20.5
6. defenders_in_triangle: 13.3
7. keeper_distance_from_line: 12.8
8. shot_type_Open Play: 11.5
9. technique_Lob: 9.7
10. closest_defender_distance: 8.9

**Key Insight:** Defensive pressure features (freeze frames) account for 80% of model's predictive power beyond basic geometry.

## Limitations

1. **Average Player Assumption:** Model does not account for individual shooter skill
2. **Phase Comparison:**
   - **Phase 2 Tuned (Pure xG):** Evaluates opportunity quality only - fair comparison with commercial models
   - **Phase 3 (Post-Shot xG):** Includes execution quality - technically xGOT, not pure xG
3. **Limited 360 Coverage:** Only 10% of shots have complete spatial data
4. **Temporal Invariance:** Does not model game state or tactical evolution

## Future Work

- Bayesian hierarchical model for player-specific xG
- Full xGOT implementation with ball trajectory tracking data
- Integration with full tracking data
- Temporal models for in-game context
- Phase 5 validation and feature importance analysis

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
conda install -c conda-forge numpy pandas scikit-learn matplotlib xgboost mplsoccer shapely pillow
```

Or with pip:
```bash
pip install numpy pandas scikit-learn matplotlib xgboost mplsoccer shapely pillow
```

## Data Access

StatsBomb Open Data: https://github.com/statsbomb/open-data

## License

MIT License

## Author

Jaime Oriol
Project: Football Decoded
