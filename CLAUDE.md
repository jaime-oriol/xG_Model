# CLAUDE.md - xG Model Development Guide

## Core Philosophy

Ve paso a paso, uno a uno. Despacio es el camino más rápido. Escribe siempre el código lo más compacto y conciso posible, y que cumpla exactamente lo pedido al 100%. Sin emojis ni florituras. Usa nombres claros y estándar. Incluye solo comentarios útiles y necesarios.

Antes de realizar cualquier tarea, revisa cuidadosamente el archivo CLAUDE.md.

### Development Principles

- **KISS**: Choose straightforward solutions over complex ones
- **YAGNI**: Implement features only when needed
- **Fail Fast**: Check for errors early and raise exceptions immediately
- **Single Responsibility**: Each function, class, and module has one clear purpose
- **Dependency Inversion**: High-level modules depend on abstractions, not implementations

## Project Objective

**Modelado de Goles Esperados (xG) de Próxima Generación usando StatsBomb Open Data**

Este proyecto desarrolla un modelo profesional de xG aprovechando las características únicas de StatsBomb Open Data:

### Ventajas Diferenciales de StatsBomb

1. **Freeze Frames**: Coordenadas (x, y) de todos los jugadores relevantes en el instante del disparo
2. **Shot Impact Height**: Altura exacta del balón cuando es golpeado
3. **360 Data**: Datos de visión periférica para modelar líneas de pase y presión defensiva dinámica
4. **Granularidad**: ~3,400 eventos por partido

### Marco Técnico

**Algoritmos Principales**:
- **XGBoost**: Regularización L1/L2 superior, ideal para variables espaciales colineales
- **LightGBM**: Entrenamiento leaf-wise rápido para iteraciones masivas
- **Random Forest**: Baseline y benchmarks de estabilidad

**Función Objetivo**: Log-Loss (Entropía Cruzada Binaria)

**Métricas de Evaluación**:
- **Brier Score**: Target ~0.068 (error cuadrático medio entre probabilidad predicha y resultado real)
- **AUC-ROC**: Target 0.80-0.88 (con Freeze Frames, superior al ~0.78 de modelos basados solo en eventos)

### Ingeniería de Características

**Variables de Visibilidad y Presión**:
- **Keeper Cone**: Ángulo de visión de la portería disponible, restando obstrucción del portero y defensores
- **Defenders in Triangle**: Conteo de defensores en el triángulo balón-postes
- **Distance to Closest Defender**: Medida de presión inmediata

**Variables Biomecánicas y de Contexto**:
- **Shot Impact Height**: Distinguir remates aéreos vs disparos a ras de suelo
- **Preceding Event Sequence**: Incremento de xG si el disparo viene precedido de through ball o cut back

**Variables Base**:
- Distancia a portería
- Ángulo de disparo
- Parte del cuerpo (pie, cabeza, otros)
- Técnica de disparo
- Situación de juego (open play, corner, free kick, etc.)

### Pipeline de Desarrollo

1. **Ingesta**: `statsbombpy` para cargar JSON en Pandas DataFrames
2. **Normalización**: Campo de 120x80 yardas, centrar portería en punto fijo
3. **Feature Engineering**: Calcular variables espaciales mediante trigonometría
4. **Encoding**: One-Hot o Target Encoding para variables categóricas
5. **Modelado**: XGBoost/LightGBM con Log-Loss
6. **Tuning**: Optimización bayesiana con Optuna
7. **Validación**: Brier Score + AUC-ROC
8. **Calibración**: Ajuste de probabilidades predichas
9. **Interpretabilidad**: SHAP values para explicar predicciones individuales

### Estado del Arte (2025)

**xGOT (Expected Goals on Target)**: xG post-shot considerando ubicación en portería y velocidad del balón

**DxT (Dynamic Expected Threat)**: Probabilidad de que cada acción termine en gol en los próximos 10-15 segundos

### Herramientas Técnicas

- **Python**: 3.10+
- **Data**: Pandas, NumPy
- **Modeling**: Scikit-learn, XGBoost, LightGBM, Optuna
- **Visualization**: mplsoccer, Matplotlib
- **Interpretability**: SHAP

## Repository Structure

```
xG_Model/
├── data/                   # StatsBomb JSON data
│   ├── raw/               # Original data
│   └── processed/         # Cleaned and engineered features
│
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                   # Source code
│   ├── data/             # Data loading and processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── visualization/    # Plotting utilities
│
├── models/               # Saved models
├── outputs/              # Plots, reports, predictions
├── tests/               # Unit tests
├── environment.yml      # Conda environment
├── requirements.txt     # Pip dependencies
└── CLAUDE.md           # This file
```

## Development Standards

### Code Style

```python
# Naming
shot_distance = 15.3           # snake_case
class XGModel:                 # PascalCase
MAX_DISTANCE = 40.0            # UPPER_CASE
_calculate_angle()             # Leading _ for private

# Type hints required
def calculate_keeper_cone(shot_x: float, shot_y: float,
                         keeper_x: float, keeper_y: float) -> float:
    """Calculate keeper cone angle."""

# Docstrings mandatory for public functions
def extract_freeze_frame_features(freeze_frame: List[Dict]) -> pd.DataFrame:
    """
    Extract defensive pressure features from freeze frame data.

    Args:
        freeze_frame: List of player positions at shot moment

    Returns:
        DataFrame with engineered features
    """
```

### Error Handling

```python
# Specific exceptions
try:
    features = engineer_features(shot_data)
except KeyError as e:
    logger.error(f"Missing required field: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid data format: {e}")
    return None
```

### Data Validation

```python
# Validate coordinates
assert 0 <= shot_x <= 120, f"Invalid x coordinate: {shot_x}"
assert 0 <= shot_y <= 80, f"Invalid y coordinate: {shot_y}"

# Validate probabilities
assert 0 <= xg_value <= 1, f"Invalid xG: {xg_value}"
```

## Git Workflow

### Branch Strategy

```
main (protected)
  ├── feature/freeze-frame-features
  ├── feature/xgboost-model
  ├── fix/coordinate-normalization
  └── docs/update-readme
```

### Commit Format

```bash
# Conventional commits: <type>(<scope>): <subject>

feat: New feature
fix: Bug fix
docs: Documentation
refactor: Code restructuring
perf: Performance improvement
test: Tests
chore: Maintenance

# Ejemplos
git commit -m "feat(features): add keeper cone calculation"
git commit -m "feat(model): implement XGBoost with Optuna tuning"
git commit -m "fix(data): correct coordinate normalization"
git commit -m "docs: add feature engineering explanation"
```

## Model Development Checklist

### Data Preparation
- [ ] Load StatsBomb data with statsbombpy
- [ ] Filter shot events
- [ ] Normalize coordinates (120x80 field)
- [ ] Handle missing values
- [ ] Validate data quality

### Feature Engineering
- [ ] Basic features (distance, angle)
- [ ] Keeper Cone calculation
- [ ] Defenders in Triangle count
- [ ] Distance to Closest Defender
- [ ] Shot Impact Height processing
- [ ] Preceding Event Sequence encoding
- [ ] Categorical encoding (body part, technique)

### Modeling
- [ ] Train/test split (temporal or random)
- [ ] Baseline model (Random Forest)
- [ ] XGBoost implementation
- [ ] LightGBM implementation
- [ ] Hyperparameter tuning (Optuna)
- [ ] Cross-validation

### Evaluation
- [ ] Brier Score calculation
- [ ] AUC-ROC curve
- [ ] Calibration plot
- [ ] Feature importance analysis
- [ ] SHAP values for interpretability
- [ ] Test on holdout competitions

### Visualization
- [ ] Shot maps with xG values
- [ ] Feature importance plots
- [ ] Calibration curves
- [ ] SHAP summary plots
- [ ] Performance comparison charts

## Best Practices

### Monotonicity Constraints
Implementar restricciones de monotonicidad en el modelo:
- A mayor distancia → menor xG
- Mayor ángulo (más perpendicular) → mayor xG
- Más defensores en triángulo → menor xG

### Reproducibility
- Set random seeds (42 by convention)
- Document all preprocessing steps
- Version control for data and models
- Save model artifacts with metadata

### Performance
- Use vectorized operations (NumPy/Pandas)
- Avoid loops when possible
- Cache intermediate results
- Profile code for bottlenecks

## References

### StatsBomb Resources
- [StatsBomb Open Data Repository](https://github.com/statsbomb/open-data)
- [StatsBombPy Documentation](https://github.com/statsbomb/statsbombpy)
- [StatsBomb Data Specification](https://github.com/statsbomb/open-data/tree/master/doc)

### Technical Papers
- [Expected Goals: Explaining Match Results Using Predictive Analytics](https://www.google.com/search?q=expected+goals+predictive+analytics)
- [xGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

### Visualization
- [mplsoccer Documentation](https://mplsoccer.readthedocs.io/)
- [Friends of Tracking YouTube Channel](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w)

---

**Remember**: Este archivo es la única fuente de verdad para el desarrollo del modelo. Mantenlo actualizado. Referencia para prácticas de desarrollo consistentes.
