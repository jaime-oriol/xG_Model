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

---

## Project Objective

**Modelado de Goles Esperados (xG) TOP usando StatsBomb Open Data**

Desarrollar UN modelo profesional de xG que iguale o supere el Brier Score de StatsBomb (~0.068) aprovechando características únicas de sus datos: freeze frames, shot impact height, y contexto de jugada.

### Target Metrics

- **Brier Score**: ≤ 0.068 (igualar/superar StatsBomb)
- **AUC-ROC**: ≥ 0.82
- **Calibration Error**: < 0.03

---

## Decisión Técnica: UN SOLO MODELO (XGBoost)

### Por Qué XGBoost (NO Red Neuronal)

**Red Neuronal NO es adecuada porque**:
1. Datos insuficientes (~25k-200k shots vs >500k necesarios para NN)
2. Datos tabulares (XGBoost domina vs NN en tabular data)
3. Sin monotonicity constraints nativos
4. Caja negra (baja interpretabilidad)
5. Lento (horas vs minutos)
6. Mayor riesgo de overfitting

**XGBoost ES óptimo porque**:
1. Perfecto para datos tabulares (73% de Kaggle wins)
2. Suficiente con 10k-100k samples
3. Monotonicity constraints nativos (crítico para xG realista)
4. SHAP values (interpretabilidad total)
5. Rápido (iteración rápida)
6. Probado (estándar industria para xG)

**NO ensemble**: Un XGBoost bien tunneado es suficiente. Ensemble añade 0.001-0.003 Brier vs 0.010-0.015 que dan buenos features.

**Prioridad**: 80% esfuerzo en feature engineering, 20% en tuning

### Configuración XGBoost

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'monotone_constraints': {
        'distance_to_goal': -1,      # Más lejos → menor xG
        'angle_to_goal': 1,           # Mayor ángulo → mayor xG
        'defenders_in_triangle': -1,  # Más defensores → menor xG
    },
    'early_stopping_rounds': 20,
    'random_state': 42,
}
```

---

## Datos Disponibles: StatsBomb Open Data

### Inventario Completo (Verificado 100%)

**Total descargado**: 985 partidos de 23 temporadas

**Fase 1 - Modernas sin 360** (8 temporadas):
- La Liga 2018/19, 2019/20, 2020/21
- Premier League 2015/16
- Champions League 2015/16, 2016/17, 2017/18, 2018/19

**Fase 2 - Históricas** (5 temporadas):
- Premier League 2003/04
- La Liga 2004/05, 2005/06
- Champions League 2008/09, 2009/10

**Fase 3 - Con 360** (11 temporadas):
- Bundesliga 2023/24 (34 partidos)
- FIFA World Cup 2022 (64 partidos)
- La Liga 2020/21 (35 partidos)
- Ligue 1 2022/23, 2021/22
- MLS 2023
- UEFA Euro 2024, 2020
- Women's Euro 2025, 2022
- Women's World Cup 2023

**Estimación total**: ~25,000 shots

### Estructura de Datos

```
data/raw/
├── competitions/
│   ├── competitions.json      # 75 competiciones/temporadas
│   └── selection.json          # Competiciones seleccionadas
├── matches/
│   └── {comp_id}/{season_id}.json   # 985 partidos
├── events/
│   └── {match_id}.json         # ~3,000-4,500 eventos/partido
└── three-sixty/
    └── {match_id}.json         # Datos 360 (partidos seleccionados)
```

### Campos Disponibles en Shot Events

**SIEMPRE presentes**:
- `location`: [x, y] coordenadas del disparo (campo 120x80 yardas)
- `shot.end_location`: [x, y, z] - z es SHOT IMPACT HEIGHT
- `shot.body_part`: Right Foot, Left Foot, Head, Other
- `shot.technique`: Normal, Volley, Half Volley, Lob, Overhead Kick, etc.
- `shot.type`: Open Play, Free Kick, Corner, Penalty
- `shot.outcome`: Goal, Saved, Off T, Post, Blocked, Wayward
- `shot.freeze_frame`: Array de posiciones de jugadores (12-15 jugadores)
- `shot.statsbomb_xg`: xG de StatsBomb (para comparación)

**Condicionales** (presentes cuando aplica):
- `under_pressure`: Boolean - jugador bajo presión (3-32% de shots)
- `shot.first_time`: Boolean - disparo de primera (12-44% de shots)
- `shot.one_on_one`: Boolean - mano a mano (3-8% de shots)
- `shot.deflected`: Boolean - balón desviado (6-8% de shots)
- `shot.aerial_won`: Boolean - ganó duelo aéreo (3-8% de shots)
- `shot.open_goal`: Boolean - portería vacía (<1% de shots)
- `shot.key_pass_id`: UUID del pase previo

### Cobertura Verificada

| Feature | Cobertura |
|---------|-----------|
| Freeze Frames | 93-100% de shots |
| Shot Height (end_location[2]) | 63-83% de shots |
| Datos 360 | 92-95% de shots (solo comps con 360) |

### Coordenadas

```
Campo: 120 yardas (largo) x 80 yardas (ancho)

Portería:
- Centro: (120, 40)
- Poste izquierdo: (120, 36.16)
- Poste derecho: (120, 43.84)
- Ancho: 7.32 yardas
- Alto: 2.44 yardas
```

---

## Estrategia de Implementación: Desarrollo Iterativo

### Fase 1: Baseline (Brier ~0.085)

**Datos**: La Liga 2018-2020, Premier 2015/16

**Features**:
- `distance_to_goal`: Distancia euclidiana a (120, 40)
- `angle_to_goal`: Ángulo entre shot y postes
- `x_coordinate`: Coordenada X directa
- `y_deviation`: abs(shot_y - 40)
- `body_part`: One-Hot encoding
- `technique`: One-Hot encoding
- `shot_type`: Open Play, Free Kick, etc.

**Modelo**: XGBoost con defaults razonables

**Validación**: Comparar con statsbomb_xg

### Fase 2: + Freeze Frames (Brier ~0.072)

**Features añadidas**:
- `keeper_distance_from_line`: keeper_x - 120
- `keeper_lateral_deviation`: Desviación lateral del portero
- `keeper_cone_blocked`: Ángulo obstruido por portero
- `defenders_in_triangle`: Conteo en triángulo shot-postes
- `closest_defender_distance`: Distancia al defensor más cercano
- `defenders_within_5m`: Conteo de defensores < 5m
- `defenders_in_shooting_lane`: Defensores entre shot y portería

**CRÍTICO**: Aquí está el 80% del valor añadido vs modelos básicos

### Fase 3: + Shot Height (Brier ~0.070)

**Features añadidas**:
- `shot_height`: end_location[2] (imputar 0.0 si falta)
- `shot_height_category`: ground, low, medium, high
- `is_header_aerial_won`: (body_part == Head) & (aerial_won == True)
- `keeper_forward_high_shot`: (keeper_distance > 2) & (height > 1.8)

### Fase 4: + Contexto (Brier ~0.068)

**Features añadidas**:
- `first_time`, `under_pressure`, `one_on_one`, `deflected` (boolean flags)
- `preceding_action_type`: Buscar evento con key_pass_id
  - Through Ball, Cross, Cut Back, etc.
- `attack_pattern`: play_pattern (Counter, Set Piece, etc.)

### Fase 5: Fine-tuning con 360 (Brier ~0.065)

**Datos**: Bundesliga 2023/24, World Cup 2022

**Features añadidas**:
- `visible_area_size`: Área del polígono visible
- `goal_visibility_score`: % de portería visible
- `pressure_density_360`: Densidad de rivales en área visible

**Approach**: Transfer learning
- Partir de modelo Fase 4 pre-entrenado
- Fine-tune con learning_rate bajo
- Solo ~130 partidos con 360, necesita transfer learning

---

## Validación y Comparación

### Estrategia

**Entrenar con todos los datos** de cada fase, validar comparando predicciones vs `statsbomb_xg`:

1. Calcular TU xG para cada shot
2. Calcular CORRELACIÓN con statsbomb_xg
3. Calcular DIFERENCIAS absolutas promedio
4. Identificar en qué tipo de shots eres mejor/peor
5. Calcular Brier Score de ambos modelos

### Métricas

**Primarias**:
- Brier Score: Error cuadrático medio (target ≤ 0.068)
- Correlation con StatsBomb xG (target > 0.90)

**Secundarias**:
- AUC-ROC (target ≥ 0.82)
- Calibration plot (expected vs observed goal rate)
- Calibration Error ECE (target < 0.03)

### Análisis por Categorías

Comparar vs StatsBomb en:
- Headers vs foot shots
- Inside box vs outside box
- First time vs controlled
- One-on-one vs contested
- Open play vs set pieces

---

## Feature Engineering: Cálculos Clave

### Geometría Básica

```python
# Distancia a portería
distance = np.sqrt((120 - shot_x)**2 + (40 - shot_y)**2)

# Ángulo a portería
left_post = np.array([120, 36.16])
right_post = np.array([120, 43.84])
shot_pos = np.array([shot_x, shot_y])

angle_left = np.arctan2(left_post[1] - shot_y, left_post[0] - shot_x)
angle_right = np.arctan2(right_post[1] - shot_y, right_post[0] - shot_x)
angle = abs(angle_left - angle_right) * 180 / np.pi
```

### Freeze Frame Features

```python
# Extraer portero
keeper = [p for p in freeze_frame if p['position']['name'] == 'Goalkeeper'
          and not p['teammate']][0]

# Distancia portero a línea
keeper_distance_from_line = keeper['location'][0] - 120

# Defensores en triángulo shot-postes
from shapely.geometry import Polygon, Point

triangle = Polygon([
    shot_pos,
    [120, 36.16],
    [120, 43.84]
])

defenders = [p for p in freeze_frame
             if not p['teammate']
             and p['position']['name'] != 'Goalkeeper']

defenders_in_triangle = sum(
    triangle.contains(Point(p['location']))
    for p in defenders
)

# Distancia al defensor más cercano
distances = [
    np.sqrt((shot_x - d['location'][0])**2 + (shot_y - d['location'][1])**2)
    for d in defenders
]
closest_defender_distance = min(distances) if distances else 999
```

---

## Development Workflow

### 1. Data Preparation

```bash
# Datos ya descargados en data/raw/
# 985 partidos, ~25,000 shots estimados
```

### 2. Feature Engineering

```python
# src/features/geometric.py
def calculate_basic_features(shot_events: pd.DataFrame) -> pd.DataFrame:
    """Distancia, ángulo, coordenadas."""

# src/features/freeze_frame.py
def calculate_freeze_frame_features(shot_events: pd.DataFrame) -> pd.DataFrame:
    """Keeper, defensores, presión."""
```

### 3. Model Training

```python
# src/models/train.py
def train_xgboost_model(X: pd.DataFrame, y: pd.Series) -> xgb.Booster:
    """Entrenar XGBoost con early stopping."""
```

### 4. Validation

```python
# src/models/evaluate.py
def compare_with_statsbomb(y_pred: np.ndarray,
                          statsbomb_xg: np.ndarray,
                          y_true: np.ndarray) -> Dict:
    """Comparar modelo vs StatsBomb."""
```

### 5. Iteration

- SHAP analysis → ¿Qué features importan?
- Error analysis → ¿Dónde falla el modelo?
- Añadir features → Siguiente fase
- Re-entrenar → Validar mejora

---

## Repository Structure

```
xG_Model/
├── data/
│   ├── raw/                    # StatsBomb JSON (gitignored)
│   │   ├── competitions/
│   │   ├── matches/
│   │   ├── events/
│   │   └── three-sixty/
│   └── processed/              # Features engineered (gitignored)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Cargar y parsear JSONs
│   ├── features/
│   │   ├── __init__.py
│   │   ├── geometric.py        # Features geométricas
│   │   ├── freeze_frame.py     # Features de freeze frame
│   │   └── contextual.py       # Features de contexto
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Entrenamiento XGBoost
│   │   └── evaluate.py         # Validación y comparación
│   └── visualization/
│       ├── __init__.py
│       └── plots.py            # Calibration, SHAP, etc.
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_phase1_baseline.ipynb
│   ├── 03_phase2_freeze_frames.ipynb
│   └── 04_phase3_shot_height.ipynb
│
├── models/                     # Modelos guardados (.pkl, .json)
├── outputs/                    # Plots, reports
├── tests/                      # Unit tests
│
├── .gitignore                  # data/, CLAUDE.md
├── requirements.txt
├── CLAUDE.md                   # Este archivo
└── README.md                   # Project overview
```

---

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

# Docstrings mandatory
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
# Validate inputs
assert 0 <= shot_x <= 120, f"Invalid x: {shot_x}"
assert 0 <= shot_y <= 80, f"Invalid y: {shot_y}"

# Specific exceptions
try:
    features = engineer_features(shot_data)
except KeyError as e:
    logger.error(f"Missing field: {e}")
    raise
```

### Reproducibility

```python
# Random seeds everywhere
np.random.seed(42)
random.seed(42)

params = {
    'random_state': 42,
    # ...
}
```

---

## Git Workflow

### Commits

```bash
# Conventional commits
feat(features): add keeper cone calculation
feat(model): implement XGBoost with monotonicity constraints
fix(data): correct coordinate normalization
docs: update CLAUDE.md with implementation strategy
```

### Branches

```
main
├── feature/geometric-features
├── feature/freeze-frame-features
├── feature/shot-height
└── feature/360-data
```

---

## Best Practices

### Monotonicity Constraints

**NO negociable**. Sin constraints el modelo puede aprender patrones absurdos:
- "A más distancia más goles" (correlación espuria)
- "One-on-one baja xG" (dataset pequeño de 1v1)

Constraints aseguran realismo físico.

### Feature Importance

Después de cada fase, analizar:
- SHAP values → ¿Qué features son top?
- Feature importance nativo de XGBoost
- Si un feature no aporta, eliminarlo

### Calibration

**Tan importante como Brier Score**. Verificar calibration plot después de cada experimento.

Modelo no calibrado:
- Predice 0.3 xG pero tasa real es 0.5
- Inútil para aplicaciones reales

---

## References

### StatsBomb
- [Open Data Repository](https://github.com/statsbomb/open-data)
- [Data Specification](https://github.com/statsbomb/open-data/tree/master/doc)

### Papers
- [xGBoost: Scalable Tree Boosting](https://arxiv.org/abs/1603.02754)
- [SHAP: Unified Interpretability](https://arxiv.org/abs/1705.07874)

### Tools
- [mplsoccer](https://mplsoccer.readthedocs.io/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [SHAP](https://shap.readthedocs.io/)

---

**Este archivo es la única fuente de verdad para el desarrollo del modelo.**
