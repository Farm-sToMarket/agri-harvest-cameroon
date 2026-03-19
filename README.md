# Agri-Harvest

Yield prediction platform for Cameroon agriculture. Combines soil, weather, satellite, and crop survey data into ML pipelines that predict harvest yields across 8 agroecological zones and 13 crop types.

## Models

Two pipeline versions target different dataset scales:

### v0 — scikit-learn (up to ~500K rows)

Spatial train/test split on `agroecological_zone` via `GroupShuffleSplit`, ensuring no zone leaks across sets. Preprocessing: `StandardScaler` on 40 continuous features, passthrough for 22 binary + 4 ordinal.

| Model | Type | Key hyperparameters |
|---|---|---|
| Ridge | Linear | alpha=1.0 |
| Random Forest | Bagging | 300 trees, max_depth=20, min_leaf=10 |
| Hist Gradient Boosting | Boosting | 500 iters, max_depth=8, lr=0.05 |
| Stacking | Ensemble | RF + HGB base, Ridge meta-learner |
| Baseline | Dummy | mean strategy (sanity check) |

Evaluation: RMSE, MAE, R2, MAPE — globally and per crop group / zone. Tree-based and permutation feature importances on the best model.

### v1 — LightGBM / XGBoost / PyTorch (10M+ rows)

Polars-based loading with dtype optimization (Float64 -> Float32, string -> Categorical). Stratified split on `agroecological_zone`. Missing value imputation fitted on train set.

| Model | Type | Key hyperparameters |
|---|---|---|
| LightGBM | GBDT (native API) | 128 leaves, lr=0.05, 2000 rounds, early stop 50 |
| XGBoost | GBDT (hist) | depth=12, lr=0.05, 800 estimators |
| YieldNet | PyTorch feedforward | [512, 256, 128], dropout=0.3, AdamW + cosine LR |
| HybridYieldModel | LSTM + Dense | 2-layer LSTM (hidden=128) + tabular branch |
| TransformerYieldModel | Transformer + Dense | 4-layer encoder, 8 heads, GELU, sinusoidal PE |

**Time-series models** (Hybrid, Transformer) consume per-field daily weather sequences (temperature min/max, precipitation, humidity, solar radiation) up to 180 days alongside static features. Variable-length sequences handled via masked mean pooling. Training uses cosine LR scheduling and gradient clipping.

**Hyperparameter tuning**: Optuna with pruning callbacks for all three v1 models. Search spaces defined in `config/yaml/models_v1.yaml`.

### Leakage guard

Both pipelines drop target-derived features before training:

```
nue, wue, yield_gap_ratio, harvest_index, biomass_kg_ha
```

### Feature set

66 features across 6 categories:

| Category | Examples | Count |
|---|---|---|
| Location | latitude, longitude, elevation, zone one-hots | 8 |
| Climate | temp min/max/mean, precipitation, humidity, solar radiation, GDD, VPD, aridity index | 15 |
| Soil | pH, organic carbon, nitrogen, phosphorus, sand/clay %, fertility index, CEC | 12 |
| Management | fertilizer N/P, organic fertilizer, irrigation, input intensity | 7 |
| Temporal | month sin/cos, day-of-year sin/cos, year, season ordinal | 6 |
| Engineered | diurnal range, C:N ratio, humidity-temp interaction, rain-OC interaction, disease risk score | 18 |

## Data validation

Pydantic v2 schemas enforce Cameroon-specific constraints:

- **Coordinates**: lat 1.6-13.1 N, lon 8.3-16.2 E, elevation 0-4095 m
- **Soil**: texture percentages sum to 100% (1% tolerance), pH 3.5-9.5, bulk density 0.8-2.0
- **Weather**: temperature -5 to 50 C, precipitation 0-500 mm, pressure 600-1050 hPa, auto-derived fields (day_of_year, season, rainfall_regime)
- **Crops**: 24 crop types across 7 groups, harvest index ranges per crop, yield <= biomass, intercropping LER 0.5-3.0

## Agroecological zones

Classification based on latitude, longitude, and elevation:

| Zone | Criteria | Typical crops |
|---|---|---|
| Sahel savanna | lat > 10 N | sorghum, millet, cotton |
| Sudan savanna | lat 8-10 N | sorghum, groundnut |
| Guinea savanna | lat 6-8 N | maize, yam |
| Western highlands | elev > 1200 m, lon < 11.5, lat 4.5-7.5 | potato, maize |
| Forest-savanna transition | lat 5-6 N | maize, cassava |
| Humid forest (coast) | lat < 5, lon < 10, elev < 500 m | cocoa, plantain |
| Humid forest (inland) | lat < 5, interior | cocoa, cassava |
| Mont Cameroun volcanic | lat 4.0-4.35, lon 9.0-9.35, elev > 2500 m | specialty crops |

Season classification: bimodal (south, < 6 N) with 5 seasons, monomodal (north, >= 6 N) with 4 seasons.

## Configuration

All hardcoded values are externalized to YAML (`config/yaml/`):

| File | Contents |
|---|---|
| `geography.yaml` | Cameroon bounds, elevation range, zone thresholds, season definitions, validation ranges |
| `agriculture.yaml` | 13 crop types, soil texture classes, 5 IRAD research centers with coordinates |
| `models_v0.yaml` | v0 feature lists, estimator hyperparameters, train/test split config |
| `models_v1.yaml` | v1 LightGBM/XGBoost/YieldNet params, Optuna search spaces, time-series config |

Runtime settings (API host/port, log level, secret key) are configured via `.env` and `pydantic-settings`.

## Installation

```bash
git clone <repository-url>
cd agri-harvest
python -m venv .venv && source .venv/bin/activate

pip install -e "."           # core (FastAPI, Pydantic, PyYAML, NumPy, Pandas)
pip install -e ".[ml]"       # + scikit-learn, LightGBM, XGBoost, PyTorch, Optuna, SHAP
pip install -e ".[geo]"      # + GeoPandas, Rasterio, GeoPy
pip install -e ".[dev]"      # + pytest, black, mypy

cp .env.example .env         # edit SECRET_KEY at minimum
```

Requires Python 3.12+.

## Usage

### v0 pipeline

```python
from models.trainer import YieldModelTrainer

trainer = YieldModelTrainer("data/features.csv")
comparison = trainer.run()                     # all 5 models
comparison = trainer.run(["random_forest"])     # single model
```

### v1 pipeline (10M+ rows)

```python
from models.v1.trainer import YieldModelTrainer

trainer = YieldModelTrainer("data/features.parquet")
comparison = trainer.run()                                # LightGBM + XGBoost + YieldNet
comparison = trainer.run(["lightgbm"], optimize=True)     # with Optuna tuning
```

### Inference

```python
from models.predict import YieldPredictor

predictor = YieldPredictor("data/models/best_model.joblib")
yield_kg_ha = predictor.predict_single({
    # Location
    "latitude": 3.87, "longitude": 11.52, "elevation": 650,
    # Climate
    "temperature_min": 20.0, "temperature_max": 31.0, "temperature_mean": 25.5,
    "precipitation_daily": 15.2, "relative_humidity": 78.5, "solar_radiation": 18.5,
    # Soil
    "ph_water": 6.2, "organic_carbon": 2.1, "total_nitrogen": 0.18,
    "available_phosphorus": 18.5, "sand_percentage": 45.0, "clay_percentage": 22.0,
    # Management
    "fertilizer_nitrogen": 120.0, "fertilizer_phosphorus": 40.0,
    "organic_fertilizer": 5.0, "disease_incidence": 12.5,
    # Temporal (cyclical encoding)
    "month_sin": 0.866, "month_cos": 0.5,       # March
    "doy_sin": 0.97, "doy_cos": -0.26,          # ~day 75
    "month": 3, "day_of_year": 75, "year": 2024, "season_ordinal": 1,
    # Derived climate
    "gdd_base10": 15.5, "gdd_base15": 10.5, "diurnal_range": 11.0,
    "aridity_index": 0.85, "vpd_kpa": 1.2, "temp_altitude_residual": -2.3,
    # Derived soil
    "soil_fertility_index": 0.82, "cn_ratio": 11.7, "cec_estimate": 12.5,
    # Derived management
    "input_intensity": 0.65, "total_mineral_fert": 160.0,
    "organic_mineral_ratio": 0.03,
    # Interactions
    "humidity_temp_interaction": 2003.0, "rain_oc_interaction": 31.9,
    "input_soil_interaction": 0.53, "water_supply_index": 0.72,
    "disease_risk_score": 0.15,
    # Binary flags
    "irrigation_applied": 0, "heavy_rain_day": 0, "is_rainy_season": 1,
    "data_quality_flag": 0, "rainfall_regime": 1,
    # Zone one-hots
    "zone_coastal_forest": 0, "zone_forest_savanna": 0,
    "zone_guinea_savanna": 0, "zone_highlands": 0,
    "zone_mont_cameroun_volcanic": 0, "zone_sahel_savanna": 0,
    # Crop group one-hots
    "grp_cereals": 1, "grp_industrial_crops": 0, "grp_legumes": 0,
    "grp_root_tubers": 0, "grp_tree_crops": 0, "grp_vegetables": 0,
    # Altitude class one-hots
    "alt_highland": 0, "alt_lowland": 0, "alt_mid_altitude": 1,
    "alt_mountain": 0, "alt_plateau": 0,
})
```

### Tests

```bash
pytest tests/ -v    # 141 tests
```

## License

Copyright (c) 2025 SYNTHI-AI. Released under the [MIT License](LICENSE).

## Contact

**SYNTHI-AI** — [https://synthi-ai.com](https://synthi-ai.com)

For questions, bug reports, or feature requests, open an issue on this repository.
