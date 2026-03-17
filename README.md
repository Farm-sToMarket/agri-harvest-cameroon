# Agri-Harvest: Cameroon Agricultural Data Management System

A comprehensive platform for processing and analyzing agricultural data from Cameroon, including satellite imagery, soil data, weather information, and machine learning models for agricultural insights.

## Features

- **Multi-source Data Integration**: Combine satellite, soil, weather, and crop data
- **MongoDB Database**: Scalable NoSQL database with geospatial indexing
- **Machine Learning Pipeline**: LightGBM, XGBoost, PyTorch YieldNet, LSTM+Tabular hybrid, and Transformer models for yield prediction at scale (10M+ rows)
- **Hyperparameter Tuning**: Optuna-based tuning with pruning for all model types
- **Geospatial Analysis**: Advanced spatial analysis tools for Cameroon's agroecological zones
- **Data Quality Control**: Automated validation and quality assessment via Pydantic v2 schemas
- **IRAD Integration**: Support for Cameroon's agricultural research centers

## Project Structure

```
agri-harvest/
├── config/                 # Configuration files
│   ├── settings.py         # Main configuration settings (env-driven)
│   ├── database.py         # MongoDB connection management
│   └── schema/             # Pydantic data schemas
│       ├── crop_schema.py
│       ├── weather_schema.py
│       └── soil_schema.py
├── models/                 # ML models and evaluation
│   ├── __init__.py         # Lazy imports (no heavy deps at import time)
│   ├── evaluator.py        # Base evaluation metrics (RMSE, MAE, R2, MAPE)
│   ├── trainer.py          # Model training orchestration
│   ├── predict.py          # Inference pipeline
│   └── v1/                 # Scaled pipeline for 10M+ rows
│       ├── config.py       # Model hyperparameters and ModelConfig dataclass
│       ├── data_loader.py  # Polars-based data loading with stratified splits
│       ├── estimators.py   # LightGBM, XGBoost, PyTorch YieldNet
│       ├── time_series.py  # LSTM + Transformer weather-aware models
│       ├── tuning.py       # Optuna hyperparameter optimization
│       ├── evaluator.py    # v1 evaluation metrics
│       └── convert_parquet.py  # CSV-to-Parquet conversion
├── utils/                  # Utility functions
│   ├── constants.py        # Application constants (bounds, crops, zones)
│   ├── date_utils.py       # Date/time and agricultural season utilities
│   ├── file_utils.py       # File handling utilities
│   └── geospatial_utils.py # Geospatial analysis tools
├── tests/                  # Test suite
│   └── unit/
│       ├── test_schemas.py
│       └── test_models_v1.py
├── notebooks/              # Jupyter notebooks for analysis
├── docs/                   # Documentation
└── pyproject.toml          # Build configuration and dependencies
```

## Installation

### Prerequisites
- Python 3.12 or higher
- MongoDB 4.4 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agri-harvest
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e "."            # Core dependencies
   pip install -e ".[ml]"        # ML dependencies (scikit-learn, polars, lightgbm, xgboost, torch, optuna, shap)
   pip install -e ".[dev]"       # Development tools (pytest, black, mypy)
   pip install -e ".[geo]"       # Geospatial tools (geopandas, rasterio)
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection details and SECRET_KEY
   ```

## Configuration

The application uses environment variables for configuration via `pydantic-settings`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGODB_DATABASE` | `cameroon_agricultural_data` | Database name |
| `SECRET_KEY` | `change-me-in-production` | Application secret key |
| `ENVIRONMENT` | `development` | Runtime environment |
| `API_HOST` | `127.0.0.1` | API bind address |
| `API_PORT` | `8000` | API port |
| `LOG_LEVEL` | `INFO` | Logging level |

See `config/settings.py` for all available configuration options.

## Usage

### ML Pipeline

```python
from models.v1.config import ModelConfig
from models.v1.data_loader import load_dataset, prepare_features, stratified_train_test_split

# Load data (supports CSV and Parquet)
df = load_dataset("data/cameroon_agricultural_features.parquet")
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, df)

# Train with default config
config = ModelConfig(use_gpu=False)
```

### Data Validation

```python
from config.schema.crop_schema import CropDataModel
from config.schema.soil_schema import SoilDataModel

# Validate crop data against Cameroon-specific constraints
crop = CropDataModel(
    field_id="CMR_FIELD_001",
    crop_type="maize",
    season="first_rainy_season",
    year=2024,
    latitude=3.8667,
    longitude=11.5167,
)
```

### Database Operations

```python
from config.database import get_database

db = await get_database()
```

### Running Tests
```bash
pytest tests/ -v
```

## Data Models

The system supports several data types, all with Pydantic v2 validation:

- **Soil Data**: Physical and chemical soil properties (USDA texture classes, CEC, pH, nutrients)
- **Weather Data**: Meteorological observations, derived indices (GDD, ET0, SPI), quality control
- **Crop Data**: Cultivation information, yield data, intercropping, crop health observations

All data models include geospatial validation (Cameroon bounds: 1.6-13.1N, 8.3-16.2E) and quality control metadata.

## ML Models

| Model | Type | Best For |
|-------|------|----------|
| LightGBM | Gradient boosting | Large tabular datasets (10M+ rows) |
| XGBoost | Gradient boosting | Balanced accuracy/speed |
| YieldNet | PyTorch feedforward | Tabular with non-linear patterns |
| HybridYieldModel | LSTM + Dense | Weather sequence + static features |
| TransformerYieldModel | Transformer + Dense | Long-range climate dependencies |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Submit a pull request

## Development Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive tests
- Update documentation for new features
- Use English for all code comments and documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the development team or create an issue in the repository.
