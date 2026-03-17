# Configuration

## Environment variables

All settings are managed via `config/settings.py` using `pydantic-settings`.
Values can be set via environment variables or a `.env` file.

### Core settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `Cameroon Agricultural Data System` | Application name |
| `APP_VERSION` | `2.0.0` | Application version |
| `DEBUG` | `False` | Debug mode |
| `ENVIRONMENT` | `development` | One of: development, testing, staging, production |

### MongoDB

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URL` | `mongodb://localhost:27017` | Connection string |
| `MONGODB_DATABASE` | `cameroon_agricultural_data` | Database name |
| `MONGODB_USERNAME` | `None` | Optional auth username |
| `MONGODB_PASSWORD` | `None` | Optional auth password |
| `MONGODB_COLLECTION_PREFIX` | `agri_` | Collection name prefix |
| `MONGODB_MAX_POOL_SIZE` | `50` | Max connection pool size |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `change-me-in-production` | Application secret key |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token expiry |

### API

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `127.0.0.1` | Bind address |
| `API_PORT` | `8000` | Port |
| `API_RELOAD` | `False` | Auto-reload on code changes |

### ML / Data

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_STORAGE_PATH` | `./models` | Model artifact storage |
| `RANDOM_SEED` | `42` | Global random seed |
| `MAX_WORKERS` | `4` | Parallel workers |
| `CHUNK_SIZE` | `1000` | Data processing chunk size |

## ModelConfig (ML pipeline)

`models/v1/config.py` defines a `ModelConfig` dataclass for ML training:

```python
from models.v1.config import ModelConfig

config = ModelConfig(
    target="yield_tha",
    test_size=0.15,
    use_gpu=False,
    lgb_num_boost_round=2000,
    lgb_early_stopping=50,
    xgb_early_stopping=50,
    yieldnet_epochs=30,
    ts_lstm_hidden=128,
    ts_epochs=30,
    optuna_n_trials=50,
    shap_sample_size=50_000,
)
```

## Database connection

```python
from config.database import get_database_config, get_database

# Singleton config
db_config = get_database_config()

# Async connection
db = await get_database()

# Health check
status = await db_config.health_check()

# Backup summary (document counts)
summary = await db_config.get_backup_summary()
```

### Transactions

```python
from config.database import get_db_transaction

async with get_db_transaction() as session:
    await collection.insert_one(doc, session=session)
    # Commits automatically on success, aborts on exception
```
