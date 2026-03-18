"""
Configuration settings for Cameroon Agricultural Data Management System
Centralized configuration settings with MongoDB support
"""

from typing import Optional, List, Dict, Any
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

from utils.constants import CAMEROON_BOUNDS, MAIN_CROPS

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Centralized configuration for Cameroon agricultural data system"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_assignment=True,
    )

    # Project metadata
    app_name: str = Field("Cameroon Agricultural Data System", alias="APP_NAME")
    app_version: str = Field("2.0.0", alias="APP_VERSION")
    debug: bool = Field(False, alias="DEBUG")
    environment: str = Field("development", alias="ENVIRONMENT")

    # MongoDB configuration
    mongodb_username: Optional[str] = Field(None, alias="MONGODB_USERNAME")
    mongodb_password: Optional[str] = Field(None, alias="MONGODB_PASSWORD")
    mongodb_url: str = Field("mongodb://localhost:27017", alias="MONGODB_URL")
    mongodb_database: str = Field("cameroon_agricultural_data", alias="MONGODB_DATABASE")
    mongodb_collection_prefix: str = Field("agri_", alias="MONGODB_COLLECTION_PREFIX")
    mongodb_max_pool_size: int = Field(50, alias="MONGODB_MAX_POOL_SIZE")
    mongodb_min_pool_size: int = Field(5, alias="MONGODB_MIN_POOL_SIZE")
    mongodb_socket_timeout_ms: int = Field(60000, alias="MONGODB_SOCKET_TIMEOUT_MS")
    mongodb_connect_timeout_ms: int = Field(20000, alias="MONGODB_CONNECT_TIMEOUT_MS")
    mongodb_server_selection_timeout_ms: int = Field(10000, alias="MONGODB_SERVER_SELECTION_TIMEOUT_MS")

    # Cameroon regional configuration
    default_timezone: str = Field("Africa/Douala", alias="DEFAULT_TIMEZONE")
    default_coordinate_system: str = Field("WGS84", alias="DEFAULT_COORDINATE_SYSTEM")
    country_bounds: Dict[str, float] = Field(
        default=CAMEROON_BOUNDS,
    )

    # IRAD configuration
    irad_data_access_email: str = Field("dg@irad-cameroon.org", alias="IRAD_DATA_ACCESS_EMAIL")
    irad_data_quality_threshold: float = Field(0.8, alias="IRAD_DATA_QUALITY_THRESHOLD")

    # Synthetic data configuration
    synthetic_data_size: int = Field(10000, alias="SYNTHETIC_DATA_SIZE")
    random_seed: int = Field(42, alias="RANDOM_SEED")
    enable_synthetic_generation: bool = Field(True, alias="ENABLE_SYNTHETIC_GENERATION")

    # ML/AI configuration
    model_storage_path: str = Field("./models", alias="MODEL_STORAGE_PATH")
    feature_store_enabled: bool = Field(True, alias="FEATURE_STORE_ENABLED")
    ml_experiment_tracking: bool = Field(True, alias="ML_EXPERIMENT_TRACKING")

    # API configuration
    api_host: str = Field("127.0.0.1", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_reload: bool = Field(False, alias="API_RELOAD")
    api_cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="API_CORS_ORIGINS",
    )

    # Logging configuration
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT",
    )
    log_file_path: Optional[str] = Field(None, alias="LOG_FILE_PATH")

    # Security configuration
    secret_key: str = Field("change-me-in-production", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # External data configuration
    weather_api_key: Optional[str] = Field(None, alias="WEATHER_API_KEY")
    satellite_data_sources: List[str] = Field(
        ["sentinel-2", "landsat-8", "modis"],
        alias="SATELLITE_DATA_SOURCES",
    )

    # Agroecological zones configuration
    agroecological_zones: Dict[str, Dict[str, Any]] = Field(
        default={
            "sahel_savanna": {
                "description": "Northern Sahel zone",
                "rainfall_range": [300, 600],
                "temperature_range": [28, 35],
                "elevation_range": [300, 500],
                "rainfall_regime": "monomodal",
            },
            "sudan_savanna": {
                "description": "Sudanian savanna",
                "rainfall_range": [600, 1000],
                "temperature_range": [26, 32],
                "elevation_range": [350, 800],
                "rainfall_regime": "monomodal",
            },
            "guinea_savanna": {
                "description": "Guinean savanna",
                "rainfall_range": [1000, 1400],
                "temperature_range": [24, 30],
                "elevation_range": [400, 1200],
                "rainfall_regime": "monomodal",
            },
            "forest_savanna_transition": {
                "description": "Forest-savanna transition zone",
                "rainfall_range": [1400, 1800],
                "temperature_range": [23, 28],
                "elevation_range": [400, 1000],
                "rainfall_regime": "bimodal",
            },
            "humid_forest_inland": {
                "description": "Continental humid forest",
                "rainfall_range": [1600, 2500],
                "temperature_range": [24, 28],
                "elevation_range": [300, 800],
                "rainfall_regime": "bimodal",
            },
            "humid_forest_coast": {
                "description": "Coastal humid forest",
                "rainfall_range": [2000, 4000],
                "temperature_range": [25, 29],
                "elevation_range": [0, 500],
                "rainfall_regime": "bimodal",
            },
            "western_highlands": {
                "description": "Western highlands",
                "rainfall_range": [1800, 2500],
                "temperature_range": [16, 24],
                "elevation_range": [1200, 2500],
                "rainfall_regime": "monomodal",
            },
        }
    )

    # Main crops configuration
    main_crops: List[str] = Field(
        default=MAIN_CROPS,
        alias="MAIN_CROPS",
    )

    # Data validation configuration
    data_validation_enabled: bool = Field(True, alias="DATA_VALIDATION_ENABLED")
    data_quality_min_score: float = Field(0.7, alias="DATA_QUALITY_MIN_SCORE")
    outlier_detection_enabled: bool = Field(True, alias="OUTLIER_DETECTION_ENABLED")

    # Performance configuration
    max_workers: int = Field(4, alias="MAX_WORKERS")
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    cache_ttl_seconds: int = Field(3600, alias="CACHE_TTL_SECONDS")

    # Monitoring configuration
    enable_monitoring: bool = Field(True, alias="ENABLE_MONITORING")
    metrics_export_interval: int = Field(60, alias="METRICS_EXPORT_INTERVAL")
    health_check_interval: int = Field(30, alias="HEALTH_CHECK_INTERVAL")

    @field_validator("mongodb_url")
    @classmethod
    def validate_mongodb_url(cls, v: str) -> str:
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError("MongoDB URL must start with mongodb:// or mongodb+srv://")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed_envs = ["development", "testing", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()

    @field_validator("country_bounds")
    @classmethod
    def validate_country_bounds(cls, v: Dict[str, float]) -> Dict[str, float]:
        required_keys = ["north", "south", "east", "west"]
        if not all(key in v for key in required_keys):
            raise ValueError(f"Country bounds must contain: {required_keys}")
        if v["south"] >= v["north"]:
            raise ValueError("South bound must be less than north bound")
        if v["west"] >= v["east"]:
            raise ValueError("West bound must be less than east bound")
        return v

    @property
    def mongodb_connection_string(self) -> str:
        url = self.mongodb_url
        if self.mongodb_username and self.mongodb_password:
            # Inject credentials into the connection URL
            from urllib.parse import quote_plus
            creds = f"{quote_plus(self.mongodb_username)}:{quote_plus(self.mongodb_password)}@"
            if "://" in url:
                scheme, rest = url.split("://", 1)
                url = f"{scheme}://{creds}{rest}"
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}maxPoolSize={self.mongodb_max_pool_size}"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    def get_collection_name(self, collection_type: str) -> str:
        return f"{self.mongodb_collection_prefix}{collection_type}"

    def get_irad_center_config(self, center_name: str) -> Optional[Dict[str, Any]]:
        from utils.constants import IRAD_CENTERS
        return IRAD_CENTERS.get(center_name)


@lru_cache()
def get_settings() -> Settings:
    """
    Cached function to get configuration settings.
    Uses lru_cache to avoid reloading configuration on each call.
    """
    try:
        settings = Settings()
        logger.info("Configuration loaded for environment: %s", settings.environment)
        return settings
    except Exception as e:
        logger.error("Error loading configuration: %s", e)
        raise


def setup_logging(settings: Settings) -> None:
    """Configures the logging system"""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": settings.log_format,
            },
        },
        "handlers": {
            "console": {
                "level": settings.log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
            "agricultural_data": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False,
            },
        },
    }

    if settings.log_file_path:
        log_config["handlers"]["file"] = {
            "level": settings.log_level,
            "class": "logging.handlers.RotatingFileHandler",
            "filename": settings.log_file_path,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "standard",
        }
        log_config["loggers"][""]["handlers"].append("file")
        log_config["loggers"]["agricultural_data"]["handlers"].append("file")

    import logging.config
    logging.config.dictConfig(log_config)
