"""
Weather data schema optimized for ML - Cameroon
Based on meteorological infrastructure and agricultural needs
"""

from typing import Optional, List, Dict, Any
import datetime as _dt
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator
from utils.date_utils import get_agricultural_season


class StationType(str, Enum):
    """Meteorological station types"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"
    SATELLITE_DERIVED = "satellite_derived"


class DataQuality(str, Enum):
    """Weather data quality levels"""
    GOOD = "good"
    QUESTIONABLE = "questionable"
    POOR = "poor"
    MISSING = "missing"
    ESTIMATED = "estimated"


class InterpolationMethod(str, Enum):
    """Spatial interpolation methods"""
    IDW = "idw"
    KRIGING = "kriging"
    SPLINE = "spline"
    REANALYSIS_BLEND = "reanalysis_blend"


class PrecipitationType(str, Enum):
    """Precipitation types"""
    RAIN = "rain"
    DRIZZLE = "drizzle"
    SHOWER = "shower"
    THUNDERSTORM = "thunderstorm"
    HAIL = "hail"


class RainfallRegime(str, Enum):
    """Rainfall regime types in Cameroon"""
    BIMODAL = "bimodal"      # Southern Cameroon (below ~6N)
    MONOMODAL = "monomodal"  # Northern Cameroon (above ~6N)


class WeatherStationModel(BaseModel):
    """Meteorological station model"""
    station_id: str = Field(..., description="Unique station identifier")
    station_name: str = Field(..., description="Station name")
    station_type: StationType = Field(..., description="Station type")

    # Location
    latitude: float = Field(..., ge=1.6, le=13.1, description="Latitude WGS84")
    longitude: float = Field(..., ge=8.3, le=16.2, description="Longitude WGS84")
    elevation: float = Field(..., ge=0, le=4095, description="Elevation in meters")

    # Measurement characteristics
    measurement_height_temp: Optional[float] = Field(2.0, description="Temperature measurement height (m)")
    measurement_height_wind: Optional[float] = Field(10.0, description="Wind measurement height (m)")
    measurement_height_precip: Optional[float] = Field(1.5, description="Rain gauge height (m)")

    # Environment
    surrounding_environment: Optional[str] = Field(None, description="Station environment")
    microclimate_effects: Optional[bool] = Field(False, description="Microclimate effects")

    # Operational metadata
    installation_date: Optional[_dt.date] = Field(None, description="Installation date")
    data_availability_start: Optional[_dt.date] = Field(None, description="Data availability start")
    data_availability_end: Optional[_dt.date] = Field(None, description="Data availability end")
    completeness_percentage: Optional[float] = Field(None, ge=0, le=100, description="Completeness %")

    # Contact and management
    operator: Optional[str] = Field(None, description="Station operator")
    maintenance_schedule: Optional[str] = Field(None, description="Maintenance schedule")
    calibration_date: Optional[_dt.date] = Field(None, description="Last calibration date")

    # Status
    is_active: bool = Field(True, description="Active station")
    data_quality_rating: Optional[float] = Field(None, ge=0, le=1, description="Data quality rating")

    # Rainfall regime
    rainfall_regime: Optional[RainfallRegime] = Field(None, description="Rainfall regime")

    @model_validator(mode='after')
    def infer_rainfall_regime(self) -> 'WeatherStationModel':
        """Infers rainfall regime from latitude if not provided"""
        if self.rainfall_regime is None:
            self.rainfall_regime = (
                RainfallRegime.MONOMODAL if self.latitude >= 6.0
                else RainfallRegime.BIMODAL
            )
        return self


class TemperatureModel(BaseModel):
    """Temperature data model"""
    temperature_min: Optional[float] = Field(None, ge=-5, le=45, description="Min temperature (C)")
    temperature_max: Optional[float] = Field(None, ge=5, le=50, description="Max temperature (C)")
    temperature_avg: Optional[float] = Field(None, ge=0, le=40, description="Avg temperature (C)")
    temperature_range: Optional[float] = Field(None, description="Diurnal temperature range")

    # Soil temperature
    soil_temp_5cm: Optional[float] = Field(None, ge=10, le=45, description="Soil temp 5cm (C)")
    soil_temp_10cm: Optional[float] = Field(None, ge=10, le=45, description="Soil temp 10cm (C)")
    soil_temp_20cm: Optional[float] = Field(None, ge=10, le=45, description="Soil temp 20cm (C)")

    # Dew point
    dew_point: Optional[float] = Field(None, ge=-10, le=35, description="Dew point (C)")

    @model_validator(mode='after')
    def calculate_derived_temps(self) -> 'TemperatureModel':
        """Calculates average and range if not provided"""
        if self.temperature_min is not None and self.temperature_max is not None:
            if self.temperature_avg is None:
                self.temperature_avg = (self.temperature_min + self.temperature_max) / 2
            if self.temperature_range is None:
                self.temperature_range = self.temperature_max - self.temperature_min
        return self


class PrecipitationModel(BaseModel):
    """Precipitation data model"""
    precipitation_daily: Optional[float] = Field(None, ge=0, le=500, description="Daily precipitation (mm)")
    precipitation_intensity_max: Optional[float] = Field(None, ge=0, le=200, description="Max intensity (mm/h)")
    precipitation_duration: Optional[float] = Field(None, ge=0, le=24, description="Precipitation duration (h)")
    precipitation_type: Optional[PrecipitationType] = Field(None, description="Precipitation type")

    # Event characteristics
    storm_occurrence: Optional[bool] = Field(None, description="Storm occurrence")
    thunder_occurrence: Optional[bool] = Field(None, description="Thunder occurrence")
    hail_occurrence: Optional[bool] = Field(None, description="Hail occurrence")

    # Associated measurements
    wind_with_precipitation: Optional[float] = Field(None, ge=0, le=50, description="Wind with precip (m/s)")


class HumidityModel(BaseModel):
    """Humidity data model"""
    relative_humidity_avg: Optional[float] = Field(None, ge=0, le=100, description="Avg relative humidity (%)")
    relative_humidity_min: Optional[float] = Field(None, ge=0, le=100, description="Min relative humidity (%)")
    relative_humidity_max: Optional[float] = Field(None, ge=0, le=100, description="Max relative humidity (%)")

    # Humidity at specific times
    rh_morning: Optional[float] = Field(None, ge=0, le=100, description="Morning RH 6h (%)")
    rh_afternoon: Optional[float] = Field(None, ge=0, le=100, description="Afternoon RH 14h (%)")

    # Vapor pressure
    actual_vapor_pressure: Optional[float] = Field(None, description="Actual vapor pressure (kPa)")
    saturation_vapor_pressure: Optional[float] = Field(None, description="Saturation vapor pressure (kPa)")
    vapor_pressure_deficit: Optional[float] = Field(None, description="Vapor pressure deficit (kPa)")

    # Specific humidity
    specific_humidity: Optional[float] = Field(None, ge=0, le=30, description="Specific humidity (g/kg)")


class RadiationModel(BaseModel):
    """Radiation data model"""
    solar_radiation_daily: Optional[float] = Field(None, ge=0, le=35, description="Solar radiation (MJ/m2/day)")
    sunshine_hours: Optional[float] = Field(None, ge=0, le=14, description="Sunshine hours")
    sunshine_fraction: Optional[float] = Field(None, ge=0, le=1, description="Sunshine fraction")

    # Radiation components
    global_radiation: Optional[float] = Field(None, description="Global radiation")
    direct_radiation: Optional[float] = Field(None, description="Direct radiation")
    diffuse_radiation: Optional[float] = Field(None, description="Diffuse radiation")

    # UV index and photoperiod
    uv_index: Optional[float] = Field(None, ge=0, le=15, description="UV index")
    photoperiod: Optional[float] = Field(None, ge=10.5, le=13.5, description="Photoperiod (h)")

    # Net radiation
    net_radiation: Optional[float] = Field(None, description="Net radiation (MJ/m2/day)")


class WindModel(BaseModel):
    """Wind data model"""
    wind_speed_avg: Optional[float] = Field(None, ge=0, le=50, description="Avg wind speed (m/s)")
    wind_speed_max: Optional[float] = Field(None, ge=0, le=100, description="Max wind speed (m/s)")
    wind_direction: Optional[float] = Field(None, ge=0, le=360, description="Wind direction (degrees)")
    wind_gust: Optional[float] = Field(None, ge=0, le=100, description="Gusts (m/s)")

    # Wind components
    wind_u_component: Optional[float] = Field(None, description="U wind component")
    wind_v_component: Optional[float] = Field(None, description="V wind component")


class PressureModel(BaseModel):
    """Pressure data model -- bounds adjusted for Cameroon's highland elevations"""
    atmospheric_pressure: Optional[float] = Field(
        None, ge=600, le=1040,
        description="Atmospheric pressure (hPa) -- lower bound supports elevations up to ~4000m"
    )
    sea_level_pressure: Optional[float] = Field(None, ge=990, le=1035, description="Sea level pressure (hPa)")
    pressure_tendency: Optional[float] = Field(None, description="Pressure tendency (hPa/3h)")


class DerivedIndicesModel(BaseModel):
    """Derived indices model"""
    # Thermal indices
    growing_degree_days_base10: Optional[float] = Field(None, description="GDD base 10C")
    growing_degree_days_base15: Optional[float] = Field(None, description="GDD base 15C")
    cooling_degree_days: Optional[float] = Field(None, description="Cooling degree days")
    heat_index: Optional[float] = Field(None, description="Heat index")

    # Evapotranspiration
    reference_et_penman_monteith: Optional[float] = Field(None, description="ET0 Penman-Monteith (mm)")
    potential_et_priestley_taylor: Optional[float] = Field(None, description="PET Priestley-Taylor (mm)")

    # Drought indices
    aridity_index: Optional[float] = Field(None, description="Aridity index")
    standardized_precipitation_index: Optional[float] = Field(None, description="SPI")

    # Water balance
    water_balance: Optional[float] = Field(None, description="Water balance (mm)")
    cumulative_water_deficit: Optional[float] = Field(None, description="Cumulative water deficit (mm)")


class ExtremeEventsModel(BaseModel):
    """Extreme events model"""
    # Temperature events
    frost_occurrence: Optional[bool] = Field(None, description="Frost occurrence")
    frost_duration_hours: Optional[float] = Field(None, description="Frost duration (hours)")
    heat_wave_occurrence: Optional[bool] = Field(None, description="Heat wave occurrence")
    heat_wave_duration_days: Optional[int] = Field(None, description="Heat wave duration (days)")

    # Precipitation events
    heavy_rainfall_event: Optional[bool] = Field(None, description="Heavy rainfall")
    drought_event: Optional[bool] = Field(None, description="Drought event")
    dry_spell_length: Optional[int] = Field(None, description="Dry spell length (days)")

    # Wind events
    high_wind_event: Optional[bool] = Field(None, description="High wind")
    storm_damage_potential: Optional[float] = Field(None, description="Storm damage potential")


class QualityControlModel(BaseModel):
    """Quality control model"""
    # Quality flags by variable
    temperature_quality: Optional[DataQuality] = Field(None, description="Temperature quality")
    precipitation_quality: Optional[DataQuality] = Field(None, description="Precipitation quality")
    humidity_quality: Optional[DataQuality] = Field(None, description="Humidity quality")
    wind_quality: Optional[DataQuality] = Field(None, description="Wind quality")
    radiation_quality: Optional[DataQuality] = Field(None, description="Radiation quality")

    # Quality metadata
    overall_quality_score: Optional[float] = Field(None, ge=0, le=1, description="Overall quality score")
    data_completeness: Optional[float] = Field(None, ge=0, le=100, description="Data completeness (%)")
    temporal_consistency: Optional[bool] = Field(None, description="Temporal consistency")
    spatial_consistency: Optional[bool] = Field(None, description="Spatial consistency")

    # Applied methods
    gap_filling_applied: Optional[bool] = Field(None, description="Gap filling applied")
    interpolation_method: Optional[InterpolationMethod] = Field(None, description="Interpolation method")
    bias_correction_applied: Optional[bool] = Field(None, description="Bias correction applied")


class WeatherDataModel(BaseModel):
    """Main weather data model"""
    # Identifiers and metadata
    id: Optional[str] = Field(None, alias="_id")
    station_id: str = Field(..., description="Station identifier")
    date: _dt.date = Field(..., description="Measurement date")

    # Location (may differ from station for interpolated data)
    latitude: Optional[float] = Field(None, ge=1.6, le=13.1, description="Point latitude")
    longitude: Optional[float] = Field(None, ge=8.3, le=16.2, description="Point longitude")
    elevation: Optional[float] = Field(None, ge=0, le=4095, description="Point elevation")

    # Weather data
    temperature: Optional[TemperatureModel] = Field(None, description="Temperature data")
    precipitation: Optional[PrecipitationModel] = Field(None, description="Precipitation data")
    humidity: Optional[HumidityModel] = Field(None, description="Humidity data")
    radiation: Optional[RadiationModel] = Field(None, description="Radiation data")
    wind: Optional[WindModel] = Field(None, description="Wind data")
    pressure: Optional[PressureModel] = Field(None, description="Pressure data")

    # Derived indices and extreme events
    derived_indices: Optional[DerivedIndicesModel] = Field(None, description="Derived indices")
    extreme_events: Optional[ExtremeEventsModel] = Field(None, description="Extreme events")

    # Quality control
    quality_control: Optional[QualityControlModel] = Field(None, description="Quality control")

    # Temporal metadata
    day_of_year: Optional[int] = Field(None, ge=1, le=366, description="Day of year")
    season: Optional[str] = Field(None, description="Season")
    rainfall_regime: Optional[RainfallRegime] = Field(None, description="Rainfall regime")
    agricultural_day: Optional[int] = Field(None, description="Agricultural day since season start")

    # System metadata
    created_at: Optional[datetime] = Field(None, description="Creation date")
    updated_at: Optional[datetime] = Field(None, description="Update date")
    version: Optional[int] = Field(1, description="Document version")
    data_source: Optional[str] = Field(None, description="Data source")
    processing_level: Optional[str] = Field(None, description="Processing level")

    @model_validator(mode='after')
    def calculate_temporal_fields(self) -> 'WeatherDataModel':
        """Calculates day_of_year, season, and rainfall regime if not provided"""
        if self.day_of_year is None:
            self.day_of_year = self.date.timetuple().tm_yday

        if self.season is None:
            self.season = get_agricultural_season(self.date, self.latitude)

        if self.rainfall_regime is None and self.latitude is not None:
            self.rainfall_regime = (
                RainfallRegime.MONOMODAL if self.latitude >= 6.0
                else RainfallRegime.BIMODAL
            )
        return self

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class WeatherSchema:
    """MongoDB schema for weather data"""

    @staticmethod
    def get_collection_name() -> str:
        return "agri_weather_data"

    @staticmethod
    def get_stations_collection_name() -> str:
        return "agri_weather_stations"

    @staticmethod
    def get_indexes() -> List[Dict[str, Any]]:
        return [
            {"date": -1},
            {"created_at": -1},
            {"latitude": 1, "longitude": 1},
            {"station_id": 1},
            {"station_id": 1, "date": -1},
            {"temperature.temperature_avg": 1},
            {"precipitation.precipitation_daily": 1},
            {"humidity.relative_humidity_avg": 1},
            {"radiation.solar_radiation_daily": 1},
            {"quality_control.overall_quality_score": 1},
            {"data_source": 1},
            {"date": -1, "station_id": 1, "quality_control.overall_quality_score": 1},
            {"extreme_events.frost_occurrence": 1},
            {"extreme_events.heat_wave_occurrence": 1},
            {"extreme_events.heavy_rainfall_event": 1},
            {"derived_indices.reference_et_penman_monteith": 1},
            {"derived_indices.growing_degree_days_base10": 1},
        ]

    @staticmethod
    def get_stations_indexes() -> List[Dict[str, Any]]:
        return [
            {"latitude": 1, "longitude": 1},
            {"elevation": 1},
            {"station_id": 1},
            {"station_type": 1},
            {"is_active": 1},
            {"operator": 1},
            {"installation_date": 1},
            {"data_availability_start": 1, "data_availability_end": 1},
            {"data_quality_rating": 1},
        ]

    @staticmethod
    def get_validation_schema() -> Dict[str, Any]:
        return {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["station_id", "date"],
                "properties": {
                    "station_id": {
                        "bsonType": "string",
                        "description": "Station identifier required",
                    },
                    "date": {
                        "bsonType": "date",
                        "description": "Measurement date required",
                    },
                    "latitude": {"bsonType": "double", "minimum": 1.6, "maximum": 13.1},
                    "longitude": {"bsonType": "double", "minimum": 8.3, "maximum": 16.2},
                    "elevation": {"bsonType": "double", "minimum": 0, "maximum": 4095},
                    "temperature.temperature_min": {
                        "bsonType": "double",
                        "minimum": -5,
                        "maximum": 45,
                    },
                    "temperature.temperature_max": {
                        "bsonType": "double",
                        "minimum": 5,
                        "maximum": 50,
                    },
                    "precipitation.precipitation_daily": {
                        "bsonType": "double",
                        "minimum": 0,
                        "maximum": 500,
                    },
                },
            }
        }

    @staticmethod
    def get_sample_document() -> Dict[str, Any]:
        return {
            "station_id": "CMR_WEATHER_001",
            "date": _dt.date.today(),
            "latitude": 3.8667,
            "longitude": 11.5167,
            "elevation": 650,
            "temperature": {
                "temperature_min": 22.5,
                "temperature_max": 31.2,
                "temperature_avg": 26.9,
                "temperature_range": 8.7,
                "soil_temp_5cm": 25.8,
                "dew_point": 21.5,
            },
            "precipitation": {
                "precipitation_daily": 15.2,
                "precipitation_intensity_max": 25.8,
                "precipitation_duration": 2.5,
                "precipitation_type": "shower",
                "thunder_occurrence": True,
            },
            "humidity": {
                "relative_humidity_avg": 78.5,
                "rh_morning": 95.2,
                "rh_afternoon": 62.3,
                "vapor_pressure_deficit": 1.2,
            },
            "radiation": {
                "solar_radiation_daily": 18.5,
                "sunshine_hours": 6.8,
                "sunshine_fraction": 0.57,
            },
            "wind": {
                "wind_speed_avg": 2.1,
                "wind_speed_max": 8.5,
                "wind_direction": 225,
            },
            "pressure": {
                "atmospheric_pressure": 1013.2,
                "sea_level_pressure": 1013.8,
            },
            "derived_indices": {
                "growing_degree_days_base10": 16.9,
                "reference_et_penman_monteith": 4.2,
                "water_balance": 11.0,
            },
            "quality_control": {
                "overall_quality_score": 0.92,
                "data_completeness": 98.5,
                "temperature_quality": "good",
                "precipitation_quality": "good",
            },
            "day_of_year": 45,
            "season": "grand_dry_season",
            "rainfall_regime": "bimodal",
            "data_source": "automatic_station",
            "created_at": datetime.now(timezone.utc),
            "version": 1,
        }
