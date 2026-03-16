"""
Soil data schema optimized for ML - Cameroon
Based on IRAD infrastructure and international standards
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator
from bson import ObjectId


class TextureClass(str, Enum):
    """USDA soil texture classification"""
    SAND = "sand"
    LOAMY_SAND = "loamy_sand"
    SANDY_LOAM = "sandy_loam"
    LOAM = "loam"
    SILT_LOAM = "silt_loam"
    SILT = "silt"
    SANDY_CLAY_LOAM = "sandy_clay_loam"
    CLAY_LOAM = "clay_loam"
    SILTY_CLAY_LOAM = "silty_clay_loam"
    SANDY_CLAY = "sandy_clay"
    SILTY_CLAY = "silty_clay"
    CLAY = "clay"


class DrainageClass(str, Enum):
    """Soil drainage classification"""
    VERY_POOR = "very_poor"
    POOR = "poor"
    IMPERFECT = "imperfect"
    MODERATE = "moderate"
    GOOD = "good"
    EXCESSIVE = "excessive"


class SoilQualityLevel(str, Enum):
    """Soil parameter quality levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SamplingStrategy(str, Enum):
    """Soil sampling strategies"""
    SYSTEMATIC_GRID = "systematic_grid"
    RANDOM_STRATIFIED = "random_stratified"
    TARGETED_ZONES = "targeted_zones"
    COMPOSITE = "composite"


class CoordinatesModel(BaseModel):
    """Geographic coordinates model"""
    latitude: float = Field(..., ge=1.6, le=13.1, description="Latitude in WGS84 degrees")
    longitude: float = Field(..., ge=8.3, le=16.2, description="Longitude in WGS84 degrees")
    elevation: float = Field(..., ge=0, le=4095, description="Elevation in meters")
    slope: Optional[float] = Field(None, ge=0, le=45, description="Slope in degrees")
    aspect: Optional[float] = Field(None, ge=0, le=360, description="Aspect in degrees")


class TextureModel(BaseModel):
    """Soil texture model"""
    sand_percentage: float = Field(..., ge=0, le=100, description="Sand percentage")
    silt_percentage: float = Field(..., ge=0, le=100, description="Silt percentage")
    clay_percentage: float = Field(..., ge=0, le=100, description="Clay percentage")
    texture_class: TextureClass = Field(..., description="USDA texture class")

    @model_validator(mode='after')
    def validate_texture_sum(self) -> 'TextureModel':
        """Validates that texture fractions sum to 100%"""
        total = self.sand_percentage + self.silt_percentage + self.clay_percentage
        if abs(total - 100) > 1:  # 1% tolerance
            raise ValueError("Sand + silt + clay must equal 100%")
        return self


class WaterPropertiesModel(BaseModel):
    """Soil water properties model"""
    field_capacity: float = Field(..., ge=0.05, le=0.60, description="Field capacity (cm3/cm3)")
    wilting_point: float = Field(..., ge=0.02, le=0.40, description="Wilting point (cm3/cm3)")
    available_water_capacity: Optional[float] = Field(None, description="Available water capacity")
    saturated_hydraulic_conductivity: Optional[float] = Field(
        None, ge=0.01, le=100, description="Saturated hydraulic conductivity (cm/h)"
    )
    infiltration_rate: Optional[float] = Field(
        None, ge=1, le=300, description="Infiltration rate (mm/h)"
    )
    drainage_class: Optional[DrainageClass] = Field(None, description="Drainage class")

    @model_validator(mode='after')
    def calculate_awc(self) -> 'WaterPropertiesModel':
        """Calculates available water capacity if not provided"""
        if self.available_water_capacity is None:
            self.available_water_capacity = max(0, self.field_capacity - self.wilting_point)
        return self


class ChemicalPropertiesModel(BaseModel):
    """Soil chemical properties model"""
    ph_water: float = Field(..., ge=3.5, le=9.5, description="pH water (1:2.5)")
    ph_kcl: Optional[float] = Field(None, ge=3.0, le=8.5, description="pH KCl")
    organic_carbon: float = Field(..., ge=0.1, le=10.0, description="Organic carbon (%)")
    organic_matter: Optional[float] = Field(None, description="Organic matter (%)")
    total_nitrogen: Optional[float] = Field(None, ge=0.01, le=1.0, description="Total nitrogen (%)")
    c_n_ratio: Optional[float] = Field(None, ge=5, le=50, description="C/N ratio")

    # Macronutrients
    available_phosphorus: Optional[float] = Field(None, ge=1, le=150, description="Available P (mg/kg)")
    exchangeable_potassium: Optional[float] = Field(None, ge=0.05, le=3.0, description="Exchangeable K (cmol/kg)")
    exchangeable_calcium: Optional[float] = Field(None, ge=0.5, le=25.0, description="Exchangeable Ca (cmol/kg)")
    exchangeable_magnesium: Optional[float] = Field(None, ge=0.1, le=8.0, description="Exchangeable Mg (cmol/kg)")

    # Exchange properties
    cation_exchange_capacity: Optional[float] = Field(None, ge=2, le=60, description="CEC (cmol/kg)")
    base_saturation: Optional[float] = Field(None, ge=10, le=100, description="Base saturation (%)")
    aluminum_saturation: Optional[float] = Field(None, ge=0, le=80, description="Aluminum saturation (%)")

    @model_validator(mode='after')
    def calculate_derived_properties(self) -> 'ChemicalPropertiesModel':
        """Calculates organic matter and C/N ratio if not provided"""
        if self.organic_matter is None:
            self.organic_matter = self.organic_carbon * 1.724  # van Bemmelen factor
        if self.c_n_ratio is None and self.total_nitrogen is not None and self.total_nitrogen > 0:
            self.c_n_ratio = self.organic_carbon / self.total_nitrogen
        return self


class MicronutrientsModel(BaseModel):
    """Soil micronutrients model"""
    iron: Optional[float] = Field(None, ge=1, le=100, description="Available Fe (mg/kg)")
    manganese: Optional[float] = Field(None, ge=0.5, le=50, description="Available Mn (mg/kg)")
    zinc: Optional[float] = Field(None, ge=0.1, le=20, description="Available Zn (mg/kg)")
    copper: Optional[float] = Field(None, ge=0.1, le=10, description="Available Cu (mg/kg)")
    boron: Optional[float] = Field(None, ge=0.1, le=5, description="Available B (mg/kg)")


class PhysicalPropertiesModel(BaseModel):
    """Soil physical properties model"""
    bulk_density: float = Field(..., ge=0.8, le=2.0, description="Bulk density (g/cm3)")
    porosity: Optional[float] = Field(None, ge=20, le=70, description="Total porosity (%)")
    aggregate_stability: Optional[float] = Field(None, ge=0, le=100, description="Aggregate stability (%)")

    @model_validator(mode='after')
    def calculate_porosity(self) -> 'PhysicalPropertiesModel':
        """Calculates porosity if not provided (particle density assumed 2.65 g/cm3)"""
        if self.porosity is None:
            self.porosity = (1 - self.bulk_density / 2.65) * 100
        return self


class SoilQualityIndex(BaseModel):
    """Soil quality index"""
    fertility_index: Optional[float] = Field(None, ge=0, le=1, description="Fertility index")
    health_index: Optional[float] = Field(None, ge=0, le=1, description="Health index")
    productivity_potential: Optional[float] = Field(None, ge=0, le=1, description="Productivity potential")
    limiting_factors: Optional[List[str]] = Field(None, description="Limiting factors")


class SamplingMetadata(BaseModel):
    """Sampling metadata"""
    sampling_date: datetime = Field(..., description="Sampling date")
    sampling_strategy: SamplingStrategy = Field(..., description="Sampling strategy")
    sampling_depth: str = Field("0-20cm", description="Sampling depth")
    sample_id: Optional[str] = Field(None, description="Sample identifier")
    laboratory_id: Optional[str] = Field(None, description="Laboratory identifier")
    analysis_date: Optional[datetime] = Field(None, description="Analysis date")
    quality_flag: Optional[int] = Field(None, ge=0, le=3, description="Quality flag (0=good, 3=invalid)")


class SoilDataModel(BaseModel):
    """Main soil data model"""
    # Identifiers and metadata
    id: Optional[str] = Field(None, alias="_id")
    field_id: str = Field(..., description="Unique field identifier")

    # Location
    coordinates: CoordinatesModel = Field(..., description="Geographic coordinates")
    agroecological_zone: Optional[str] = Field(None, description="Agroecological zone")

    # Sampling
    sampling_metadata: SamplingMetadata = Field(..., description="Sampling metadata")

    # Soil properties
    texture: TextureModel = Field(..., description="Texture properties")
    physical_properties: PhysicalPropertiesModel = Field(..., description="Physical properties")
    chemical_properties: ChemicalPropertiesModel = Field(..., description="Chemical properties")
    water_properties: WaterPropertiesModel = Field(..., description="Water properties")
    micronutrients: Optional[MicronutrientsModel] = Field(None, description="Micronutrients")

    # Quality index
    quality_indices: Optional[SoilQualityIndex] = Field(None, description="Quality indices")

    # System metadata
    created_at: Optional[datetime] = Field(None, description="Creation date")
    updated_at: Optional[datetime] = Field(None, description="Update date")
    version: Optional[int] = Field(1, description="Document version")
    data_source: Optional[str] = Field(None, description="Data source")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={
            ObjectId: str,
            datetime: lambda v: v.isoformat(),
        },
    )


class SoilSchema:
    """MongoDB schema for soil data"""

    @staticmethod
    def get_collection_name() -> str:
        return "agri_soil_data"

    @staticmethod
    def get_indexes() -> List[Dict[str, Any]]:
        return [
            {"coordinates.coordinates": "2dsphere"},
            {"coordinates.latitude": 1, "coordinates.longitude": 1},
            {"sampling_metadata.sampling_date": -1},
            {"created_at": -1},
            {"field_id": 1},
            {"chemical_properties.ph_water": 1},
            {"chemical_properties.organic_carbon": 1},
            {"texture.texture_class": 1},
            {"agroecological_zone": 1},
            {
                "coordinates.coordinates": "2dsphere",
                "sampling_metadata.sampling_date": -1,
            },
            {
                "chemical_properties.ph_water": 1,
                "chemical_properties.organic_carbon": 1,
            },
            {
                "texture.clay_percentage": 1,
                "chemical_properties.cation_exchange_capacity": 1,
            },
        ]

    @staticmethod
    def get_validation_schema() -> Dict[str, Any]:
        return {
            "$jsonSchema": {
                "bsonType": "object",
                "required": [
                    "field_id", "coordinates", "sampling_metadata",
                    "texture", "physical_properties",
                    "chemical_properties", "water_properties",
                ],
                "properties": {
                    "field_id": {
                        "bsonType": "string",
                        "description": "Unique field identifier required",
                    },
                    "coordinates": {
                        "bsonType": "object",
                        "required": ["latitude", "longitude", "elevation"],
                        "properties": {
                            "latitude": {"bsonType": "double", "minimum": 1.6, "maximum": 13.1},
                            "longitude": {"bsonType": "double", "minimum": 8.3, "maximum": 16.2},
                            "elevation": {"bsonType": "double", "minimum": 0, "maximum": 4095},
                        },
                    },
                    "chemical_properties.ph_water": {
                        "bsonType": "double",
                        "minimum": 3.5,
                        "maximum": 9.5,
                    },
                    "chemical_properties.organic_carbon": {
                        "bsonType": "double",
                        "minimum": 0.1,
                        "maximum": 10.0,
                    },
                    "texture": {
                        "bsonType": "object",
                        "required": [
                            "sand_percentage", "silt_percentage",
                            "clay_percentage", "texture_class",
                        ],
                    },
                    "physical_properties.bulk_density": {
                        "bsonType": "double",
                        "minimum": 0.8,
                        "maximum": 2.0,
                    },
                },
            }
        }

    @staticmethod
    def get_sample_document() -> Dict[str, Any]:
        return {
            "field_id": "CMR_FIELD_001",
            "coordinates": {
                "latitude": 3.8667,
                "longitude": 11.5167,
                "elevation": 650,
                "slope": 2.5,
                "aspect": 180,
            },
            "agroecological_zone": "humid_forest_inland",
            "sampling_metadata": {
                "sampling_date": datetime.now(),
                "sampling_strategy": "systematic_grid",
                "sampling_depth": "0-20cm",
                "sample_id": "SOIL_001",
                "laboratory_id": "IRAD_LAB_YDE",
                "quality_flag": 0,
            },
            "texture": {
                "sand_percentage": 45.2,
                "silt_percentage": 32.8,
                "clay_percentage": 22.0,
                "texture_class": "loam",
            },
            "physical_properties": {
                "bulk_density": 1.35,
                "porosity": 49.1,
                "aggregate_stability": 75.5,
            },
            "chemical_properties": {
                "ph_water": 6.2,
                "ph_kcl": 5.8,
                "organic_carbon": 2.1,
                "organic_matter": 3.6,
                "total_nitrogen": 0.18,
                "c_n_ratio": 11.7,
                "available_phosphorus": 18.5,
                "exchangeable_potassium": 0.25,
                "exchangeable_calcium": 8.2,
                "exchangeable_magnesium": 2.1,
                "cation_exchange_capacity": 12.5,
                "base_saturation": 85.6,
                "aluminum_saturation": 5.2,
            },
            "water_properties": {
                "field_capacity": 0.32,
                "wilting_point": 0.18,
                "available_water_capacity": 0.14,
                "saturated_hydraulic_conductivity": 12.5,
                "infiltration_rate": 45.2,
                "drainage_class": "good",
            },
            "micronutrients": {
                "iron": 25.8,
                "manganese": 12.3,
                "zinc": 2.1,
                "copper": 1.8,
                "boron": 0.8,
            },
            "quality_indices": {
                "fertility_index": 0.82,
                "health_index": 0.78,
                "productivity_potential": 0.85,
                "limiting_factors": [],
            },
            "data_source": "IRAD_field_survey",
            "created_at": datetime.now(),
            "version": 1,
        }
