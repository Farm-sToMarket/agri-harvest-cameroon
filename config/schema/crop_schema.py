"""
Crop data schema optimized for ML - Cameroon
Based on IRAD agricultural taxonomy and local cultivation practices
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from bson import ObjectId


class CropGroup(str, Enum):
    """Crop groups following agricultural taxonomy"""
    CEREALS = "cereals"
    LEGUMES = "legumes"
    ROOT_TUBERS = "root_tubers"
    VEGETABLES = "vegetables"
    TREE_CROPS = "tree_crops"
    CASH_CROPS = "cash_crops"
    INDUSTRIAL_CROPS = "industrial_crops"


class CropType(str, Enum):
    """Main crop types in Cameroon"""
    # Cereals
    MAIZE = "maize"
    RICE = "rice"
    SORGHUM = "sorghum"
    MILLET = "millet"

    # Legumes
    COWPEA = "cowpea"
    GROUNDNUT = "groundnut"
    COMMON_BEAN = "common_bean"
    SOYBEAN = "soybean"

    # Root tubers
    CASSAVA = "cassava"
    SWEET_POTATO = "sweet_potato"
    YAM = "yam"
    POTATO = "potato"

    # Vegetables
    TOMATO = "tomato"
    PEPPER = "pepper"
    ONION = "onion"
    CABBAGE = "cabbage"
    CARROT = "carrot"
    LETTUCE = "lettuce"
    CUCUMBER = "cucumber"
    OKRA = "okra"

    # Industrial crops
    COTTON = "cotton"

    # Perennial crops
    PLANTAIN_BANANA = "plantain_banana"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    MANGO = "mango"
    PINEAPPLE = "pineapple"


class Season(str, Enum):
    """Agricultural seasons in Cameroon"""
    # Bimodal (South)
    GRAND_DRY_SEASON = "grand_dry_season"
    FIRST_RAINY_SEASON = "first_rainy_season"
    PETIT_DRY_SEASON = "petit_dry_season"
    SECOND_RAINY_SEASON = "second_rainy_season"
    TRANSITION_TO_DRY = "transition_to_dry"
    # Monomodal (North)
    DRY_SEASON = "dry_season"
    EARLY_RAINY = "early_rainy"
    RAINY_SEASON = "rainy_season"
    LATE_RAINY = "late_rainy"
    # Generic
    OFF_SEASON = "off_season"


class GrowthStage(str, Enum):
    """Growth stages"""
    GERMINATION = "germination"
    VEGETATIVE = "vegetative"
    FLOWERING = "flowering"
    GRAIN_FILLING = "grain_filling"
    MATURITY = "maturity"
    HARVEST = "harvest"


class CultivationSystem(str, Enum):
    """Cultivation systems"""
    OPEN_FIELD = "open_field"
    GREENHOUSE = "greenhouse"
    SHADE_HOUSE = "shade_house"
    HYDROPONIC = "hydroponic"
    IRRIGATED = "irrigated"
    RAINFED = "rainfed"
    AGROFORESTRY = "agroforestry"


class ManagementIntensity(str, Enum):
    """Management intensity levels"""
    LOW_INPUT = "low_input"
    MEDIUM_INPUT = "medium_input"
    HIGH_INPUT = "high_input"
    ORGANIC = "organic"
    INTEGRATED = "integrated"


class MarketDestination(str, Enum):
    """Market destination"""
    LOCAL_MARKET = "local_market"
    REGIONAL_MARKET = "regional_market"
    NATIONAL_MARKET = "national_market"
    EXPORT = "export"
    PROCESSING = "processing"
    SUBSISTENCE = "subsistence"


class DiseaseType(str, Enum):
    """Disease type classification"""
    FUNGAL = "fungal"
    BACTERIAL = "bacterial"
    VIRAL = "viral"
    NEMATODE = "nematode"
    PARASITIC_PLANT = "parasitic_plant"


class CropHealthModel(BaseModel):
    """Crop health observation model"""
    disease_name: str = Field(..., description="Disease name")
    disease_type: DiseaseType = Field(..., description="Disease type")
    pathogen: Optional[str] = Field(None, description="Pathogen name")
    incidence_percentage: float = Field(..., ge=0, le=100, description="Incidence percentage")
    severity: Optional[str] = Field(None, description="Severity level (low/moderate/high/severe)")
    affected_plant_part: Optional[str] = Field(None, description="Affected plant part")
    treatment_applied: Optional[str] = Field(None, description="Treatment applied")


class IntercroppingModel(BaseModel):
    """Intercropping system model"""
    primary_crop: str = Field(..., description="Primary crop type")
    companion_crops: List[str] = Field(..., description="Companion crop types")
    planting_pattern: Optional[str] = Field(None, description="Planting pattern (row, strip, mixed)")
    spatial_arrangement: Optional[str] = Field(None, description="Spatial arrangement description")
    land_equivalent_ratio: Optional[float] = Field(None, ge=0.5, le=3.0, description="Land equivalent ratio")


# Typical harvest index by crop type (agronomic reference values)
HARVEST_INDEX_BY_CROP: Dict[str, tuple[float, float]] = {
    "maize": (0.45, 0.55),
    "rice": (0.40, 0.55),
    "sorghum": (0.25, 0.40),
    "millet": (0.20, 0.35),
    "cowpea": (0.25, 0.40),
    "groundnut": (0.30, 0.45),
    "common_bean": (0.35, 0.50),
    "soybean": (0.35, 0.50),
    "cassava": (0.50, 0.70),
    "sweet_potato": (0.50, 0.65),
    "yam": (0.45, 0.60),
    "potato": (0.70, 0.85),
    "tomato": (0.55, 0.70),
    "pepper": (0.40, 0.55),
    "onion": (0.65, 0.80),
    "cabbage": (0.50, 0.70),
    "carrot": (0.55, 0.75),
    "lettuce": (0.60, 0.80),
    "cucumber": (0.55, 0.70),
    "okra": (0.30, 0.45),
    "plantain_banana": (0.30, 0.45),
    "cocoa": (0.20, 0.30),
    "coffee": (0.25, 0.35),
    "oil_palm": (0.20, 0.30),
    "mango": (0.30, 0.45),
    "pineapple": (0.40, 0.55),
    "cotton": (0.30, 0.45),
}


class VarietyModel(BaseModel):
    """Crop variety model"""
    variety_name: str = Field(..., description="Variety name")
    variety_type: Optional[str] = Field(None, description="Variety type (improved/local/hybrid)")
    origin: Optional[str] = Field(None, description="Variety origin")

    # Agronomic characteristics
    maturity_days: Optional[int] = Field(None, ge=30, le=600, description="Maturity duration (days)")
    yield_potential_min: Optional[float] = Field(None, ge=0, description="Min yield potential (kg/ha)")
    yield_potential_max: Optional[float] = Field(None, ge=0, description="Max yield potential (kg/ha)")

    # Stress tolerances
    drought_tolerance: Optional[str] = Field(None, description="Drought tolerance")
    heat_tolerance: Optional[str] = Field(None, description="Heat tolerance")
    disease_resistance: Optional[str] = Field(None, description="Disease resistance")
    pest_resistance: Optional[str] = Field(None, description="Pest resistance")

    # Nutritional qualities
    protein_content: Optional[float] = Field(None, ge=0, le=50, description="Protein content (%)")
    oil_content: Optional[float] = Field(None, ge=0, le=60, description="Oil content (%)")
    starch_content: Optional[float] = Field(None, ge=0, le=90, description="Starch content (%)")

    # Regional adaptations
    adapted_zones: Optional[List[str]] = Field(None, description="Adaptation zones")
    altitude_range: Optional[List[float]] = Field(None, description="Altitude range [min, max]")

    @field_validator('yield_potential_max')
    @classmethod
    def validate_yield_range(cls, v: Optional[float], info) -> Optional[float]:
        """Validates that max yield > min yield"""
        y_min = info.data.get('yield_potential_min')
        if y_min is not None and v is not None and v <= y_min:
            raise ValueError("Max yield must be > min yield")
        return v


class PlantingDetailsModel(BaseModel):
    """Planting details model"""
    planting_date: Optional[date] = Field(None, description="Planting date")
    planting_method: Optional[str] = Field(None, description="Planting method")
    seed_rate: Optional[float] = Field(None, ge=0, description="Seed rate (kg/ha)")
    planting_density: Optional[int] = Field(None, ge=50, le=200000, description="Planting density (plants/ha)")

    # Spacing
    row_spacing_cm: Optional[float] = Field(None, ge=10, le=300, description="Row spacing (cm)")
    plant_spacing_cm: Optional[float] = Field(None, ge=5, le=200, description="Plant spacing (cm)")

    # Depth and preparation
    planting_depth_cm: Optional[float] = Field(None, ge=0.5, le=15, description="Sowing depth (cm)")
    land_preparation: Optional[str] = Field(None, description="Land preparation method")

    # Support
    plant_support: Optional[str] = Field(None, description="Plant support type")
    mulching: Optional[bool] = Field(None, description="Mulching applied")


class FertilizationModel(BaseModel):
    """Fertilization model"""
    # Mineral fertilizers (kg/ha)
    nitrogen_kg_ha: Optional[float] = Field(None, ge=0, le=500, description="Nitrogen applied (kg/ha)")
    phosphorus_kg_ha: Optional[float] = Field(None, ge=0, le=200, description="Phosphorus applied (kg/ha)")
    potassium_kg_ha: Optional[float] = Field(None, ge=0, le=300, description="Potassium applied (kg/ha)")

    # Organic fertilizers
    organic_fertilizer_tha: Optional[float] = Field(None, ge=0, le=50, description="Organic fertilizer (t/ha)")
    compost_tha: Optional[float] = Field(None, ge=0, le=30, description="Compost applied (t/ha)")

    # Application details
    fertilizer_type: Optional[str] = Field(None, description="Main fertilizer type (NPK, urea, etc.)")
    application_method: Optional[str] = Field(None, description="Application method (broadcast, banding, etc.)")
    split_applications: Optional[int] = Field(None, ge=1, le=5, description="Number of split applications")


class IrrigationModel(BaseModel):
    """Irrigation model"""
    irrigated: bool = Field(False, description="Whether field is irrigated")
    irrigation_method: Optional[str] = Field(None, description="Irrigation method (drip, furrow, sprinkler)")
    water_source: Optional[str] = Field(None, description="Water source (river, borehole, rain harvesting)")
    irrigation_frequency: Optional[str] = Field(None, description="Irrigation frequency")
    water_applied_mm: Optional[float] = Field(None, ge=0, le=2000, description="Total water applied (mm)")


class YieldModel(BaseModel):
    """Yield and production model"""
    yield_tha: float = Field(..., ge=0, le=100, description="Yield (t/ha)")
    biomass_tha: Optional[float] = Field(None, ge=0, le=200, description="Total biomass (t/ha)")
    harvest_index: Optional[float] = Field(None, ge=0.05, le=0.95, description="Harvest index")

    @model_validator(mode='after')
    def validate_yield_biomass(self) -> 'YieldModel':
        """Validates yield <= biomass when both are provided"""
        if self.biomass_tha is not None and self.yield_tha > self.biomass_tha:
            raise ValueError("Yield cannot exceed total biomass")
        return self


class CropDataModel(BaseModel):
    """Main crop data model"""
    # Identifiers
    id: Optional[str] = Field(None, alias="_id")
    field_id: str = Field(..., description="Unique field identifier")

    # Crop information
    crop_type: CropType = Field(..., description="Crop type")
    crop_group: Optional[CropGroup] = Field(None, description="Crop group")
    variety: Optional[VarietyModel] = Field(None, description="Variety details")
    season: Season = Field(..., description="Agricultural season")
    year: int = Field(..., ge=2000, le=2100, description="Crop year")

    # Location
    latitude: float = Field(..., ge=1.6, le=13.1, description="Latitude WGS84")
    longitude: float = Field(..., ge=8.3, le=16.2, description="Longitude WGS84")
    elevation: Optional[float] = Field(None, ge=0, le=4095, description="Elevation (m)")
    agroecological_zone: Optional[str] = Field(None, description="Agroecological zone")
    area_hectares: Optional[float] = Field(None, ge=0.01, le=10000, description="Field area (ha)")

    # Cultivation details
    cultivation_system: Optional[CultivationSystem] = Field(None, description="Cultivation system")
    management_intensity: Optional[ManagementIntensity] = Field(None, description="Management intensity")
    planting_details: Optional[PlantingDetailsModel] = Field(None, description="Planting details")
    fertilization: Optional[FertilizationModel] = Field(None, description="Fertilization details")
    irrigation: Optional[IrrigationModel] = Field(None, description="Irrigation details")

    # Growth and yield
    growth_stage: Optional[GrowthStage] = Field(None, description="Current growth stage")
    cycle_length_days: Optional[int] = Field(None, ge=30, le=600, description="Crop cycle length (days)")
    harvest_date: Optional[date] = Field(None, description="Harvest date")
    yield_data: Optional[YieldModel] = Field(None, description="Yield data")

    # Crop health
    pest_pressure: Optional[str] = Field(None, description="Pest pressure level")
    disease_incidence: Optional[float] = Field(None, ge=0, le=100, description="DEPRECATED: Use crop_health instead. Disease incidence (%)")
    crop_health: Optional[List[CropHealthModel]] = Field(None, description="Structured crop health observations")

    # Intercropping
    intercropping: Optional[IntercroppingModel] = Field(None, description="Intercropping system details")

    # Market
    market_destination: Optional[MarketDestination] = Field(None, description="Market destination")

    # System metadata
    created_at: Optional[datetime] = Field(None, description="Creation date")
    updated_at: Optional[datetime] = Field(None, description="Update date")
    version: Optional[int] = Field(1, description="Document version")
    data_source: Optional[str] = Field(None, description="Data source")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class CropSchema:
    """MongoDB schema for crop data"""

    @staticmethod
    def get_collection_name() -> str:
        return "agri_crop_data"

    @staticmethod
    def get_indexes() -> List[Dict[str, Any]]:
        return [
            {"coordinates.coordinates": "2dsphere"},
            {"latitude": 1, "longitude": 1},
            {"created_at": -1},
            {"field_id": 1},
            {"crop_type": 1},
            {"season": 1},
            {"year": -1},
            {"agroecological_zone": 1},
            {"crop_type": 1, "season": 1, "year": -1},
            {"yield_data.yield_tha": 1},
            {"cultivation_system": 1},
        ]

    @staticmethod
    def get_validation_schema() -> Dict[str, Any]:
        return {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["field_id", "crop_type", "season", "year", "latitude", "longitude"],
                "properties": {
                    "field_id": {
                        "bsonType": "string",
                        "description": "Unique field identifier required",
                    },
                    "latitude": {
                        "bsonType": "double",
                        "minimum": 1.6,
                        "maximum": 13.1,
                    },
                    "longitude": {
                        "bsonType": "double",
                        "minimum": 8.3,
                        "maximum": 16.2,
                    },
                    "year": {
                        "bsonType": "int",
                        "minimum": 2000,
                        "maximum": 2100,
                    },
                },
            }
        }

    @staticmethod
    def get_sample_document() -> Dict[str, Any]:
        return {
            "field_id": "CMR_FIELD_001",
            "crop_type": "maize",
            "crop_group": "cereals",
            "variety": {
                "variety_name": "CMS_8704",
                "variety_type": "improved",
                "maturity_days": 120,
                "yield_potential_min": 1000,
                "yield_potential_max": 8000,
                "drought_tolerance": "moderate",
            },
            "season": "first_rainy_season",
            "year": 2024,
            "latitude": 3.8667,
            "longitude": 11.5167,
            "elevation": 650,
            "agroecological_zone": "humid_forest_inland",
            "area_hectares": 2.5,
            "cultivation_system": "open_field",
            "management_intensity": "medium_input",
            "fertilization": {
                "nitrogen_kg_ha": 120,
                "phosphorus_kg_ha": 40,
                "potassium_kg_ha": 40,
                "organic_fertilizer_tha": 5.0,
                "split_applications": 2,
            },
            "irrigation": {"irrigated": False},
            "cycle_length_days": 120,
            "yield_data": {
                "yield_tha": 3.5,
                "biomass_tha": 7.0,
                "harvest_index": 0.50,
            },
            "pest_pressure": "moderate",
            "disease_incidence": 12.5,
            "crop_health": [
                {
                    "disease_name": "Maize Streak Virus",
                    "disease_type": "viral",
                    "pathogen": "Maize streak virus (MSV)",
                    "incidence_percentage": 12.5,
                    "severity": "moderate",
                    "affected_plant_part": "leaves",
                    "treatment_applied": "resistant_variety",
                },
            ],
            "intercropping": {
                "primary_crop": "maize",
                "companion_crops": ["groundnut"],
                "planting_pattern": "row",
                "spatial_arrangement": "alternating rows 1:1",
                "land_equivalent_ratio": 1.35,
            },
            "data_source": "IRAD_field_survey",
            "created_at": datetime.now(),
            "version": 1,
        }
