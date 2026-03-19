"""
Soil data schema optimized for ML - Cameroon
Based on IRAD infrastructure and international standards
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator


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
    )


