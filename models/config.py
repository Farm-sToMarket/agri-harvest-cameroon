"""Feature groups, leakage guard, and model configuration."""

from dataclasses import dataclass, field

TARGET = "yield_kg_ha"

LEAKAGE_FEATURES = [
    "nue",
    "wue",
    "yield_gap_ratio",
    "harvest_index",
    "biomass_kg_ha",
]

DROP_COLUMNS = [
    "observation_date",
    "data_source",
    "agroecological_zone",
    "season",
    "crop_name",
    "crop_group",
]

CONTINUOUS_FEATURES = [
    "latitude",
    "longitude",
    "elevation",
    "temperature_min",
    "temperature_max",
    "temperature_mean",
    "precipitation_daily",
    "relative_humidity",
    "solar_radiation",
    "ph_water",
    "organic_carbon",
    "total_nitrogen",
    "available_phosphorus",
    "sand_percentage",
    "clay_percentage",
    "fertilizer_nitrogen",
    "fertilizer_phosphorus",
    "organic_fertilizer",
    "disease_incidence",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "gdd_base10",
    "gdd_base15",
    "diurnal_range",
    "aridity_index",
    "vpd_kpa",
    "temp_altitude_residual",
    "soil_fertility_index",
    "cn_ratio",
    "cec_estimate",
    "input_intensity",
    "total_mineral_fert",
    "organic_mineral_ratio",
    "humidity_temp_interaction",
    "rain_oc_interaction",
    "input_soil_interaction",
    "water_supply_index",
    "disease_risk_score",
]

BINARY_FEATURES = [
    "irrigation_applied",
    "heavy_rain_day",
    "is_rainy_season",
    "data_quality_flag",
    "rainfall_regime",
    "zone_coastal_forest",
    "zone_forest_savanna",
    "zone_guinea_savanna",
    "zone_highlands",
    "zone_mont_cameroun_volcanic",
    "zone_sahel_savanna",
    "grp_cereals",
    "grp_industrial_crops",
    "grp_legumes",
    "grp_root_tubers",
    "grp_tree_crops",
    "grp_vegetables",
    "alt_highland",
    "alt_lowland",
    "alt_mid_altitude",
    "alt_mountain",
    "alt_plateau",
]

ORDINAL_FEATURES = ["month", "day_of_year", "year", "season_ordinal"]


@dataclass
class ModelConfig:
    """Configuration for model training."""

    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
