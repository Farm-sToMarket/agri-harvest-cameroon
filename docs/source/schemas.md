# Data Schemas

All schemas are Pydantic v2 `BaseModel` subclasses with Cameroon-specific validation
(coordinate bounds, agroecological zones, IRAD taxonomy).

## Crop schema (`config/schema/crop_schema.py`)

### Key models

- **CropDataModel**: Main model with field ID, crop type, season, location, yield, health
- **YieldModel**: Yield (t/ha) with biomass and harvest index validation
- **VarietyModel**: Crop variety with agronomic characteristics and stress tolerances
- **FertilizationModel**: NPK + organic fertilizer application details
- **IntercroppingModel**: Companion crops with land equivalent ratio
- **CropHealthModel**: Disease observations (type, pathogen, incidence, severity)

### Enums

- `CropType`: 26 crop types (maize, rice, cassava, cocoa, cotton, ...)
- `Season`: Bimodal (south) and monomodal (north) agricultural seasons
- `GrowthStage`: germination through harvest
- `CultivationSystem`: open field, greenhouse, agroforestry, etc.
- `ManagementIntensity`: low/medium/high input, organic, integrated

### Constants

- `HARVEST_INDEX_BY_CROP`: Reference harvest index ranges for all 26 crops

## Weather schema (`config/schema/weather_schema.py`)

### Key models

- **WeatherDataModel**: Daily observations with auto-computed `day_of_year`, `season`, `rainfall_regime`
- **WeatherStationModel**: Station metadata with auto-inferred rainfall regime from latitude
- **TemperatureModel**: Min/max/avg with auto-computed range and average
- **PrecipitationModel**: Daily amount, intensity, duration, storm events
- **DerivedIndicesModel**: GDD, ET0, aridity index, SPI, water balance
- **ExtremeEventsModel**: Frost, heat waves, heavy rainfall, dry spells
- **QualityControlModel**: Per-variable quality flags and completeness

### Key validators

- Rainfall regime inferred from latitude (>= 6.0N = monomodal, < 6.0N = bimodal)
- Temperature average and range auto-calculated from min/max
- Agricultural season computed from date and latitude via `utils.date_utils`

## Soil schema (`config/schema/soil_schema.py`)

### Key models

- **SoilDataModel**: Main model with coordinates, texture, chemistry, water properties
- **TextureModel**: Sand/silt/clay percentages with sum-to-100 validation
- **ChemicalPropertiesModel**: pH, organic carbon, NPK, CEC with auto-computed organic matter and C/N ratio
- **WaterPropertiesModel**: Field capacity, wilting point, auto-computed AWC
- **PhysicalPropertiesModel**: Bulk density with auto-computed porosity
- **SoilQualityIndex**: Fertility, health, and productivity indices

### Enums

- `TextureClass`: 12 USDA texture classes
- `DrainageClass`: very poor through excessive
- `SamplingStrategy`: systematic grid, random stratified, targeted, composite

## Geospatial validation

All location fields are validated against Cameroon bounds:
- Latitude: 1.6 - 13.1 (N)
- Longitude: 8.3 - 16.2 (E)
- Elevation: 0 - 4095 m (Mount Cameroon)

These bounds are defined in `utils/constants.py::CAMEROON_BOUNDS` and imported
by `config/settings.py` for the `country_bounds` setting.
