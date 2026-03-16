"""
Tests for Pydantic schema validators
"""

import pytest
from datetime import date, datetime
from pydantic import ValidationError

from config.schema.soil_schema import (
    TextureModel, TextureClass, ChemicalPropertiesModel,
    WaterPropertiesModel, PhysicalPropertiesModel,
    CoordinatesModel,
)
from config.schema.weather_schema import (
    TemperatureModel, PressureModel, WeatherStationModel,
    WeatherDataModel, StationType, RainfallRegime,
)
from utils.date_utils import get_agricultural_season
from config.schema.crop_schema import (
    CropDataModel, CropType, Season, YieldModel, VarietyModel,
    HARVEST_INDEX_BY_CROP, IntercroppingModel, CropHealthModel,
    DiseaseType, PlantingDetailsModel,
)


class TestTextureModel:
    def test_valid_texture(self):
        t = TextureModel(
            sand_percentage=45.0,
            silt_percentage=33.0,
            clay_percentage=22.0,
            texture_class=TextureClass.LOAM,
        )
        assert t.sand_percentage == 45.0

    def test_sum_not_100_raises(self):
        with pytest.raises(ValidationError, match="100%"):
            TextureModel(
                sand_percentage=50.0,
                silt_percentage=50.0,
                clay_percentage=50.0,
                texture_class=TextureClass.CLAY,
            )

    def test_sum_within_tolerance(self):
        """1% tolerance should pass"""
        t = TextureModel(
            sand_percentage=45.0,
            silt_percentage=33.5,
            clay_percentage=22.0,
            texture_class=TextureClass.LOAM,
        )
        assert t is not None


class TestChemicalPropertiesModel:
    def test_organic_matter_van_bemmelen(self):
        """Organic matter = organic_carbon * 1.724"""
        chem = ChemicalPropertiesModel(
            ph_water=6.2,
            organic_carbon=2.0,
        )
        assert abs(chem.organic_matter - 2.0 * 1.724) < 0.01

    def test_cn_ratio_calculated(self):
        chem = ChemicalPropertiesModel(
            ph_water=6.2,
            organic_carbon=2.0,
            total_nitrogen=0.2,
        )
        assert abs(chem.c_n_ratio - 10.0) < 0.01

    def test_cn_ratio_not_calculated_without_nitrogen(self):
        chem = ChemicalPropertiesModel(
            ph_water=6.2,
            organic_carbon=2.0,
        )
        assert chem.c_n_ratio is None

    def test_ph_too_low(self):
        with pytest.raises(ValidationError):
            ChemicalPropertiesModel(ph_water=2.0, organic_carbon=2.0)

    def test_ph_too_high(self):
        with pytest.raises(ValidationError):
            ChemicalPropertiesModel(ph_water=10.0, organic_carbon=2.0)


class TestWaterPropertiesModel:
    def test_awc_calculated(self):
        wp = WaterPropertiesModel(field_capacity=0.32, wilting_point=0.18)
        assert abs(wp.available_water_capacity - 0.14) < 0.01

    def test_awc_provided(self):
        wp = WaterPropertiesModel(
            field_capacity=0.32,
            wilting_point=0.18,
            available_water_capacity=0.15,
        )
        assert wp.available_water_capacity == 0.15


class TestPhysicalPropertiesModel:
    def test_porosity_calculated(self):
        """Porosity = (1 - bulk_density / 2.65) * 100"""
        pp = PhysicalPropertiesModel(bulk_density=1.35)
        expected = (1 - 1.35 / 2.65) * 100
        assert abs(pp.porosity - expected) < 0.1

    def test_porosity_provided(self):
        pp = PhysicalPropertiesModel(bulk_density=1.35, porosity=50.0)
        assert pp.porosity == 50.0

    def test_bulk_density_too_low(self):
        with pytest.raises(ValidationError):
            PhysicalPropertiesModel(bulk_density=0.5)

    def test_bulk_density_too_high(self):
        with pytest.raises(ValidationError):
            PhysicalPropertiesModel(bulk_density=2.5)


class TestCoordinatesModel:
    def test_valid_cameroon_coordinates(self):
        c = CoordinatesModel(latitude=3.87, longitude=11.52, elevation=650)
        assert c.latitude == 3.87

    def test_latitude_out_of_bounds(self):
        with pytest.raises(ValidationError):
            CoordinatesModel(latitude=0.0, longitude=11.52, elevation=650)

    def test_longitude_out_of_bounds(self):
        with pytest.raises(ValidationError):
            CoordinatesModel(latitude=3.87, longitude=5.0, elevation=650)

    def test_elevation_negative(self):
        with pytest.raises(ValidationError):
            CoordinatesModel(latitude=3.87, longitude=11.52, elevation=-10)


class TestTemperatureModel:
    def test_avg_calculated(self):
        t = TemperatureModel(temperature_min=20.0, temperature_max=30.0)
        assert t.temperature_avg == 25.0

    def test_range_calculated(self):
        t = TemperatureModel(temperature_min=20.0, temperature_max=30.0)
        assert t.temperature_range == 10.0

    def test_avg_not_overwritten(self):
        t = TemperatureModel(
            temperature_min=20.0, temperature_max=30.0, temperature_avg=26.0
        )
        assert t.temperature_avg == 26.0

    def test_min_too_low(self):
        with pytest.raises(ValidationError):
            TemperatureModel(temperature_min=-10.0)


class TestPressureModel:
    def test_highland_pressure_accepted(self):
        """Pressure at ~4000m elevation should be ~600 hPa"""
        p = PressureModel(atmospheric_pressure=620.0)
        assert p.atmospheric_pressure == 620.0

    def test_sea_level_pressure(self):
        p = PressureModel(atmospheric_pressure=1013.25)
        assert p.atmospheric_pressure == 1013.25

    def test_pressure_too_low(self):
        with pytest.raises(ValidationError):
            PressureModel(atmospheric_pressure=500.0)

    def test_pressure_too_high(self):
        with pytest.raises(ValidationError):
            PressureModel(atmospheric_pressure=1100.0)


class TestWeatherStationModel:
    def test_rainfall_regime_inferred_south(self):
        ws = WeatherStationModel(
            station_id="S001",
            station_name="Yaounde",
            station_type=StationType.AUTOMATIC,
            latitude=3.87,
            longitude=11.52,
            elevation=650,
        )
        assert ws.rainfall_regime == RainfallRegime.BIMODAL

    def test_rainfall_regime_inferred_north(self):
        ws = WeatherStationModel(
            station_id="S002",
            station_name="Maroua",
            station_type=StationType.AUTOMATIC,
            latitude=10.58,
            longitude=14.32,
            elevation=420,
        )
        assert ws.rainfall_regime == RainfallRegime.MONOMODAL

    def test_rainfall_regime_explicit(self):
        ws = WeatherStationModel(
            station_id="S003",
            station_name="Custom",
            station_type=StationType.MANUAL,
            latitude=6.0,
            longitude=10.0,
            elevation=500,
            rainfall_regime=RainfallRegime.BIMODAL,
        )
        assert ws.rainfall_regime == RainfallRegime.BIMODAL


class TestGetAgriculturalSeason:
    def test_south_grand_dry_season(self):
        assert get_agricultural_season(date(2024, 1, 15)) == "grand_dry_season"

    def test_south_first_rainy(self):
        assert get_agricultural_season(date(2024, 4, 15)) == "first_rainy_season"

    def test_south_petit_dry(self):
        assert get_agricultural_season(date(2024, 7, 15)) == "petit_dry_season"

    def test_south_second_rainy(self):
        assert get_agricultural_season(date(2024, 10, 15)) == "second_rainy_season"

    def test_south_transition_to_dry(self):
        assert get_agricultural_season(date(2024, 11, 15)) == "transition_to_dry"

    def test_north_dry(self):
        assert get_agricultural_season(date(2024, 1, 15), latitude=10.0) == "dry_season"

    def test_north_rainy(self):
        assert get_agricultural_season(date(2024, 7, 15), latitude=10.0) == "rainy_season"

    def test_north_early_rainy(self):
        assert get_agricultural_season(date(2024, 4, 15), latitude=10.0) == "early_rainy"

    def test_north_late_rainy(self):
        assert get_agricultural_season(date(2024, 10, 15), latitude=10.0) == "late_rainy"


class TestWeatherDataModel:
    def test_day_of_year_calculated(self):
        wd = WeatherDataModel(
            station_id="S001",
            date=date(2024, 2, 14),
        )
        assert wd.day_of_year == 45

    def test_season_calculated_south(self):
        wd = WeatherDataModel(
            station_id="S001",
            date=date(2024, 1, 15),
            latitude=3.87,
        )
        assert wd.season == "grand_dry_season"
        assert wd.rainfall_regime == RainfallRegime.BIMODAL

    def test_season_calculated_north(self):
        wd = WeatherDataModel(
            station_id="S001",
            date=date(2024, 7, 15),
            latitude=10.5,
        )
        assert wd.season == "rainy_season"
        assert wd.rainfall_regime == RainfallRegime.MONOMODAL


class TestYieldModel:
    def test_valid_yield(self):
        y = YieldModel(yield_tha=3.5, biomass_tha=7.0, harvest_index=0.50)
        assert y.yield_tha == 3.5

    def test_yield_exceeds_biomass(self):
        with pytest.raises(ValidationError, match="biomass"):
            YieldModel(yield_tha=10.0, biomass_tha=5.0)

    def test_yield_without_biomass(self):
        y = YieldModel(yield_tha=3.5)
        assert y.biomass_tha is None


class TestVarietyModel:
    def test_valid_variety(self):
        v = VarietyModel(
            variety_name="CMS_8704",
            maturity_days=120,
            yield_potential_min=1000,
            yield_potential_max=8000,
        )
        assert v.variety_name == "CMS_8704"

    def test_max_yield_less_than_min(self):
        with pytest.raises(ValidationError):
            VarietyModel(
                variety_name="BAD",
                yield_potential_min=8000,
                yield_potential_max=1000,
            )


class TestCropDataModel:
    def test_valid_crop(self):
        c = CropDataModel(
            field_id="CMR_FIELD_001",
            crop_type=CropType.MAIZE,
            season=Season.FIRST_RAINY_SEASON,
            year=2024,
            latitude=3.87,
            longitude=11.52,
        )
        assert c.crop_type == CropType.MAIZE

    def test_latitude_out_of_cameroon(self):
        with pytest.raises(ValidationError):
            CropDataModel(
                field_id="CMR_FIELD_001",
                crop_type=CropType.MAIZE,
                season=Season.FIRST_RAINY_SEASON,
                year=2024,
                latitude=0.0,
                longitude=11.52,
            )

    def test_year_bounds(self):
        with pytest.raises(ValidationError):
            CropDataModel(
                field_id="CMR_FIELD_001",
                crop_type=CropType.MAIZE,
                season=Season.FIRST_RAINY_SEASON,
                year=1999,
                latitude=3.87,
                longitude=11.52,
            )


class TestHarvestIndexByCrop:
    def test_all_crop_types_covered(self):
        """All CropType enum values should have a harvest index entry"""
        for crop_type in CropType:
            assert crop_type.value in HARVEST_INDEX_BY_CROP, (
                f"Missing harvest index for {crop_type.value}"
            )

    def test_harvest_index_range_valid(self):
        """All harvest index ranges should have min < max and be in [0, 1]"""
        for crop, (hi_min, hi_max) in HARVEST_INDEX_BY_CROP.items():
            assert 0 < hi_min < hi_max <= 1.0, (
                f"Invalid harvest index range for {crop}: ({hi_min}, {hi_max})"
            )


class TestCottonCropType:
    def test_cotton_in_crop_type(self):
        assert CropType.COTTON.value == "cotton"

    def test_cotton_has_harvest_index(self):
        assert "cotton" in HARVEST_INDEX_BY_CROP

    def test_cotton_crop_data(self):
        c = CropDataModel(
            field_id="CMR_FIELD_COTTON",
            crop_type=CropType.COTTON,
            season=Season.RAINY_SEASON,
            year=2024,
            latitude=10.0,
            longitude=14.0,
        )
        assert c.crop_type == CropType.COTTON


class TestIntercroppingModel:
    def test_valid_intercropping(self):
        ic = IntercroppingModel(
            primary_crop="maize",
            companion_crops=["groundnut", "cowpea"],
            planting_pattern="row",
            land_equivalent_ratio=1.35,
        )
        assert ic.primary_crop == "maize"
        assert len(ic.companion_crops) == 2
        assert ic.land_equivalent_ratio == 1.35

    def test_ler_bounds(self):
        with pytest.raises(ValidationError):
            IntercroppingModel(
                primary_crop="maize",
                companion_crops=["groundnut"],
                land_equivalent_ratio=0.3,
            )

        with pytest.raises(ValidationError):
            IntercroppingModel(
                primary_crop="maize",
                companion_crops=["groundnut"],
                land_equivalent_ratio=3.5,
            )


class TestCropHealthModel:
    def test_valid_crop_health(self):
        ch = CropHealthModel(
            disease_name="Maize Streak Virus",
            disease_type=DiseaseType.VIRAL,
            pathogen="MSV",
            incidence_percentage=15.5,
            severity="moderate",
            affected_plant_part="leaves",
        )
        assert ch.disease_name == "Maize Streak Virus"
        assert ch.disease_type == DiseaseType.VIRAL
        assert ch.incidence_percentage == 15.5

    def test_incidence_bounds(self):
        with pytest.raises(ValidationError):
            CropHealthModel(
                disease_name="Bad",
                disease_type=DiseaseType.FUNGAL,
                incidence_percentage=101,
            )

    def test_disease_types(self):
        assert DiseaseType.FUNGAL.value == "fungal"
        assert DiseaseType.PARASITIC_PLANT.value == "parasitic_plant"


class TestPlantingDensityBounds:
    def test_low_density_accepted(self):
        """Oil palm ~143/ha, mango ~100/ha should be accepted"""
        pd_model = PlantingDetailsModel(planting_density=100)
        assert pd_model.planting_density == 100

    def test_very_low_density_accepted(self):
        pd_model = PlantingDetailsModel(planting_density=50)
        assert pd_model.planting_density == 50

    def test_below_minimum_rejected(self):
        with pytest.raises(ValidationError):
            PlantingDetailsModel(planting_density=49)
