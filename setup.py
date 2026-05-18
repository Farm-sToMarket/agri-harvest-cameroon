from setuptools import setup, find_packages

setup(
    name="agri-harvest",
    version="2.0.0",
    description="Yield prediction platform for Cameroon agriculture",
    author="SYNTHI-AI",
    author_email="contact@synthi-ai.com",
    url="https://github.com/farmstomarket/agri-harvest-cameroon",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.12",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.11.9",
        "pydantic-settings>=2.3.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.3.2",
        "datasets>=2.0.0",
        "huggingface-hub>=0.20.0",
    ],
    extras_require={
        "ml": [
            "scikit-learn>=1.3.0",
            "polars>=0.20.0",
            "lightgbm>=4.0.0",
            "xgboost>=2.0.0",
            "torch>=2.0.0",
            "optuna>=3.3.0",
            "shap>=0.43.0",
            "joblib>=1.3.0",
            "matplotlib>=3.7.0",
        ],
        "geo": [
            "geopandas>=0.14.0",
            "rasterio>=1.3.0",
            "geopy>=2.3.0",
        ],
        "climate": [
            "xarray>=2024.1.0",
            "netCDF4>=1.6.0",
            "dask[array]>=2024.1.0",
            "pyarrow>=14.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
