# Agri-Harvest: Cameroon Agricultural Data Management System

A comprehensive platform for processing and analyzing agricultural data from Cameroon, including satellite imagery, soil data, weather information, and machine learning models for agricultural insights.

## Features

- **Multi-source Data Integration**: Combine satellite, soil, weather, and crop data
- **MongoDB Database**: Scalable NoSQL database with geospatial indexing
- **Machine Learning Ready**: Built-in support for ML model training and inference
- **Geospatial Analysis**: Advanced spatial analysis tools for agricultural zones
- **API Framework**: FastAPI-based REST API for data access
- **Data Quality Control**: Automated validation and quality assessment
- **IRAD Integration**: Support for Cameroon's agricultural research centers

## Project Structure

```
agri-harvest/
├── config/                 # Configuration files
│   ├── settings.py         # Main configuration settings
│   ├── database.py         # MongoDB connection management
│   └── schema/             # Data schemas and validation
├── src/agri_harvest/       # Main application package
│   ├── api/               # FastAPI application
│   ├── common/            # Common utilities and logging
│   └── database.py        # Database utilities
├── utils/                  # Utility functions
│   ├── constants.py       # Application constants
│   ├── date_utils.py      # Date/time utilities
│   ├── file_utils.py      # File handling utilities
│   └── geospatial_utils.py # Geospatial analysis tools
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks for analysis
└── docs/                   # Documentation

```

## Installation

### Prerequisites
- Python 3.10 or higher
- MongoDB 4.4 or higher
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd agri-harvest
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env  # Create environment file
   # Edit .env with your MongoDB connection details
   ```

5. **Initialize database**:
   ```bash
   python main.py  # Will create indexes and verify connection
   ```

## Configuration

The application uses environment variables for configuration. Key settings include:

- `MONGODB_URL`: MongoDB connection string
- `MONGODB_DATABASE`: Database name
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `ENVIRONMENT`: Runtime environment (development, production)

See `config/settings.py` for all available configuration options.

## Usage

### Running the Application
```bash
python main.py
```

### Starting the API Server
```bash
cd src/agri_harvest/api
uvicorn main:app --reload
```

### Running Tests
```bash
pytest tests/
```

### Data Processing Examples
```python
from config.database import get_database
from utils.geospatial_utils import validate_coordinates

# Connect to database
db = await get_database()

# Validate coordinates for Cameroon
is_valid, message = validate_coordinates(3.8667, 11.5167)
```

## Data Models

The system supports several data types:

- **Soil Data**: Physical and chemical soil properties
- **Weather Data**: Meteorological observations and derived indices
- **Crop Data**: Cultivation information and yield data
- **Satellite Data**: Remote sensing imagery and derived products

All data models include geospatial indexing and quality control metadata.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Submit a pull request

## Development Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive tests
- Update documentation for new features
- Use English for all code comments and documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the development team or create an issue in the repository.

