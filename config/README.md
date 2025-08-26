# Configuration

This directory contains the configuration management system for the PDF Data Extraction project.

## Overview

The configuration system uses Pydantic settings to provide type-safe, validated configuration with environment variable support.

## Files

- `settings.py` - Main configuration classes and settings
- `__init__.py` - Package initialization and exports

## Usage

### Basic Usage

```python
from config import settings

# Access settings
print(settings.api.port)  # 8000
print(settings.data.data_dir)  # Path('data')
print(settings.processing.enable_ocr)  # True
```

### Environment Variables

You can override any setting using environment variables. The naming convention uses double underscores (`__`) to separate nested settings:

```bash
# Override API port
export API__PORT=9000

# Override data directory
export DATA__DATA_DIR=/custom/data/path

# Override OCR settings
export PROCESSING__ENABLE_OCR=false
export PROCESSING__OCR_LANGUAGE=spa
```

### Environment File

Create a `.env` file in your project root to set environment-specific values:

```bash
# .env
ENVIRONMENT=production
API__DEBUG=false
LOGGING__LOG_LEVEL=WARNING
SECURITY__ENABLE_AUTH=true
```

## Configuration Sections

### DataSettings
- File processing directories
- Batch processing settings
- Data retention policies

### APISettings
- Server configuration
- CORS settings
- Rate limiting
- API versioning

### LoggingSettings
- Log levels and formats
- File and console logging
- Structured logging options

### ProcessingSettings
- OCR configuration
- Image processing settings
- Table and text extraction options

### DatabaseSettings
- Database type and connection
- SQLite and PostgreSQL support
- Connection pooling

### SecuritySettings
- Authentication configuration
- API key requirements
- Rate limiting settings

### MonitoringSettings
- Metrics collection
- Health checks
- Performance monitoring

## Environment-Specific Overrides

The system automatically adjusts settings based on the environment:

- **Development**: Debug enabled, verbose logging
- **Staging**: Info-level logging, debug disabled
- **Production**: Warning-level logging, auth enabled, debug disabled

## Validation

All settings are validated using Pydantic:
- Type checking
- Value validation
- Automatic directory creation for paths
- Environment variable parsing

## Example Configuration

```python
from config import Settings

# Create custom settings instance
custom_settings = Settings(
    environment="production",
    api__port=9000,
    data__max_file_size_mb=500
)

# Use in your application
app_port = custom_settings.api.port
max_file_size = custom_settings.data.max_file_size_mb
```
