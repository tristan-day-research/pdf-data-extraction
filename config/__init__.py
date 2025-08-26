"""
Configuration package for PDF Data Extraction project.

This package provides centralized configuration management using Pydantic settings.
"""

from .settings import (
    Settings,
    DataSettings,
    APISettings,
    LoggingSettings,
    ProcessingSettings,
    DatabaseSettings,
    SecuritySettings,
    MonitoringSettings,
    settings,
)

__all__ = [
    "Settings",
    "DataSettings",
    "APISettings",
    "LoggingSettings",
    "ProcessingSettings",
    "DatabaseSettings",
    "SecuritySettings",
    "MonitoringSettings",
    "settings",
]

# Version info
__version__ = "1.0.0"
