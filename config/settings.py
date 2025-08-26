from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic import BaseSettings, Field, validator


class DataSettings(BaseSettings):
    """Settings for data processing and storage."""
    
    # Data directories
    data_dir: Path = Field(default="data", description="Directory containing PDF files")
    output_dir: Path = Field(default="output", description="Directory for extracted data output")
    temp_dir: Path = Field(default="temp", description="Directory for temporary files")
    
    # File processing settings
    max_file_size_mb: int = Field(default=100, description="Maximum PDF file size in MB")
    supported_formats: List[str] = Field(default=["pdf"], description="Supported file formats")
    batch_size: int = Field(default=10, description="Number of files to process in a batch")
    
    # Data retention
    keep_temp_files: bool = Field(default=False, description="Whether to keep temporary files after processing")
    max_output_files: int = Field(default=1000, description="Maximum number of output files to keep")
    
    @validator('data_dir', 'output_dir', 'temp_dir', pre=True)
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path


class APISettings(BaseSettings):
    """Settings for API configuration."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="Allowed CORS methods")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute per IP")
    
    # API versioning
    api_version: str = Field(default="v1", description="API version")
    api_prefix: str = Field(default="/api", description="API endpoint prefix")


class LoggingSettings(BaseSettings):
    """Settings for logging configuration."""
    
    # Log level
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # Log files
    log_file: Optional[Path] = Field(default=None, description="Log file path")
    log_file_level: str = Field(default="DEBUG", description="Log file level")
    max_log_size_mb: int = Field(default=10, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, description="Number of log file backups")
    
    # Console logging
    console_logging: bool = Field(default=True, description="Enable console logging")
    console_log_level: str = Field(default="INFO", description="Console log level")
    
    # Structured logging
    json_logging: bool = Field(default=False, description="Enable JSON formatted logging")


class ProcessingSettings(BaseSettings):
    """Settings for PDF processing."""
    
    # OCR settings
    enable_ocr: bool = Field(default=True, description="Enable OCR processing")
    ocr_language: str = Field(default="eng", description="OCR language")
    ocr_confidence_threshold: float = Field(default=0.7, description="Minimum OCR confidence threshold")
    
    # Image processing
    image_dpi: int = Field(default=300, description="Image DPI for processing")
    image_format: str = Field(default="PNG", description="Image output format")
    max_image_dimension: int = Field(default=4000, description="Maximum image dimension")
    
    # Table extraction
    enable_table_extraction: bool = Field(default=True, description="Enable table extraction")
    table_min_accuracy: float = Field(default=0.8, description="Minimum table extraction accuracy")
    
    # Text processing
    enable_text_extraction: bool = Field(default=True, description="Enable text extraction")
    text_encoding: str = Field(default="utf-8", description="Text encoding")
    preserve_formatting: bool = Field(default=True, description="Preserve text formatting")


class DatabaseSettings(BaseSettings):
    """Settings for database configuration."""
    
    # Database type
    database_type: str = Field(default="sqlite", description="Database type (sqlite, postgresql, mysql)")
    
    # SQLite settings
    sqlite_path: Optional[Path] = Field(default="data/extraction.db", description="SQLite database path")
    
    # PostgreSQL settings
    postgres_host: Optional[str] = Field(default=None, description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: Optional[str] = Field(default=None, description="PostgreSQL username")
    postgres_password: Optional[str] = Field(default=None, description="PostgreSQL password")
    postgres_database: Optional[str] = Field(default=None, description="PostgreSQL database name")
    
    # Connection settings
    connection_timeout: int = Field(default=30, description="Database connection timeout in seconds")
    pool_size: int = Field(default=10, description="Database connection pool size")
    
    @validator('sqlite_path', pre=True)
    def create_sqlite_directory(cls, v):
        """Create SQLite database directory if it doesn't exist."""
        if v:
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return v


class SecuritySettings(BaseSettings):
    """Settings for security configuration."""
    
    # Authentication
    enable_auth: bool = Field(default=False, description="Enable authentication")
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for JWT tokens")
    token_expire_minutes: int = Field(default=30, description="JWT token expiration time in minutes")
    
    # API keys
    require_api_key: bool = Field(default=False, description="Require API key for requests")
    api_key_header: str = Field(default="X-API-Key", description="Header name for API key")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    max_requests_per_minute: int = Field(default=60, description="Maximum requests per minute per IP")


class MonitoringSettings(BaseSettings):
    """Settings for monitoring and metrics."""
    
    # Metrics collection
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    
    # Health checks
    enable_health_checks: bool = Field(default=True, description="Enable health check endpoints")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    slow_query_threshold_ms: int = Field(default=1000, description="Slow query threshold in milliseconds")


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    
    # Configuration sections
    data: DataSettings = Field(default_factory=DataSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override settings based on environment
        if self.environment == "production":
            self.logging.log_level = "WARNING"
            self.api.debug = False
            self.security.enable_auth = True
        elif self.environment == "staging":
            self.logging.log_level = "INFO"
            self.api.debug = False
        else:  # development
            self.logging.log_level = "DEBUG"
            self.api.debug = True


# Global settings instance
settings = Settings()

# Convenience imports for easy access
__all__ = [
    "Settings",
    "DataSettings",
    "APISettings", 
    "LoggingSettings",
    "ProcessingSettings",
    "DatabaseSettings",
    "SecuritySettings",
    "MonitoringSettings",
    "settings"
]
