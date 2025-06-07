"""Configuration management for the arXiv MCP server."""

import os
from typing import Optional, List, Literal
from pathlib import Path

from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Log level")
    format: Literal["json", "text", "colored"] = Field(
        default="colored", 
        description="Log format style"
    )
    file_path: Optional[str] = Field(
        default=None, 
        description="Optional log file path"
    )
    max_file_size: int = Field(
        default=10_000_000,  # 10MB
        description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    
    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    model_config = ConfigDict(env_prefix="ARXIV_MCP_LOG_")


class ArxivAPIConfig(BaseSettings):
    """arXiv API configuration settings."""
    
    base_url: str = Field(
        default="http://export.arxiv.org/api/query",
        description="Base URL for arXiv API"
    )
    user_agent: str = Field(
        default="arxiv-mcp/0.1.0",
        description="User agent string for API requests"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay between retries in seconds"
    )
    max_retry_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay between retries in seconds"
    )
    max_results_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum results allowed per search"
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per rate limit window"
    )
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        description="Rate limit window in seconds"
    )
    
    @field_validator('max_retry_delay')
    @classmethod
    def validate_max_retry_delay(cls, v, info):
        if info.data.get('retry_delay') is not None and v < info.data['retry_delay']:
            raise ValueError("max_retry_delay must be >= retry_delay")
        return v
    
    model_config = ConfigDict(env_prefix="ARXIV_MCP_API_")


class ServerConfig(BaseSettings):
    """Server configuration settings."""
    
    name: str = Field(
        default="arxiv-mcp",
        description="Server name"
    )
    version: str = Field(
        default="0.1.0",
        description="Server version"
    )
    description: str = Field(
        default="Model Context Protocol server for arXiv",
        description="Server description"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable metrics collection"
    )
    temp_dir: Optional[str] = Field(
        default=None,
        description="Temporary directory for downloads"
    )
    
    @field_validator('temp_dir')
    @classmethod
    def validate_temp_dir(cls, v):
        if v is not None:
            path = Path(v)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create temp directory {v}: {e}")
            if not path.is_dir():
                raise ValueError(f"Temp path {v} is not a directory")
        return v
    
    model_config = ConfigDict(env_prefix="ARXIV_MCP_SERVER_")


class SecurityConfig(BaseSettings):
    """Security configuration settings."""
    
    max_download_size: int = Field(
        default=100_000_000,  # 100MB
        ge=1_000_000,  # 1MB minimum
        description="Maximum PDF download size in bytes"
    )
    allowed_domains: List[str] = Field(
        default=["arxiv.org", "export.arxiv.org"],
        description="Allowed domains for requests"
    )
    sanitize_inputs: bool = Field(
        default=True,
        description="Enable input sanitization"
    )
    validate_arxiv_ids: bool = Field(
        default=True,
        description="Enable strict arXiv ID validation"
    )
    
    model_config = ConfigDict(env_prefix="ARXIV_MCP_SECURITY_")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Sub-configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    arxiv_api: ArxivAPIConfig = Field(default_factory=ArxivAPIConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Global settings
    environment: Literal["development", "production", "testing"] = Field(
        default="development",
        description="Application environment"
    )
    config_file: Optional[str] = Field(
        default=None,
        description="Path to configuration file"
    )
    
    def __init__(self, **kwargs):
        # Load from config file if specified
        if 'config_file' in kwargs or os.getenv('ARXIV_MCP_CONFIG_FILE'):
            config_file = kwargs.get('config_file') or os.getenv('ARXIV_MCP_CONFIG_FILE')
            if config_file and Path(config_file).exists():
                try:
                    import json
                    import yaml
                    
                    config_path = Path(config_file)
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        with open(config_file, 'r') as f:
                            file_config = yaml.safe_load(f)
                    elif config_path.suffix.lower() == '.json':
                        with open(config_file, 'r') as f:
                            file_config = json.load(f)
                    else:
                        raise ConfigurationError(f"Unsupported config file format: {config_file}")
                    
                    # Merge file config with kwargs
                    for key, value in file_config.items():
                        if key not in kwargs:
                            kwargs[key] = value
                            
                except Exception as e:
                    raise ConfigurationError(f"Error loading config file {config_file}: {e}")
        
        super().__init__(**kwargs)
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        if v not in ['development', 'production', 'testing']:
            raise ValueError("Environment must be 'development', 'production', or 'testing'")
        return v
    
    model_config = ConfigDict(env_prefix="ARXIV_MCP_", case_sensitive=False)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        reload: Force reload of settings
        
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None or reload:
        try:
            _settings = Settings()
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    return _settings


def update_settings(**kwargs) -> Settings:
    """
    Update settings with new values.
    
    Args:
        **kwargs: Settings to update
        
    Returns:
        Updated settings instance
    """
    global _settings
    try:
        _settings = Settings(**kwargs)
        return _settings
    except Exception as e:
        raise ConfigurationError(f"Failed to update configuration: {e}")


def reset_settings() -> None:
    """
    Reset the global settings instance.
    
    This is primarily useful for testing to ensure a clean state.
    """
    global _settings
    _settings = None


def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().environment == "development"


def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().environment == "production"


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_settings().environment == "testing"


# Configuration validation
def validate_configuration() -> List[str]:
    """
    Validate the current configuration and return any issues.
    
    Returns:
        List of validation error messages
    """
    errors = []
    
    try:
        settings = get_settings()
        
        # Check if temp directory is writable
        if settings.server.temp_dir:
            temp_path = Path(settings.server.temp_dir)
            try:
                test_file = temp_path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"Temp directory not writable: {e}")
        
        # Validate API settings
        if settings.arxiv_api.timeout <= 0:
            errors.append("API timeout must be positive")
        
        if settings.arxiv_api.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        # Validate security settings
        if settings.security.max_download_size <= 0:
            errors.append("Max download size must be positive")
        
        if not settings.security.allowed_domains:
            errors.append("At least one allowed domain must be specified")
        
    except Exception as e:
        errors.append(f"Configuration validation error: {e}")
    
    return errors


# Alias for backward compatibility and simpler naming
get_config = get_settings
