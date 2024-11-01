# config.py
"""
Configuration management module for documentation generator.

This module defines dataclasses for managing configuration settings related to
OpenAI and Azure API usage, caching, and general application settings. It supports
loading configurations from YAML files and environment variables.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API settings.

    Attributes:
        api_key (str): The API key for accessing the OpenAI service.
        model (str): The model to use for API requests, defaulting to "gpt-4-0125-preview".
        max_tokens (int): The maximum number of tokens for API responses.
        temperature (float): The temperature setting for the API, controlling randomness.
    """
    api_key: str
    model: str = "gpt-4-0125-preview"
    max_tokens: int = 6000
    temperature: float = 0.2
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OpenAIConfig':
        """Create an OpenAIConfig instance from a dictionary.

        This method allows for the creation of an OpenAIConfig instance using
        a dictionary of settings, with support for environment variable defaults.

        Args:
            data (Dict): A dictionary containing configuration settings.

        Returns:
            OpenAIConfig: An instance of OpenAIConfig with the specified settings.
        """
        return cls(
            api_key=data.get('api_key', os.getenv('OPENAI_API_KEY', '')),
            model=data.get('model', cls.model),
            max_tokens=data.get('max_tokens', cls.max_tokens),
            temperature=data.get('temperature', cls.temperature)
        )

@dataclass
class AzureConfig:
    """Configuration for Azure OpenAI API settings.

    Attributes:
        api_key (str): The API key for accessing the Azure OpenAI service.
        endpoint (str): The endpoint URL for the Azure service.
        deployment_name (str): The deployment name for the Azure service.
    """
    api_key: str
    endpoint: str
    deployment_name: str
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AzureConfig':
        """Create an AzureConfig instance from a dictionary.

        This method allows for the creation of an AzureConfig instance using
        a dictionary of settings, with support for environment variable defaults.

        Args:
            data (Dict): A dictionary containing configuration settings.

        Returns:
            AzureConfig: An instance of AzureConfig with the specified settings.
        """
        return cls(
            api_key=data.get('api_key', os.getenv('AZURE_OPENAI_API_KEY', '')),
            endpoint=data.get('endpoint', os.getenv('AZURE_OPENAI_ENDPOINT', '')),
            deployment_name=data.get('deployment_name', os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', ''))
        )

@dataclass
class CacheConfig:
    """Configuration for caching settings.

    Attributes:
        enabled (bool): Indicates whether caching is enabled.
        directory (str): The directory where cache files are stored.
        max_size_mb (int): The maximum size of the cache in megabytes.
        ttl_hours (int): The time-to-live for cache entries in hours.
    """
    enabled: bool = True
    directory: str = ".cache"
    max_size_mb: int = 100
    ttl_hours: int = 24
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheConfig':
        """Create a CacheConfig instance from a dictionary.

        This method allows for the creation of a CacheConfig instance using
        a dictionary of settings.

        Args:
            data (Dict): A dictionary containing configuration settings.

        Returns:
            CacheConfig: An instance of CacheConfig with the specified settings.
        """
        return cls(
            enabled=data.get('enabled', cls.enabled),
            directory=data.get('directory', cls.directory),
            max_size_mb=data.get('max_size_mb', cls.max_size_mb),
            ttl_hours=data.get('ttl_hours', cls.ttl_hours)
        )

@dataclass
class Config:
    """Main configuration class.

    This class encapsulates all configuration settings for the application,
    including OpenAI, Azure, and caching settings.

    Attributes:
        openai (OpenAIConfig): Configuration for OpenAI API settings.
        azure (Optional[AzureConfig]): Configuration for Azure OpenAI API settings, if used.
        cache (CacheConfig): Configuration for caching settings.
        exclude_dirs (List[str]): Directories to exclude from processing.
        concurrency_limit (int): The maximum number of concurrent operations.
    """
    openai: OpenAIConfig
    azure: Optional[AzureConfig] = None
    cache: CacheConfig = field(default_factory=CacheConfig)
    exclude_dirs: List[str] = field(default_factory=lambda: [".git", ".github", "__pycache__", "venv"])
    concurrency_limit: int = 5
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'Config':
        """Load configuration from file and environment variables.

        This method loads configuration settings from a specified YAML file,
        merging them with environment variable defaults where applicable.

        Args:
            config_path (Optional[str]): Path to the configuration file.

        Returns:
            Config: An instance of Config with loaded settings.

        Raises:
            ValueError: If the configuration file is not found.
        """
        config_data = {}
        
        # Load from file if provided
        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Configuration file not found: {config_path}")
                
        # Create config instance
        return cls(
            openai=OpenAIConfig.from_dict(config_data.get('openai', {})),
            azure=AzureConfig.from_dict(config_data.get('azure', {})) if config_data.get('azure') else None,
            cache=CacheConfig.from_dict(config_data.get('cache', {})),
            exclude_dirs=config_data.get('exclude_dirs', [".git", ".github", "__pycache__", "venv"]),
            concurrency_limit=config_data.get('concurrency_limit', 5)
        )
    
    def validate(self) -> List[str]:
        """Validate the configuration.

        This method checks the configuration for completeness and correctness,
        returning a list of errors if any issues are found.

        Returns:
            List[str]: A list of error messages, or an empty list if the configuration is valid.
        """
        errors = []
        
        # Validate OpenAI config
        if not self.openai.api_key:
            errors.append("OpenAI API key is required")
            
        # Validate Azure config if present
        if self.azure:
            if not self.azure.api_key:
                errors.append("Azure API key is required when Azure is configured")
            if not self.azure.endpoint:
                errors.append("Azure endpoint is required when Azure is configured")
            if not self.azure.deployment_name:
                errors.append("Azure deployment name is required when Azure is configured")
                
        # Validate cache config
        if self.cache.enabled:
            if self.cache.max_size_mb <= 0:
                errors.append("Cache max size must be positive")
            if self.cache.ttl_hours <= 0:
                errors.append("Cache TTL must be positive")
                
        # Validate concurrency
        if self.concurrency_limit <= 0:
            errors.append("Concurrency limit must be positive")
            
        return errors

def create_default_config(path: str = "config.yaml"):
    """Create a default configuration file.

    This function generates a default YAML configuration file with placeholders
    for API keys and settings, as well as a sample .env file for environment variables.

    Args:
        path (str): The path where the configuration file will be created.
    """
    default_config = {
        'openai': {
            'api_key': '${OPENAI_API_KEY}',
            'model': 'gpt-4-0125-preview',
            'max_tokens': 6000,
            'temperature': 0.2
        },
        'azure': {
            'api_key': '${AZURE_OPENAI_API_KEY}',
            'endpoint': '${AZURE_OPENAI_ENDPOINT}',
            'deployment_name': '${AZURE_OPENAI_DEPLOYMENT_NAME}'
        },
        'cache': {
            'enabled': True,
            'directory': '.cache',
            'max_size_mb': 100,
            'ttl_hours': 24
        },
        'exclude_dirs': [
            '.git',
            '.github',
            '__pycache__',
            'venv'
        ],
        'concurrency_limit': 5
    }
    
    with open(path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
        
    # Create a sample .env file if it doesn't exist
    env_path = os.path.join(os.path.dirname(path), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-here
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name-here
""")