"""
Configuration Management Module.

This module defines configuration models using Pydantic for robust validation
of settings related to AI services, caching, logging, concurrency, and more.
It supports loading configurations from YAML files and environment variables
with comprehensive validation.

Classes:
    BaseConfigModel: Base configuration model with enhanced validation.
    OpenAIConfig: Configuration for OpenAI API settings.
    AzureConfig: Configuration for Azure OpenAI API settings.
    CacheConfig: Configuration for caching settings.
    LoggingConfig: Configuration for logging settings.
    ConcurrencyConfig: Configuration for concurrency settings.
    ModelConfig: Configuration for model settings.
    ExtractConfig: Configuration for extraction settings.
    HierarchyConfig: Configuration for hierarchy management.
    MultiLanguageConfig: Configuration for multi-language support.
    ContextOptimizerConfig: Configuration for context optimization.
    Config: Main configuration class encapsulating all settings.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, validator, SecretStr, conint, confloat, DirectoryPath

class BaseConfigModel(BaseModel):
    """Base configuration model with enhanced validation features."""
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent additional fields
        validate_assignment = True  # Validate on attribute assignment
        arbitrary_types_allowed = True  # Allow custom types
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], env_prefix: str = "") -> "BaseConfigModel":
        """
        Create an instance from a dictionary, supporting environment variables.
        
        Args:
            data: Configuration dictionary
            env_prefix: Prefix for environment variables
            
        Returns:
            Instance of the configuration model
        """
        processed_data = {}
        for field_name in cls.__fields__:
            env_var = f"{env_prefix}{field_name}".upper()
            env_value = os.getenv(env_var)
            if env_value is not None:
                processed_data[field_name] = env_value
            elif field_name in data:
                processed_data[field_name] = data[field_name]
        return cls(**processed_data)

class OpenAIConfig(BaseConfigModel):
    """Configuration for OpenAI API settings."""
    api_key: SecretStr = Field(..., description="OpenAI API key")
    model: str = Field("gpt-4", description="Model identifier")
    max_tokens: conint(gt=0, le=32000) = Field(6000, description="Maximum tokens per request")
    temperature: confloat(ge=0, le=2) = Field(0.2, description="Temperature for response generation")

    @validator("model")
    def validate_model(cls, v):
        """Validate model identifier format."""
        allowed_models = {"gpt-4", "gpt-3.5-turbo", "text-davinci-003"}
        if not any(model in v for model in allowed_models):
            raise ValueError(f"Invalid model identifier. Must contain one of: {allowed_models}")
        return v

class AzureConfig(BaseConfigModel):
    """Configuration for Azure OpenAI API settings."""
    api_key: SecretStr = Field(..., description="Azure OpenAI API key")
    endpoint: HttpUrl = Field(..., description="Azure endpoint URL")
    deployment_name: str = Field(..., description="Azure deployment name")

    @validator("deployment_name")
    def validate_deployment_name(cls, v):
        """Validate deployment name format."""
        if not re.match(r"^[a-zA-Z0-9-]+$", v):
            raise ValueError("Deployment name must contain only letters, numbers, and hyphens")
        return v

class CacheConfig(BaseConfigModel):
    """Configuration for caching settings."""
    enabled: bool = Field(True, description="Enable/disable caching")
    directory: DirectoryPath = Field(".cache", description="Cache directory path")
    max_size_mb: conint(gt=0) = Field(100, description="Maximum cache size in MB")
    ttl_hours: conint(gt=0) = Field(24, description="Cache TTL in hours")

    @validator("directory")
    def validate_directory(cls, v):
        """Validate cache directory path."""
        if not os.access(str(v), os.W_OK):
            try:
                Path(v).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Cannot create or access cache directory: {str(e)}")
        return v

class LoggingConfig(BaseConfigModel):
    """Configuration for logging settings."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("%(asctime)s - %(levelname)s - %(message)s", description="Log format string")
    log_file: str = Field("application.log", description="Log file path")

    @validator("level")
    def validate_level(cls, v):
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of: {valid_levels}")
        return v.upper()

class ConcurrencyConfig(BaseConfigModel):
    """Configuration for concurrency settings."""
    limit: conint(gt=0, le=100) = Field(5, description="Concurrency limit")
    max_workers: conint(gt=0, le=32) = Field(4, description="Maximum worker threads")

    @validator("max_workers")
    def validate_max_workers(cls, v, values):
        """Validate max_workers does not exceed concurrency limit."""
        if "limit" in values and v > values["limit"]:
            raise ValueError("max_workers cannot exceed concurrency limit")
        return v

class ModelConfig(BaseConfigModel):
    """Configuration for model settings."""
    context_size_limit: conint(gt=0) = Field(10, description="Maximum context size")
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Embedding model identifier"
    )

class ExtractConfig(BaseConfigModel):
    """Configuration for extraction settings."""
    complexity_threshold: conint(gt=0) = Field(10, description="Complexity threshold")
    line_count_limit: conint(gt=0) = Field(100, description="Maximum lines per function")
    significant_operations: Set[str] = Field(
        default_factory=lambda: {
            "open", "connect", "execute", "write",
            "read", "send", "recv"
        },
        description="Significant operations to track"
    )

class HierarchyConfig(BaseConfigModel):
    """Configuration for hierarchy management."""
    enabled: bool = Field(True, description="Enable hierarchical documentation")
    max_depth: int = Field(5, description="Maximum depth for hierarchical structure")
    group_by: str = Field("module", description="Primary grouping criterion")
    cross_references: bool = Field(True, description="Enable cross-references")

class MultiLanguageConfig(BaseConfigModel):
    """Configuration for multi-language support."""
    enabled: bool = Field(True, description="Enable multi-language support")
    languages: List[str] = Field(
        default_factory=lambda: ["python", "javascript", "java", "cpp"],
        description="Enabled languages"
    )
    parser_timeout: int = Field(30, description="Parser timeout in seconds")
    fallback_language: str = Field("python", description="Fallback language if detection fails")

class ContextOptimizerConfig(BaseConfigModel):
    """Configuration for context optimization."""
    enabled: bool = Field(True, description="Enable context optimization")
    target_token_usage: float = Field(0.9, description="Target token usage ratio")
    prediction_confidence_threshold: float = Field(
        0.8,
        description="Minimum confidence for token predictions"
    )
    priority_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "usage_frequency": 0.4,
            "modification_recency": 0.3,
            "complexity": 0.3
        },
        description="Weights for priority calculations"
    )

class Config(BaseConfigModel):
    """Main configuration class encapsulating all settings."""
    openai: OpenAIConfig
    azure: Optional[AzureConfig] = None
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    extract: ExtractConfig = Field(default_factory=ExtractConfig)
    exclude_dirs: List[str] = Field(
        default_factory=lambda: [".git", ".github", "__pycache__", "venv"],
        description="Directories to exclude"
    )
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    multilang: MultiLanguageConfig = Field(default_factory=MultiLanguageConfig)
    context_optimizer: ContextOptimizerConfig = Field(default_factory=ContextOptimizerConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from a YAML file and environment variables.

        Args:
            config_path: Optional path to configuration file

        Returns:
            Config instance with validated settings

        Raises:
            ValueError: If configuration validation fails
        """
        try:
            # Load configuration from file
            config_data = {}
            if config_path:
                path = Path(config_path)
                if path.exists():
                    with open(path) as f:
                        config_data = yaml.safe_load(f)
                else:
                    logging.warning(f"Configuration file not found: {config_path}")

            # Create configuration instance with validation
            config = cls(
                openai=OpenAIConfig.from_dict(config_data.get("openai", {}), "OPENAI_"),
                azure=AzureConfig.from_dict(config_data.get("azure", {}), "AZURE_") if config_data.get("azure") else None,
                cache=CacheConfig.from_dict(config_data.get("cache", {}), "CACHE_"),
                logging=LoggingConfig.from_dict(config_data.get("logging", {}), "LOG_"),
                concurrency=ConcurrencyConfig.from_dict(config_data.get("concurrency", {}), "CONCURRENCY_"),
                model=ModelConfig.from_dict(config_data.get("model", {}), "MODEL_"),
                extract=ExtractConfig.from_dict(config_data.get("extract", {}), "EXTRACT_"),
                hierarchy=HierarchyConfig.from_dict(config_data.get("hierarchy", {}), "HIERARCHY_"),
                multilang=MultiLanguageConfig.from_dict(config_data.get("multilang", {}), "MULTILANG_"),
                context_optimizer=ContextOptimizerConfig.from_dict(config_data.get("context_optimizer", {}), "CONTEXT_OPTIMIZER_"),
                exclude_dirs=config_data.get("exclude_dirs", [".git", ".github", "__pycache__", "venv"])
            )

            # Perform cross-validation
            config.validate_configuration()
            return config

        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def validate_configuration(self):
        """
        Perform cross-field validation of the configuration.

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # Validate API configurations
        if not self.openai.api_key.get_secret_value():
            errors.append("OpenAI API key is required")

        if self.azure:
            if not self.azure.api_key.get_secret_value():
                errors.append("Azure API key is required when Azure is configured")
            if not self.azure.endpoint:
                errors.append("Azure endpoint is required when Azure is configured")
            if not self.azure.deployment_name:
                errors.append("Azure deployment name is required when Azure is configured")

        # Validate cache configuration
        if self.cache.enabled:
            try:
                cache_dir = Path(self.cache.directory)
                cache_dir.mkdir(parents=True, exist_ok=True)
                if not os.access(str(cache_dir), os.W_OK):
                    errors.append(f"Cache directory {cache_dir} is not writable")
            except Exception as e:
                errors.append(f"Failed to create cache directory: {str(e)}")

        # Validate logging configuration
        try:
            log_dir = Path(self.logging.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(str(log_dir), os.W_OK):
                errors.append(f"Log directory {log_dir} is not writable")
        except Exception as e:
            errors.append(f"Failed to create log directory: {str(e)}")

        # Check for configuration conflicts
        if self.model.context_size_limit > self.openai.max_tokens:
            errors.append("context_size_limit cannot exceed max_tokens")

        if errors:
            raise ValueError("\n".join(errors))

def create_default_config(path: str = "config.yaml"):
    """
    Create a default configuration file.

    Args:
        path: Path where to create the configuration file
    """
    default_config = {
        'openai': {
            'api_key': '${OPENAI_API_KEY}',
            'model': 'gpt-4',
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
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'log_file': 'application.log'
        },
        'concurrency': {
            'limit': 5,
            'max_workers': 4
        },
        'model': {
            'context_size_limit': 10,
            'embedding_model_name': 'all-MiniLM-L6-v2'
        },
        'extract': {
            'complexity_threshold': 10,
            'line_count_limit': 100,
            'significant_operations': [
                'open', 'connect', 'execute',
                'write', 'read', 'send', 'recv'
            ]
        },
        'hierarchy': {
            'enabled': True,
            'max_depth': 5,
            'group_by': 'module',
            'cross_references': True
        },
        'multilang': {
            'enabled': True,
            'languages': ['python', 'javascript', 'java', 'cpp'],
            'parser_timeout': 30,
            'fallback_language': 'python'
        },
        'context_optimizer': {
            'enabled': True,
            'target_token_usage': 0.9,
            'prediction_confidence_threshold': 0.8,
            'priority_weights': {
                'usage_frequency': 0.4,
                'modification_recency': 0.3,
                'complexity': 0.3
            }
        },
        'exclude_dirs': [
            '.git',
            '.github',
            '__pycache__',
            'venv'
        ]
    }

    with open(path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)

    # Create .env template
    env_path = os.path.join(os.path.dirname(path), '.env')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name-here
""")