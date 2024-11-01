# exceptions.py

class AIServiceError(Exception):
    """Base class for AI service-related exceptions."""
    pass

class AIServiceConfigError(AIServiceError):
    """Exception raised for configuration errors in AI service."""
    pass

class AIServiceResponseError(AIServiceError):
    """Exception raised for response errors from AI service."""
    pass
class WorkflowError(Exception):
    """Exception raised for errors in the workflow process."""
    pass
class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass