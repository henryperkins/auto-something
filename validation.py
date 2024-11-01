"""
Input validation module for documentation generator.

This module defines validation models and functions for validating inputs
throughout the documentation generation process, including file paths,
Git repositories, function analysis data, and configuration settings.
"""

from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field, HttpUrl, validator
import os
import re
from pathlib import Path

class ValidationError(Exception):
    """Exception raised for validation-related errors."""
    pass

class FileValidation(BaseModel):
    """Validation model for file and directory paths.

    Attributes:
        path (str): The file or directory path to validate.
    """
    path: str
    
    @property
    def validate_path(self) -> str:
        """Validate that the path exists and return its absolute path.

        Returns:
            str: The absolute path if valid.

        Raises:
            ValidationError: If the path does not exist.
        """
        path = Path(self.path)
        if not path.exists():
            raise ValidationError(f"Path does not exist: {self.path}")
        return str(path.absolute())

class GitRepoValidation(BaseModel):
    """Validation model for Git repository URLs.

    Attributes:
        url (HttpUrl): The URL of the Git repository.
        branch (Optional[str]): The branch name, defaulting to "main".
    """
    url: HttpUrl
    branch: Optional[str] = "main"
    
    @property
    def validate_git_url(self) -> HttpUrl:
        """Validate that the URL is a valid Git repository URL.

        Returns:
            HttpUrl: The validated Git repository URL.

        Raises:
            ValidationError: If the URL is not a valid Git repository URL.
        """
        url_str = str(self.url)
        if not url_str.endswith('.git') and not url_str.startswith(('http://', 'https://')):
            raise ValidationError("Invalid Git repository URL")
        return self.url

class DependencyImport(BaseModel):
    """Validation model for import dependencies.

    Attributes:
        module (str): The module being imported.
        name (str): The name being imported.
        alias (Optional[str]): The alias used in the import, if any.
        is_type_hint (bool): Whether this import is used for type hints.
    """
    module: str
    name: str
    alias: Optional[str] = None
    is_type_hint: bool = False

    @validator('module')
    def validate_module_name(cls, v):
        """Validate module name format."""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$', v):
            raise ValidationError(f"Invalid module name: {v}")
        return v

class DependencyInfo(BaseModel):
    """Validation model for complete dependency information.

    Attributes:
        imports (List[DependencyImport]): List of import dependencies.
        internal_calls (List[str]): List of internal function calls.
        external_calls (List[str]): List of external function/method calls.
        raises (List[Dict[str, str]]): List of exceptions that might be raised.
        affects (List[Dict[str, str]]): List of side effects.
        uses (List[Dict[str, str]]): List of significant operations.
    """
    imports: List[DependencyImport] = Field(default_factory=list)
    internal_calls: List[str] = Field(default_factory=list)
    external_calls: List[str] = Field(default_factory=list)
    raises: List[Dict[str, str]] = Field(default_factory=list)
    affects: List[Dict[str, str]] = Field(default_factory=list)
    uses: List[Dict[str, str]] = Field(default_factory=list)

    @validator('raises', 'affects', 'uses')
    def validate_dict_list(cls, v):
        """Validate dictionary lists have required keys."""
        for item in v:
            if not isinstance(item, dict) or not all(k in item for k in ['context']):
                raise ValidationError("Each item must be a dictionary with 'context' key")
        return v

class FunctionAnalysisInput(BaseModel):
    """Validation model for function analysis input."""
    name: str = Field(min_length=1)
    code: str = Field(min_length=1)
    docstring: Optional[str] = None
    params: List[Tuple[str, str]] = Field(default_factory=list)
    return_type: Optional[str] = None
    dependencies: Optional[DependencyInfo] = None
    complexity: Optional[int] = Field(None, ge=1)

    @property
    def validate_code(self) -> str:
        """Validate that the code is not empty."""
        if not self.code.strip():
            raise ValidationError("Code cannot be empty or only whitespace")
        return self.code

    @property
    def validate_params(self) -> List[Tuple[str, str]]:
        """Validate parameter format."""
        for param in self.params:
            if not isinstance(param, tuple) or len(param) != 2:
                raise ValidationError("Each parameter must be a tuple of (name, type)")
            if not isinstance(param[0], str) or not isinstance(param[1], str):
                raise ValidationError("Parameter name and type must be strings")
        return self.params

class CLIArguments(BaseModel):
    """Validation model for command line arguments.

    Attributes:
        input_path (Union[str, HttpUrl]): Input path or repository URL.
        output_file (str): Output file path.
        concurrency (int): Number of concurrent operations.
        service (str): AI service to use ('openai' or 'azure').
        config_file (Optional[str]): Path to configuration file.
    """
    input_path: Union[str, HttpUrl]
    output_file: str
    concurrency: int = Field(default=5, ge=1, le=20)
    service: str = Field(default="openai", regex="^(openai|azure)$")
    config_file: Optional[str] = None

    @property
    def validate_output_file(self) -> str:
        """Validate output file path is writable."""
        path = Path(self.output_file)
        if path.exists() and not path.is_file():
            raise ValidationError(f"Output path exists but is not a file: {self.output_file}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.unlink()
        except Exception as e:
            raise ValidationError(f"Cannot write to output path: {self.output_file} ({str(e)})")
        return str(path)

class DocstringValidation(BaseModel):
    """Validation model for docstring content.

    Attributes:
        summary (str): Brief description of the function.
        description (Optional[str]): Detailed description.
        args (Dict[str, str]): Argument descriptions.
        returns (Optional[str]): Return value description.
        raises (Optional[Dict[str, str]]): Exception descriptions.
    """
    summary: str = Field(min_length=10, max_length=1000)
    description: Optional[str] = None
    args: Dict[str, str] = Field(default_factory=dict)
    returns: Optional[str] = None
    raises: Optional[Dict[str, str]] = None

    @property
    def validate_args(self) -> Dict[str, str]:
        """Validate argument descriptions."""
        for arg_name, description in self.args.items():
            if not description or not description.strip():
                raise ValidationError(f"Empty description for argument: {arg_name}")
        return self.args

class AnalysisConfiguration(BaseModel):
    """Validation model for analysis configuration.

    Attributes:
        max_line_length (int): Maximum allowed line length.
        complexity_threshold (int): Maximum complexity threshold.
        min_docstring_length (int): Minimum required docstring length.
        include_private_methods (bool): Whether to include private methods.
        exclude_patterns (List[str]): Patterns for excluding files/directories.
    """
    max_line_length: int = Field(default=100, ge=50, le=200)
    complexity_threshold: int = Field(default=10, ge=1)
    min_docstring_length: int = Field(default=20, ge=10)
    include_private_methods: bool = False
    exclude_patterns: List[str] = Field(default_factory=list)

    @property
    def validate_patterns(self) -> List[str]:
        """Validate regex patterns are valid."""
        for pattern in self.exclude_patterns:
            try:
                re.compile(pattern)
            except re.error:
                raise ValidationError(f"Invalid regex pattern: {pattern}")
        return self.exclude_patterns

def validate_input_files(files: List[str]) -> List[str]:
    """Validate a list of input files.
    
    Args:
        files: List of file paths to validate
        
    Returns:
        List of validated absolute file paths
        
    Raises:
        ValidationError: If any file is invalid
    """
    validated_files = []
    for file in files:
        try:
            validated = FileValidation(path=file)
            if not validated.path.endswith('.py'):
                raise ValidationError(f"Not a Python file: {file}")
            validated_files.append(validated.validate_path)
        except Exception as e:
            raise ValidationError(f"Invalid file {file}: {str(e)}")
    return validated_files

def validate_git_repository(url: str, branch: Optional[str] = None) -> GitRepoValidation:
    """Validate a Git repository URL and branch.
    
    Args:
        url: Repository URL to validate
        branch: Optional branch name
        
    Returns:
        Validated GitRepoValidation instance
        
    Raises:
        ValidationError: If the repository URL or branch is invalid
    """
    try:
        return GitRepoValidation(url=url, branch=branch)
    except Exception as e:
        raise ValidationError(f"Invalid Git repository: {str(e)}")

def validate_function_analysis_input(func_data: Dict[str, Any]) -> FunctionAnalysisInput:
    """Validate function analysis input data.
    
    Args:
        func_data: Dictionary containing function data to validate
        
    Returns:
        Validated FunctionAnalysisInput instance
        
    Raises:
        ValidationError: If the input data is invalid
    """
    try:
        # Convert dependency dict to DependencyInfo if present
        if "dependencies" in func_data and isinstance(func_data["dependencies"], dict):
            func_data["dependencies"] = DependencyInfo(**func_data["dependencies"])
        
        # Validate the entire function input
        validated = FunctionAnalysisInput(**func_data)
        
        # Trigger property validators
        validated.validate_code
        validated.validate_params
        
        return validated
    except Exception as e:
        raise ValidationError(f"Invalid function analysis input: {str(e)}")

def validate_docstring_content(content: Dict[str, Any]) -> DocstringValidation:
    """Validate docstring content.
    
    Args:
        content: Dictionary containing docstring content to validate
        
    Returns:
        Validated DocstringValidation instance
        
    Raises:
        ValidationError: If the docstring content is invalid
    """
    try:
        validated = DocstringValidation(**content)
        validated.validate_args
        return validated
    except Exception as e:
        raise ValidationError(f"Invalid docstring content: {str(e)}")

def validate_analysis_config(config: Dict[str, Any]) -> AnalysisConfiguration:
    """Validate analysis configuration.
    
    Args:
        config: Dictionary containing configuration settings to validate
        
    Returns:
        Validated AnalysisConfiguration instance
        
    Raises:
        ValidationError: If the configuration is invalid
    """
    try:
        validated = AnalysisConfiguration(**config)
        validated.validate_patterns
        return validated
    except Exception as e:
        raise ValidationError(f"Invalid analysis configuration: {str(e)}")

def validate_service_configuration(service: str) -> None:
    """Validate that required environment variables are set for the selected service."""
    if service == "azure":
        required_vars = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            raise ValidationError(
                f"Missing required Azure OpenAI environment variables: {', '.join(missing_vars)}\n"
                f"Please set these in your .env file or environment."
            )
            
    elif service == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValidationError(
                "Missing OPENAI_API_KEY environment variable.\n"
                "Please set this in your .env file or environment."
            )

def validate_context_code_segments(context_code_segments: List[str], max_context_tokens: int, model_name: str) -> List[str]:
    """Validate and select context code segments within token limits."""
    total_tokens = 0
    selected_segments = []
    tokenizer = tiktoken.encoding_for_model(model_name)
    for code in context_code_segments:
        tokens = tokenizer.encode(code)
        num_tokens = len(tokens)
        if total_tokens + num_tokens <= max_context_tokens:
            selected_segments.append(code)
            total_tokens += num_tokens
        else:
            break
    return selected_segments

def validate_metadata(metadata: Dict[str, Any]) -> str:
    """Validate and format metadata for inclusion in prompts."""
    try:
        metadata_json = json.dumps(metadata, indent=2)
        return f"Metadata:\n```json\n{metadata_json}\n```\n\n"
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid metadata format: {str(e)}")

def validate_token_limits(context_window: int, max_response_tokens: int, system_prompt_tokens: int) -> int:
    """Validate and calculate available prompt tokens."""
    available_prompt_tokens = context_window - max_response_tokens - system_prompt_tokens
    if available_prompt_tokens <= 0:
        logging.warning("Available prompt tokens is non-positive. Adjust max_response_tokens or system prompts.")
        return 1024  # Default to a safe value
    return available_prompt_tokens