"""
Input validation module for documentation generator.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl, constr
import os
from pathlib import Path

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
            ValueError: If the path does not exist.
        """
        path = Path(self.path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {self.path}")
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
            ValueError: If the URL is not a valid Git repository URL.
        """
        url_str = str(self.url)
        if not url_str.endswith('.git') and not url_str.startswith(('http://', 'https://')):
            raise ValueError("Invalid Git repository URL")
        return self.url

class FunctionAnalysisInput(BaseModel):
    """Validation model for function analysis input.

    Attributes:
        name (str): The name of the function, must be at least one character.
        code (str): The code of the function, must be non-empty.
        docstring (Optional[str]): The docstring of the function, if any.
        params (List[tuple]): A list of tuples representing function parameters.
        return_type (Optional[str]): The return type of the function, if specified.
    """
    name: str = Field(min_length=1)
    code: str = Field(min_length=1)
    docstring: Optional[str] = None
    params: List[tuple] = Field(default_factory=list)
    return_type: Optional[str] = None

    @property
    def validate_code(self) -> str:
        if not self.code.strip():
            raise ValueError("Code cannot be empty or only whitespace")
        return self.code

    @property
    def validate_params(self) -> List[tuple]:
        for param in self.params:
            if not isinstance(param, tuple) or len(param) != 2:
                raise ValueError("Each parameter must be a tuple of (name, type)")
        return self.params

class CLIArguments(BaseModel):
    """Validation model for command line arguments.

    Attributes:
        input_path (Union[str, HttpUrl]): The input path, which can be a local path or a Git repository URL.
        output_file (str): The path to the output file.
        concurrency (int): The concurrency level for processing, between 1 and 20.
        service (str): The AI service to use, either 'openai' or 'azure'.
        config_file (Optional[str]): The path to the configuration file, if specified.
    """
    input_path: Union[str, HttpUrl]
    output_file: str
    concurrency: int = Field(default=5, ge=1, le=20)
    service: str = Field(default="openai", pattern="^(openai|azure)$")  # Changed from regex to pattern
    config_file: Optional[str] = None

    @property
    def validate_output_file(self) -> str:
        path = Path(self.output_file)
        if path.exists() and not path.is_file():
            raise ValueError(f"Output path exists but is not a file: {self.output_file}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.unlink()
        except Exception as e:
            raise ValueError(f"Cannot write to output path: {self.output_file} ({str(e)})")
        return str(path)

class DocstringValidation(BaseModel):
    """Validation model for docstring content.

    Attributes:
        summary (str): A brief summary of the function, with length constraints.
        description (Optional[str]): A detailed description of the function, if available.
        args (Dict[str, str]): A dictionary mapping argument names to their descriptions.
        returns (Optional[str]): A description of the return value, if applicable.
        raises (Optional[Dict[str, str]]): A dictionary of exceptions that the function may raise.
    """
    summary: str = Field(min_length=10, max_length=1000)
    description: Optional[str] = None
    args: Dict[str, str] = Field(default_factory=dict)
    returns: Optional[str] = None
    raises: Optional[Dict[str, str]] = None

    @property
    def validate_args(self) -> Dict[str, str]:
        for arg_name, description in self.args.items():
            if not description or not description.strip():
                raise ValueError(f"Empty description for argument: {arg_name}")
        return self.args

class AnalysisConfiguration(BaseModel):
    """Validation model for analysis configuration.

    Attributes:
        max_line_length (int): The maximum line length allowed, between 50 and 200.
        complexity_threshold (int): The threshold for complexity, minimum of 1.
        min_docstring_length (int): The minimum length for docstrings, minimum of 10.
        include_private_methods (bool): Whether to include private methods in the analysis.
        exclude_patterns (List[str]): A list of regex patterns for excluding certain code segments.
    """
    max_line_length: int = Field(default=100, ge=50, le=200)
    complexity_threshold: int = Field(default=10, ge=1)
    min_docstring_length: int = Field(default=20, ge=10)
    include_private_methods: bool = False
    exclude_patterns: List[str] = Field(default_factory=list)

    @property
    def validate_patterns(self) -> List[str]:
        import re
        for pattern in self.exclude_patterns:
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {pattern}")
        return self.exclude_patterns

def validate_input_files(files: List[str]) -> List[str]:
    """
    Validate a list of input files.
    
    Args:
        files: List of file paths to validate
        
    Returns:
        List of validated absolute file paths
        
    Raises:
        ValueError: If any file is invalid
    """
    validated_files = []
    for file in files:
        try:
            validated = FileValidation(path=file)
            if not validated.path.endswith('.py'):
                raise ValueError(f"Not a Python file: {file}")
            validated_files.append(validated.validate_path)
        except Exception as e:
            raise ValueError(f"Invalid file {file}: {str(e)}")
    return validated_files

def validate_git_repository(url: str, branch: Optional[str] = None) -> GitRepoValidation:
    """
    Validate a Git repository URL and branch.
    
    Args:
        url: Repository URL to validate
        branch: Optional branch name
        
    Returns:
        Validated GitRepoValidation instance
        
    Raises:
        ValueError: If the repository URL or branch is invalid
    """
    try:
        return GitRepoValidation(url=url, branch=branch)
    except Exception as e:
        raise ValueError(f"Invalid Git repository: {str(e)}")

def validate_function_analysis_input(func_data: Dict) -> FunctionAnalysisInput:
    """
    Validate function analysis input data.
    
    Args:
        func_data: Dictionary containing function data
        
    Returns:
        Validated FunctionAnalysisInput instance
        
    Raises:
        ValueError: If the input data is invalid
    """
    try:
        return FunctionAnalysisInput(**func_data)
    except Exception as e:
        raise ValueError(f"Invalid function analysis input: {str(e)}")