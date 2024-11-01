"""
Multi-Language Support Module.

This module provides functionality for detecting and parsing different programming
languages, enabling context management across various language ecosystems. It
implements a flexible parser system that can be extended to support additional
languages while maintaining consistent documentation output.

Classes:
    LanguageSpec: Defines language-specific parsing rules and patterns.
    LanguageDetector: Detects programming languages from file content and extensions.
    BaseLanguageParser: Abstract base class for language-specific parsers.
    ParserRegistry: Manages available language parsers.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Pattern, Any, Type
from pathlib import Path
import ast
import tokenize
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LanguageSpec:
    """
    Defines specifications for a programming language.
    
    Attributes:
        name: Language name
        extensions: File extensions associated with the language
        comment_single: Single-line comment marker
        comment_multi_start: Multi-line comment start marker
        comment_multi_end: Multi-line comment end marker
        string_delimiters: Set of string delimiter characters
        keywords: Set of language keywords
    """
    name: str
    extensions: Set[str]
    comment_single: str
    comment_multi_start: Optional[str] = None
    comment_multi_end: Optional[str] = None
    string_delimiters: Set[str] = field(default_factory=lambda: {'\'', '"'})
    keywords: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate and normalize the language specification."""
        self.extensions = {ext.lower().lstrip('.') for ext in self.extensions}
        if bool(self.comment_multi_start) != bool(self.comment_multi_end):
            raise ValueError("Both multi-line comment markers must be provided or neither")

# Define common language specifications
LANGUAGE_SPECS = {
    'python': LanguageSpec(
        name='Python',
        extensions={'py', 'pyw', 'pyi'},
        comment_single='#',
        comment_multi_start='"""',
        comment_multi_end='"""',
        keywords={'def', 'class', 'import', 'from', 'async', 'await'}
    ),
    'javascript': LanguageSpec(
        name='JavaScript',
        extensions={'js', 'jsx', 'ts', 'tsx'},
        comment_single='//',
        comment_multi_start='/*',
        comment_multi_end='*/',
        keywords={'function', 'class', 'import', 'export', 'const', 'let', 'var'}
    ),
    'java': LanguageSpec(
        name='Java',
        extensions={'java'},
        comment_single='//',
        comment_multi_start='/*',
        comment_multi_end='*/',
        keywords={'class', 'interface', 'enum', 'public', 'private', 'protected'}
    ),
    'cpp': LanguageSpec(
        name='C++',
        extensions={'cpp', 'hpp', 'cc', 'h', 'cxx'},
        comment_single='//',
        comment_multi_start='/*',
        comment_multi_end='*/',
        keywords={'class', 'struct', 'namespace', 'template', 'public', 'private'}
    )
}

class LanguageDetector:
    """
    Detects programming languages from file content and extensions.
    
    This class uses a combination of file extensions and content analysis
    to determine the programming language of a given file.
    """
    
    def __init__(self, specs: Dict[str, LanguageSpec] = None):
        """
        Initialize the language detector.
        
        Args:
            specs: Dictionary of language specifications
        """
        self.specs = specs or LANGUAGE_SPECS
        self._extension_map = self._build_extension_map()
        
    def _build_extension_map(self) -> Dict[str, str]:
        """Build a mapping of file extensions to language names."""
        extension_map = {}
        for lang_id, spec in self.specs.items():
            for ext in spec.extensions:
                if ext in extension_map:
                    logger.warning(f"Extension .{ext} is claimed by multiple languages")
                extension_map[ext] = lang_id
        return extension_map
        
    def detect_language(self, filepath: str, content: Optional[str] = None) -> Optional[str]:
        """
        Detect the programming language of a file.
        
        Args:
            filepath: Path to the file
            content: Optional file content (if already read)
            
        Returns:
            Language identifier or None if unknown
        """
        # Check file extension first
        ext = Path(filepath).suffix.lower().lstrip('.')
        if ext in self._extension_map:
            return self._extension_map[ext]
            
        # If no match by extension and content is provided, analyze content
        if content:
            return self._detect_from_content(content)
            
        return None
        
    def _detect_from_content(self, content: str) -> Optional[str]:
        """
        Detect language by analyzing file content.
        
        Args:
            content: File content to analyze
            
        Returns:
            Language identifier or None if unknown
        """
        # Count occurrences of language-specific patterns
        scores = {lang_id: 0 for lang_id in self.specs}
        
        for lang_id, spec in self.specs.items():
            # Check for language-specific keywords
            for keyword in spec.keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                scores[lang_id] += len(re.findall(pattern, content))
                
            # Check for comment patterns
            scores[lang_id] += content.count(spec.comment_single) * 2
            if spec.comment_multi_start:
                scores[lang_id] += content.count(spec.comment_multi_start) * 3
                
        # Return language with highest score if significant
        if scores:
            max_score = max(scores.values())
            if max_score > 5:  # Threshold for confidence
                return max(scores.items(), key=lambda x: x[1])[0]
                
        return None

class BaseLanguageParser(ABC):
    """
    Abstract base class for language-specific parsers.
    
    This class defines the interface that all language parsers must implement
    to provide consistent parsing capabilities across different languages.
    """
    
    def __init__(self, spec: LanguageSpec):
        """
        Initialize the parser.
        
        Args:
            spec: Language specification
        """
        self.spec = spec
        
    @abstractmethod
    def parse_code(self, content: str) -> Dict[str, Any]:
        """
        Parse code content and extract documentation-relevant information.
        
        Args:
            content: Source code content
            
        Returns:
            Dictionary containing parsed information
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
        
    @abstractmethod
    def extract_docstring(self, node: Any) -> Optional[str]:
        """
        Extract docstring from a code node.
        
        Args:
            node: Code node to extract docstring from
            
        Returns:
            Extracted docstring or None if not found
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_node_name(self, node: Any) -> str:
        """
        Get the name of a code node.
        
        Args:
            node: Code node to get name from
            
        Returns:
            Name of the node
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def extract_comments(self, content: str) -> List[str]:
        """
        Extract comments from code content.
        
        Args:
            content: Source code content
            
        Returns:
            List of extracted comments
        """
        comments = []
        
        # Handle single-line comments
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(self.spec.comment_single):
                comments.append(line[len(self.spec.comment_single):].strip())
                
        # Handle multi-line comments if supported
        if self.spec.comment_multi_start and self.spec.comment_multi_end:
            pattern = re.escape(self.spec.comment_multi_start) + r'(.*?)' + \
                     re.escape(self.spec.comment_multi_end)
            for match in re.finditer(pattern, content, re.DOTALL):
                comments.append(match.group(1).strip())
                
        return comments

class PythonParser(BaseLanguageParser):
    """Parser implementation for Python code."""
    
    def parse_code(self, content: str) -> Dict[str, Any]:
        """Parse Python code content."""
        try:
            tree = ast.parse(content)
            return self._process_node(tree)
        except Exception as e:
            logger.error(f"Error parsing Python code: {str(e)}")
            return {'error': str(e)}
            
    def _process_node(self, node: ast.AST) -> Dict[str, Any]:
        """Process an AST node and its children."""
        result = {
            'classes': [],
            'functions': [],
            'imports': [],
            'docstring': self.extract_docstring(node)
        }
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                result['classes'].append(self._process_class(child))
            elif isinstance(child, ast.FunctionDef):
                result['functions'].append(self._process_function(child))
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                result['imports'].extend(self._process_import(child))
                
        return result
        
    def _process_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Process a class definition node."""
        return {
            'name': node.name,
            'docstring': self.extract_docstring(node),
            'methods': [
                self._process_function(method)
                for method in node.body
                if isinstance(method, ast.FunctionDef)
            ],
            'decorators': [
                self.get_node_name(decorator)
                for decorator in node.decorator_list
            ],
            'bases': [
                self.get_node_name(base)
                for base in node.bases
            ]
        }
        
    def _process_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Process a function definition node."""
        return {
            'name': node.name,
            'docstring': self.extract_docstring(node),
            'args': [arg.arg for arg in node.args.args],
            'decorators': [
                self.get_node_name(decorator)
                for decorator in node.decorator_list
            ],
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        }
        
    def _process_import(self, node: ast.AST) -> List[Dict[str, str]]:
        """Process an import node."""
        imports = []
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append({
                    'name': name.name,
                    'alias': name.asname
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for name in node.names:
                imports.append({
                    'module': module,
                    'name': name.name,
                    'alias': name.asname
                })
        return imports
        
    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node."""
        docstring = ast.get_docstring(node)
        return docstring.strip() if docstring else None
        
    def get_node_name(self, node: ast.AST) -> str:
        """Get the name of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self.get_node_name(node.func)
        return str(node)

class JavaScriptParser(BaseLanguageParser):
    """Parser implementation for JavaScript code."""
    
    def parse_code(self, content: str) -> Dict[str, Any]:
        """Parse JavaScript code content."""
        # Note: This is a simplified implementation
        # In practice, you might want to use a proper JS parser like esprima
        result = {
            'classes': [],
            'functions': [],
            'imports': []
        }
        
        # Simple regex-based parsing for demonstration
        # Class detection
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            base_class = match.group(2)
            result['classes'].append({
                'name': class_name,
                'base': base_class,
                'docstring': self._find_preceding_comment(content, match.start())
            })
        
        # Function detection
        func_pattern = r'(?:async\s+)?(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?function)'
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1) or match.group(2)
            result['functions'].append({
                'name': func_name,
                'docstring': self._find_preceding_comment(content, match.start())
            })
        
        # Import detection
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        result['imports'] = [
            {'module': match.group(1)}
            for match in re.finditer(import_pattern, content)
        ]
        
        return result
        
    def _find_preceding_comment(self, content: str, position: int) -> Optional[str]:
        """Find JSDoc comment preceding a position in the code."""
        lines = content[:position].splitlines()
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('/**') and line.endswith('*/'):
                return line[3:-2].strip()
            elif not line or line.startswith('//'):
                continue
            else:
                break
        return None
        
    def extract_docstring(self, node: Any) -> Optional[str]:
        """Extract docstring from a node."""
        # Simplified implementation
        return None
        
    def get_node_name(self, node: Any) -> str:
        """Get the name of a node."""
        return str(node)

class ParserRegistry:
    """Registry for managing language parsers."""
    
    _parsers: Dict[str, Type[BaseLanguageParser]] = {
        'python': PythonParser,
        'javascript': JavaScriptParser
    }
    
    @classmethod
    def register_parser(cls, language: str, parser_class: Type[BaseLanguageParser]) -> None:
        """Register a new parser."""
        cls._parsers[language] = parser_class
        
    @classmethod
    def get_parser(cls, language: str, spec: LanguageSpec) -> Optional[BaseLanguageParser]:
        """Get a parser instance for a language."""
        parser_class = cls._parsers.get(language)
        return parser_class(spec) if parser_class else None

@dataclass
class ParsedFile:
    """Container for parsed file information."""
    language: str
    content: Dict[str, Any]
    filepath: str
    errors: List[str] = field(default_factory=list)

class MultiLanguageManager:
    """
    Main interface for multi-language support.
    
    This class provides the primary interface for working with multiple
    programming languages, managing detection, parsing, and documentation
    generation across different language ecosystems.
    """
    
    def __init__(self, specs: Dict[str, LanguageSpec] = None):
        """Initialize the manager."""
        self.specs = specs or LANGUAGE_SPECS
        self.detector = LanguageDetector(self.specs)
        self.parsed_files: Dict[str, ParsedFile] = {}
        
    async def process_file(self, filepath: str, content: Optional[str] = None) -> ParsedFile:
        """
        Process a source code file.
        
        Args:
            filepath: Path to the file
            content: Optional file content (if already read)
            
        Returns:
            ParsedFile containing the analysis results
        """
        try:
            # Read content if not provided
            if content is None:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            # Detect language
            language = self.detector.detect_language(filepath, content)
            if not language:
                return ParsedFile(
                    language='unknown',
                    content={},
                    filepath=filepath,
                    errors=['Unable to detect language']
                )
                
            # Get parser and parse content
            spec = self.specs[language]
            parser = ParserRegistry.get_parser(language, spec)
            if not parser:
                return ParsedFile(
                    language=language,
                    content={},
                    filepath=filepath,
                    errors=[f'No parser available for {language}']
                )
                
            # Parse the content
            parsed_content = parser.parse_code(content)
            result = ParsedFile(
                language=language,
                content=parsed_content,
                filepath=filepath
            )
            
            # Store the result
            self.parsed_files[filepath] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            return ParsedFile(
                language='unknown',
                content={},
                filepath=filepath,
                errors=[str(e)]
            )
            
    async def get_cross_references(self) -> Dict[str, List[str]]:
        """
        Get cross-references between parsed files.
        
        Returns:
            Dictionary mapping files to their dependencies
        """
        references = {}
        for filepath, parsed in self.parsed_files.items():
            deps = set()
            if 'imports' in parsed.content:
                for imp in parsed.content['imports']:
                    if isinstance(imp, dict):
                        module = imp.get('module', imp.get('name', ''))
                        if module:
                            deps.add(module)
            references[filepath] = list(deps)
        return references
        
    def save_analysis(self, output_path: str) -> None:
        """Save analysis results to a JSON file."""
        data = {
            filepath: {
                'language': parsed.language,
                'content': parsed.content,
                'errors': parsed.errors
            }
            for filepath, parsed in self.parsed_files.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_analysis(cls, input_path: str) -> 'MultiLanguageManager':
        """Load analysis results from a JSON file."""
        manager = cls()
        
        with open(input_path) as f:
            data = json.load(f)
            
        for filepath, file_data in data.items():
            manager.parsed_files[filepath] = ParsedFile(
                language=file_data['language'],
                content=file_data['content'],
                filepath=filepath,
                errors=file_data.get('errors', [])
            )
            
        return manager