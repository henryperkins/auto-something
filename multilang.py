import ast
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Type, TypeVar
from pathlib import Path
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LanguageSpec:
    """Language specification details."""
    name: str
    extensions: Set[str]
    keywords: Set[str]
    comment_single: str
    comment_multi_start: Optional[str] = None
    comment_multi_end: Optional[str] = None

    def __post_init__(self):
        """Validate specification consistency."""
        if not self.extensions:
            raise ValueError(f"Language {self.name} must have at least one file extension")
        if not self.keywords:
            raise ValueError(f"Language {self.name} must have at least one keyword")
        if bool(self.comment_multi_start) != bool(self.comment_multi_end):
            raise ValueError("Both multi-line comment markers must be provided or neither")

# Single source of language specifications
DEFAULT_LANGUAGE_SPECS = {
    'python': LanguageSpec(
        name='Python',
        extensions={'.py', '.pyw', '.pyi'},
        keywords={'def', 'class', 'import', 'from', 'async', 'await'},
        comment_single='#',
        comment_multi_start='"""',
        comment_multi_end='"""'
    ),
    'javascript': LanguageSpec(
        name='JavaScript',
        extensions={'.js', '.jsx'},
        keywords={'function', 'class', 'const', 'let', 'var'},
        comment_single='//',
        comment_multi_start='/*',
        comment_multi_end='*/'
    ),
    'typescript': LanguageSpec(
        name='TypeScript',
        extensions={'.ts', '.tsx'},
        keywords={'function', 'class', 'interface', 'type', 'const'},
        comment_single='//',
        comment_multi_start='/*',
        comment_multi_end='*/'
    )
}

class ParserException(Exception):
    """Base exception for parser errors."""
    pass

class BaseLanguageParser(ABC):
    """Abstract base class for language-specific parsers."""
    def __init__(self, spec: LanguageSpec):
        self.spec = spec
        
    @abstractmethod
    def parse_code(self, content: str) -> Dict[str, Any]:
        """Parse code content and extract documentation-relevant information."""
        pass
        
    @abstractmethod
    def extract_docstring(self, node: Any) -> Optional[str]:
        """Extract docstring from a code node."""
        pass

    @abstractmethod
    def get_node_name(self, node: Any) -> str:
        """Get the name of a code node."""
        pass

    def extract_comments(self, content: str) -> List[str]:
        """Extract comments from code content."""
        comments = []
        
        for line in content.splitlines():
            line = line.strip()
            if line.startswith(self.spec.comment_single):
                comments.append(line[len(self.spec.comment_single):].strip())
                
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
            raise ParserException(str(e))
            
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

class JavaScriptParser(BaseLanguageParser):
    """Parser implementation for JavaScript code."""
    
    def parse_code(self, content: str) -> Dict[str, Any]:
        """Parse JavaScript code content."""
        try:
            result = {
                'classes': [],
                'functions': [],
                'imports': []
            }
            
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
        except Exception as e:
            logger.error(f"Error parsing JavaScript code: {str(e)}")
            raise ParserException(str(e))

    def _find_preceding_comment(self, content: str, position: int) -> Optional[str]:
        """Find JSDoc comment preceding a code block."""
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
        return None  # Handled by _find_preceding_comment

    def get_node_name(self, node: Any) -> str:
        """Get the name of a node."""
        return str(node)

class ParserRegistry:
    """Registry for managing language parsers."""
    _parsers: Dict[str, Type[BaseLanguageParser]] = {
        'python': PythonParser,
        'javascript': JavaScriptParser,
        'typescript': JavaScriptParser  # TypeScript uses JS parser for now
    }

    @classmethod
    def get_parser(cls, language: str, spec: LanguageSpec) -> Optional[BaseLanguageParser]:
        """Get parser instance for a language."""
        parser_class = cls._parsers.get(language)
        if not parser_class:
            logger.warning(f"No parser registered for language: {language}")
            return None
        return parser_class(spec)

@dataclass
class ParsedFile:
    """Container for parsed file results."""
    language: str
    content: Dict[str, Any]
    filepath: str
    errors: List[str] = field(default_factory=list)

@dataclass
class MultiLanguageManager:
    """Manages multi-language parsing and statistics."""
    languages: Optional[List[str]] = None
    _stats: Dict[str, int] = field(default_factory=dict)
    _specs: Dict[str, LanguageSpec] = field(default_factory=dict)
    _parsed_files: Dict[str, ParsedFile] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize manager state."""
        self._stats = {lang: 0 for lang in (self.languages or [])}
        self._specs = DEFAULT_LANGUAGE_SPECS.copy()

    def register_file(self, language: str) -> None:
        """Register a processed file."""
        if language not in self._stats:
            self._stats[language] = 0
        self._stats[language] += 1

    def get_language_stats(self) -> Dict[str, int]:
        """Get language statistics."""
        return self._stats.copy()

    async def process_file(self, filepath: str, content: Optional[str] = None) -> ParsedFile:
        """Process a source code file."""
        try:
            if content is None:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

            ext = Path(filepath).suffix.lower()
            for lang, spec in self._specs.items():
                if ext in spec.extensions:
                    parser = ParserRegistry.get_parser(lang, spec)
                    if parser:
                        result = parser.parse_code(content)
                        self.register_file(lang)
                        parsed_file = ParsedFile(
                            language=lang,
                            content=result,
                            filepath=filepath
                        )
                        self._parsed_files[filepath] = parsed_file
                        return parsed_file

            return ParsedFile(
                language='unknown',
                content={},
                filepath=filepath,
                errors=['Unsupported file type']
            )

        except ParserException as e:
            logger.warning(f"Parser error for {filepath}: {str(e)}")
            return ParsedFile(
                language='unknown',
                content={},
                filepath=filepath,
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
            return ParsedFile(
                language='unknown',
                content={},
                filepath=filepath,
                errors=[str(e)]
            )

    async def get_cross_references(self) -> Dict[str, List[str]]:
        """Get cross-references between files."""
        references = {}
        for filepath, parsed in self._parsed_files.items():
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
        """Save analysis results to JSON."""
        data = {
            filepath: {
                'language': parsed.language,
                'content': parsed.content,
                'errors': parsed.errors
            }
            for filepath, parsed in self._parsed_files.items()
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_analysis(cls, input_path: str) -> 'MultiLanguageManager':
        """Load analysis results from JSON."""
        manager = cls()
        with open(input_path) as f:
            data = json.load(f)
        for filepath, file_data in data.items():
            manager._parsed_files[filepath] = ParsedFile(
                language=file_data['language'],
                content=file_data['content'],
                filepath=filepath,
                errors=file_data.get('errors', [])
            )
        return manager