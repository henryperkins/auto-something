"""
Dependency Analyzer Module.

This module provides functionality to parse code and identify dependencies
across multiple programming languages, integrating with the hierarchy system
and context management for comprehensive dependency tracking.

Classes:
    DependencyAnalyzer: Analyzes code files to extract dependency information.
"""

import ast
from typing import List, Dict, Set, Optional, Any, Tuple
import os
import logging
from pathlib import Path
from multilang import MultiLanguageManager, LanguageSpec
from hierarchy import CodeHierarchy
from context_manager import ContextManager

class DependencyAnalyzer:
    """
    Analyzes code files to extract dependency information.
    
    This class provides methods to parse code files and extract information
    about imports, function calls, and module interactions, supporting multiple
    programming languages.
    """
    
    def __init__(self, 
                 project_root: str,
                 multilang_manager: Optional[MultiLanguageManager] = None,
                 hierarchy_manager: Optional[CodeHierarchy] = None,
                 context_manager: Optional[ContextManager] = None):
        """
        Initialize the DependencyAnalyzer.
        
        Args:
            project_root: The root directory of the project
            multilang_manager: Optional multi-language support manager
            hierarchy_manager: Optional hierarchy management instance
            context_manager: Optional context management instance
        """
        self.project_root = project_root
        self.module_dependencies: Dict[str, Set[str]] = {}
        self.language_dependencies: Dict[str, Dict[str, Set[str]]] = {}
        self.cross_language_deps: List[Tuple[str, str, str]] = []
        
        # Initialize managers
        self.multilang_manager = multilang_manager or MultiLanguageManager()
        self.hierarchy_manager = hierarchy_manager or CodeHierarchy()
        self.context_manager = context_manager or ContextManager()
        
    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single file to extract dependencies.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing dependency information
        """
        try:
            # Detect language and get parser
            parsed_file = await self.multilang_manager.process_file(file_path)
            if parsed_file.errors:
                return {'errors': parsed_file.errors}
                
            # Extract dependencies based on language
            deps = await self._extract_language_dependencies(
                parsed_file.content,
                parsed_file.language,
                file_path
            )
            
            # Update hierarchy with dependency information
            node = self.hierarchy_manager.add_node(
                path=file_path,
                node_type='file',
                metadata={
                    'language': parsed_file.language,
                    'dependencies': deps
                }
            )
            
            # Update context manager with dependency information
            await self.context_manager.add_or_update_segment(
                file_path,
                parsed_file.content,
                {
                    'language': parsed_file.language,
                    'dependencies': deps,
                    'hierarchy_path': node.get_path()
                }
            )
            
            # Store dependencies
            relative_path = os.path.relpath(file_path, self.project_root)
            module_name = relative_path.replace(os.sep, '.')[:-3]  # Remove .py
            self.module_dependencies[module_name] = deps['imports']
            
            # Track language-specific dependencies
            if parsed_file.language not in self.language_dependencies:
                self.language_dependencies[parsed_file.language] = {}
            self.language_dependencies[parsed_file.language][module_name] = deps['imports']
            
            return {
                'path': file_path,
                'language': parsed_file.language,
                'dependencies': deps
            }
            
        except Exception as e:
            logging.error(f"Failed to analyze {file_path}: {str(e)}")
            return {'error': str(e)}
            
    async def _extract_language_dependencies(self, 
                                          content: Any,
                                          language: str,
                                          file_path: str) -> Dict[str, Set[str]]:
        """Extract dependencies based on language."""
        deps = {
            'imports': set(),
            'internal_calls': set(),
            'external_calls': set()
        }
        
        if language == 'python':
            deps = self._extract_python_dependencies(content)
        elif language == 'javascript':
            deps = self._extract_javascript_dependencies(content)
        # Add more language support as needed
        
        return deps
        
    def _extract_python_dependencies(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract dependencies from Python AST."""
        deps = {
            'imports': set(),
            'internal_calls': set(),
            'external_calls': set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    deps['imports'].add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    deps['imports'].add(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    deps['internal_calls'].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    deps['external_calls'].add(f"{node.func.value.id}.{node.func.attr}")
                    
        return deps
        
    def _extract_javascript_dependencies(self, content: str) -> Dict[str, Set[str]]:
        """Extract dependencies from JavaScript code."""
        deps = {
            'imports': set(),
            'internal_calls': set(),
            'external_calls': set()
        }
        
        # Simple regex-based extraction for demonstration
        import re
        
        # Match import statements
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        deps['imports'].update(re.findall(import_pattern, content))
        
        # Match require statements
        require_pattern = r'require\s*$\s*[\'"]([^\'"]+)[\'"]\s*$'
        deps['imports'].update(re.findall(require_pattern, content))
        
        return deps
        
    async def analyze_project(self) -> Dict[str, Any]:
        """
        Analyze all files in the project.
        
        Returns:
            Dictionary containing project-wide dependency information
        """
        for root, _, files in os.walk(self.project_root):
            for file in files:
                file_path = os.path.join(root, file)
                await self.analyze_file(file_path)
                
        return {
            'module_dependencies': self.module_dependencies,
            'language_dependencies': self.language_dependencies,
            'cross_language_dependencies': self.cross_language_deps
        }
        
    def get_module_dependencies(self) -> Dict[str, Set[str]]:
        """Get the mapping of modules to their dependencies."""
        return self.module_dependencies
        
    async def get_cross_language_dependencies(self) -> List[Dict[str, str]]:
        """
        Get dependencies between different programming languages.
        
        Returns:
            List of cross-language dependencies
        """
        cross_deps = []
        
        for lang1, modules1 in self.language_dependencies.items():
            for lang2, modules2 in self.language_dependencies.items():
                if lang1 != lang2:
                    for mod1, deps1 in modules1.items():
                        for mod2 in modules2.keys():
                            if any(d.startswith(mod2) for d in deps1):
                                cross_deps.append({
                                    'source_language': lang1,
                                    'target_language': lang2,
                                    'source_module': mod1,
                                    'target_module': mod2
                                })
                                
        return cross_deps
        
    def export_dependency_graph(self, output_path: str) -> None:
        """
        Export dependency information as a JSON file.
        
        Args:
            output_path: Path to save the dependency graph
        """
        import json
        
        graph_data = {
            'modules': self.module_dependencies,
            'languages': self.language_dependencies,
            'cross_language': self.cross_language_deps
        }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=list)
            
    @staticmethod
    def merge_dependency_graphs(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple dependency graphs.
        
        Args:
            graphs: List of dependency graphs to merge
            
        Returns:
            Merged dependency graph
        """
        merged = {
            'modules': {},
            'languages': {},
            'cross_language': []
        }
        
        for graph in graphs:
            # Merge module dependencies
            for module, deps in graph['modules'].items():
                if module not in merged['modules']:
                    merged['modules'][module] = set()
                merged['modules'][module].update(deps)
            
            # Merge language dependencies
            for lang, lang_deps in graph['languages'].items():
                if lang not in merged['languages']:
                    merged['languages'][lang] = {}
                for module, deps in lang_deps.items():
                    if module not in merged['languages'][lang]:
                        merged['languages'][lang][module] = set()
                    merged['languages'][lang][module].update(deps)
            
            # Merge cross-language dependencies
            merged['cross_language'].extend(graph['cross_language'])
            
        return merged