"""
Extraction module for analyzing code across multiple languages.

This module provides functions to extract information from source code,
including classes, functions, dependencies, and code metrics. It supports
multiple programming languages and integrates with the hierarchy system
and context management for comprehensive analysis.

Classes:
    ExtractionManager: Manages the extraction of code information across multiple languages.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from multilang import MultiLanguageManager, BaseLanguageParser
from hierarchy import CodeHierarchy
from context_manager import ContextManager
import sentry_sdk

class ExtractionManager:
    """
    Manages the extraction of code information across multiple languages.
    
    This class coordinates between language-specific parsers and the
    hierarchy system to extract and organize code information.
    """
    
    def __init__(self, 
                 multilang_manager: MultiLanguageManager,
                 hierarchy_manager: CodeHierarchy,
                 context_manager: ContextManager,
                 significant_operations: Optional[Set[str]] = None):
        """
        Initialize the extraction manager.
        
        Args:
            multilang_manager: Multi-language support manager
            hierarchy_manager: Hierarchy management instance
            context_manager: Context management instance
            significant_operations: Set of significant operations to track
        """
        self.multilang_manager = multilang_manager
        self.hierarchy_manager = hierarchy_manager
        self.context_manager = context_manager
        self.significant_operations = significant_operations or {
            'open', 'connect', 'execute', 'write',
            'read', 'send', 'recv'
        }
        
    async def extract_code_elements(self, 
                                    filepath: str, 
                                    content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract code elements from a file.
        
        Args:
            filepath: Path to the file
            content: Optional file content
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            # Process file with language detection
            parsed_file = await self.multilang_manager.process_file(filepath, content)
            
            if parsed_file.errors:
                return {'errors': parsed_file.errors}
                
            # Create hierarchy node for file
            file_node = self.hierarchy_manager.add_node(
                path=filepath,
                node_type='file',
                metadata={
                    'language': parsed_file.language,
                    'file_path': filepath
                }
            )
            
            # Extract elements based on language
            extractor = self._get_language_extractor(parsed_file.language)
            if extractor:
                extracted_info = await extractor.extract_elements(
                    parsed_file.content,
                    filepath,
                    self.significant_operations
                )
                
                # Add to hierarchy and context
                await self._process_extracted_elements(
                    extracted_info,
                    file_node,
                    parsed_file.language
                )
                
                return {
                    'language': parsed_file.language,
                    **extracted_info
                }
                
            return {
                'language': parsed_file.language,
                'error': f'No extractor available for {parsed_file.language}'
            }
            
        except Exception as e:
            logging.error(f"Error extracting from {filepath}: {str(e)}")
            sentry_sdk.capture_exception(e)
            return {'error': str(e)}
            
    def _get_language_extractor(self, language: str) -> Optional[BaseLanguageParser]:
        """Get the appropriate extractor for a language."""
        spec = self.multilang_manager.specs.get(language)
        return self.multilang_manager.get_parser(language, spec) if spec else None
        
    async def _process_extracted_elements(self,
                                          elements: Dict[str, Any],
                                          parent_node: Any,
                                          language: str) -> None:
        """
        Process extracted elements and add to hierarchy/context.
        
        Args:
            elements: Extracted code elements
            parent_node: Parent node in hierarchy
            language: Programming language
        """
        for element_type in ['classes', 'functions']:
            for element in elements.get(element_type, []):
                # Add to hierarchy
                element_node = self.hierarchy_manager.add_node(
                    path=f"{parent_node.get_path()}.{element['name']}",
                    node_type=element_type.rstrip('s'),  # Remove plural
                    documentation=element.get('docstring'),
                    metadata={
                        'language': language,
                        'complexity': element.get('complexity'),
                        'type': element_type,
                        **element.get('metadata', {})
                    }
                )
                
                # Add to context manager
                await self.context_manager.add_or_update_segment(
                    f"{parent_node.get_path()}.{element['name']}",
                    element['code'],
                    {
                        'language': language,
                        'complexity': element.get('complexity'),
                        'type': element_type
                    }
                )
                
                # Process nested elements (e.g., methods in classes)
                if element_type == 'classes':
                    await self._process_extracted_elements(
                        {'functions': element.get('methods', [])},
                        element_node,
                        language
                    )

def extract_functions(
    file_path: str,
    content: str,
    significant_operations: Optional[Set[str]] = None,
    multilang_manager: Optional[MultiLanguageManager] = None,
    hierarchy_manager: Optional[CodeHierarchy] = None,
    context_manager: Optional[ContextManager] = None
) -> Dict[str, List[Any]]:
    """
    Extract functions and classes from source code.
    
    Args:
        file_path: Path to the source file
        content: Source code string
        significant_operations: Optional set of operations to track
        multilang_manager: Optional multi-language support manager
        hierarchy_manager: Optional hierarchy manager
        context_manager: Optional context manager
        
    Returns:
        Dictionary containing extracted classes and functions
    """
    # Initialize managers if not provided
    multilang_manager = multilang_manager or MultiLanguageManager()
    hierarchy_manager = hierarchy_manager or CodeHierarchy()
    context_manager = context_manager or ContextManager()
    
    # Create extraction manager
    manager = ExtractionManager(
        multilang_manager,
        hierarchy_manager,
        context_manager,
        significant_operations
    )
    
    # Run extraction asynchronously
    import asyncio
    return asyncio.run(manager.extract_code_elements(file_path, content))