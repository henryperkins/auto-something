"""
Hierarchy Management Module.

This module provides functionality for organizing code elements into a hierarchical
structure, supporting efficient navigation and documentation organization. It implements
a tree-like structure that represents the relationships between different code elements
while maintaining references to their documentation.

Classes:
    HierarchyNode: Represents a single node in the hierarchy tree.
    CodeHierarchy: Manages the complete hierarchical structure of code elements.
    HierarchyBuilder: Constructs hierarchical structures from code analysis results.
"""

import ast
import logging
from typing import Dict, List, Optional, Set, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json

@dataclass
class HierarchyNode:
    """
    Represents a node in the code hierarchy tree.
    
    Each node can represent a module, class, function, or other code element,
    storing relevant metadata and maintaining parent-child relationships.
    
    Attributes:
        name: Name of the code element
        element_type: Type of code element (module, class, function, etc.)
        path: File path or qualified name
        documentation: Associated documentation
        metadata: Additional metadata about the element
        children: Child nodes in the hierarchy
        parent: Parent node reference
    """
    name: str
    element_type: str
    path: str
    documentation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['HierarchyNode'] = field(default_factory=list)
    parent: Optional['HierarchyNode'] = None

    def add_child(self, child: 'HierarchyNode') -> None:
        """Add a child node to this node."""
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: 'HierarchyNode') -> None:
        """Remove a child node from this node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)

    def find_child(self, name: str) -> Optional['HierarchyNode']:
        """Find a direct child node by name."""
        return next((child for child in self.children if child.name == name), None)

    def get_path(self) -> str:
        """Get the full path from root to this node."""
        if self.parent:
            parent_path = self.parent.get_path()
            return f"{parent_path}.{self.name}" if parent_path else self.name
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation."""
        return {
            'name': self.name,
            'type': self.element_type,
            'path': self.path,
            'documentation': self.documentation,
            'metadata': self.metadata,
            'children': [child.to_dict() for child in self.children]
        }

    def __repr__(self) -> str:
        return f"HierarchyNode(name='{self.name}', type='{self.element_type}', children={len(self.children)})"

class CodeHierarchy:
    """
    Manages the hierarchical structure of code elements.
    
    This class maintains the complete hierarchy tree and provides methods for
    navigating, searching, and manipulating the structure.
    """
    
    def __init__(self):
        """Initialize the code hierarchy."""
        self.root = HierarchyNode("root", "root", "")
        self._index: Dict[str, HierarchyNode] = {}
        self.modified = False
        
    def add_node(self, path: str, node_type: str, documentation: str = None, 
                metadata: Dict[str, Any] = None) -> HierarchyNode:
        """
        Add a new node to the hierarchy.
        
        Args:
            path: Dot-separated path to the node (e.g., 'module.class.function')
            node_type: Type of the code element
            documentation: Optional documentation string
            metadata: Optional metadata dictionary
            
        Returns:
            The created node
            
        Raises:
            ValueError: If the path is invalid
        """
        if not path:
            raise ValueError("Path cannot be empty")
            
        parts = path.split('.')
        current = self.root
        
        # Create or traverse to parent nodes
        for i, part in enumerate(parts[:-1]):
            next_node = current.find_child(part)
            if not next_node:
                next_node = HierarchyNode(
                    name=part,
                    element_type='namespace',
                    path='.'.join(parts[:i+1])
                )
                current.add_child(next_node)
            current = next_node
            
        # Create the actual node
        node = HierarchyNode(
            name=parts[-1],
            element_type=node_type,
            path=path,
            documentation=documentation,
            metadata=metadata or {}
        )
        current.add_child(node)
        
        # Update index
        self._index[path] = node
        self.modified = True
        
        return node
        
    def get_node(self, path: str) -> Optional[HierarchyNode]:
        """Get a node by its path."""
        return self._index.get(path)
        
    def remove_node(self, path: str) -> bool:
        """
        Remove a node and its children from the hierarchy.
        
        Args:
            path: Path to the node to remove
            
        Returns:
            True if the node was removed, False if not found
        """
        node = self._index.get(path)
        if not node or not node.parent:
            return False
            
        # Remove from parent
        node.parent.remove_child(node)
        
        # Update index by removing node and all its children
        to_remove = set()
        stack = [node]
        while stack:
            current = stack.pop()
            to_remove.add(current.get_path())
            stack.extend(current.children)
            
        for path in to_remove:
            self._index.pop(path, None)
            
        self.modified = True
        return True
        
    def move_node(self, source_path: str, target_path: str) -> bool:
        """
        Move a node to a new location in the hierarchy.
        
        Args:
            source_path: Current path of the node
            target_path: New path for the node
            
        Returns:
            True if the move was successful, False otherwise
        """
        node = self._index.get(source_path)
        if not node:
            return False
            
        # Get or create target parent path
        target_parent_path = '.'.join(target_path.split('.')[:-1])
        if target_parent_path:
            target_parent = self.get_node(target_parent_path)
            if not target_parent:
                return False
        else:
            target_parent = self.root
            
        # Remove from old parent
        if node.parent:
            node.parent.remove_child(node)
            
        # Add to new parent
        target_parent.add_child(node)
        
        # Update paths in the index
        old_prefix = source_path
        new_prefix = target_path
        updates = {}
        
        for path, indexed_node in self._index.items():
            if path == old_prefix or path.startswith(old_prefix + '.'):
                new_path = new_prefix + path[len(old_prefix):]
                updates[path] = (indexed_node, new_path)
                
        for old_path, (node, new_path) in updates.items():
            del self._index[old_path]
            self._index[new_path] = node
            node.path = new_path
            
        self.modified = True
        return True
        
    def search(self, query: str, node_type: Optional[str] = None) -> List[HierarchyNode]:
        """
        Search for nodes matching the given criteria.
        
        Args:
            query: Search string (supports wildcards *)
            node_type: Optional filter by node type
            
        Returns:
            List of matching nodes
        """
        results = []
        
        # Convert query to regex pattern
        import re
        pattern = re.escape(query).replace('\\*', '.*')
        regex = re.compile(pattern)
        
        for path, node in self._index.items():
            if regex.search(path) and (not node_type or node.element_type == node_type):
                results.append(node)
                
        return results
        
    def iterate_nodes(self, root: Optional[HierarchyNode] = None) -> Iterator[HierarchyNode]:
        """
        Iterate through all nodes in the hierarchy.
        
        Args:
            root: Optional starting node (defaults to hierarchy root)
            
        Yields:
            HierarchyNode objects in depth-first order
        """
        stack = [(root or self.root, 0)]
        while stack:
            node, depth = stack.pop()
            yield node
            stack.extend((child, depth + 1) for child in reversed(node.children))
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire hierarchy to a dictionary representation."""
        return self.root.to_dict()
        
    def save_to_file(self, filepath: str) -> None:
        """
        Save the hierarchy to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CodeHierarchy':
        """
        Load a hierarchy from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            CodeHierarchy instance
        """
        def build_node(data: Dict[str, Any], parent: Optional[HierarchyNode] = None) -> HierarchyNode:
            node = HierarchyNode(
                name=data['name'],
                element_type=data['type'],
                path=data['path'],
                documentation=data.get('documentation'),
                metadata=data.get('metadata', {})
            )
            if parent:
                parent.add_child(node)
            for child_data in data.get('children', []):
                build_node(child_data, node)
            return node
            
        hierarchy = cls()
        with open(filepath) as f:
            data = json.load(f)
        
        # Rebuild the hierarchy
        hierarchy.root = build_node(data)
        
        # Rebuild the index
        hierarchy._rebuild_index()
        
        return hierarchy
        
    def _rebuild_index(self) -> None:
        """Rebuild the path index from the current hierarchy."""
        self._index.clear()
        for node in self.iterate_nodes():
            if node != self.root:
                self._index[node.get_path()] = node

class HierarchyBuilder:
    """
    Constructs hierarchical structures from code analysis results.
    
    This class provides methods to build and update hierarchy structures
    based on code analysis data, maintaining relationships between elements.
    """
    
    def __init__(self):
        """Initialize the hierarchy builder."""
        self.hierarchy = CodeHierarchy()
        self.current_module: Optional[str] = None
        
    def process_module(self, module_path: str, tree: ast.AST) -> None:
        """
        Process a module and update the hierarchy.
        
        Args:
            module_path: Path to the module file
            tree: AST of the module
        """
        # Convert file path to module path
        module_name = Path(module_path).stem
        self.current_module = module_name
        
        # Add module node
        module_doc = ast.get_docstring(tree)
        self.hierarchy.add_node(
            module_name,
            'module',
            documentation=module_doc,
            metadata={'file_path': module_path}
        )
        
        # Process module contents
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, parent_path=module_name)
                
    def _process_class(self, node: ast.ClassDef) -> None:
        """Process a class definition and add it to the hierarchy."""
        if not self.current_module:
            return
            
        class_path = f"{self.current_module}.{node.name}"
        class_doc = ast.get_docstring(node)
        
        # Add class node
        self.hierarchy.add_node(
            class_path,
            'class',
            documentation=class_doc,
            metadata={
                'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
            }
        )
        
        # Process methods
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                self._process_function(child, parent_path=class_path)
                
    def _process_function(self, node: ast.FunctionDef, parent_path: str) -> None:
        """Process a function definition and add it to the hierarchy."""
        function_path = f"{parent_path}.{node.name}"
        function_doc = ast.get_docstring(node)
        
        # Add function node
        self.hierarchy.add_node(
            function_path,
            'function',
            documentation=function_doc,
            metadata={
                'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'args': [arg.arg for arg in node.args.args]
            }
        )