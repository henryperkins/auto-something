import ast
import logging
import re
from utils import format_with_black

def calculate_cyclomatic_complexity(node):
    """Calculate the cyclomatic complexity of a code segment represented by an AST node."""
    complexity = 1  # Start with a base complexity of 1

    # Iterate through the children of the node
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or, ast.ExceptHandler,
                              ast.Try, ast.With, ast.BoolOp, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            complexity += 1  # Increment complexity for control flow structures

    return complexity

def count_lines(node, content):
    """
    Count the number of code lines in a function or class.
    
    Args:
        node (ast.AST): The AST node of the function or class
        content (str): The source code content
        
    Returns:
        dict: Dictionary containing total_lines, code_lines, comment_lines, and blank_lines
    """
    try:
        source = ast.get_source_segment(content, node)
        if not source:
            return {
                "total_lines": 0,
                "code_lines": 0,
                "comment_lines": 0,
                "blank_lines": 0
            }
            
        lines = source.splitlines()
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - blank_lines - comment_lines
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "blank_lines": blank_lines
        }
    except Exception as e:
        logging.warning(f"Error counting lines: {str(e)}")
        return {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0
        }
    
def extract_dependencies(node, content):
    """
    Extract import dependencies and documentation-relevant information from a function.
    
    Args:
        node (ast.AST): The AST node of the function
        content (str): The source code content
        
    Returns:
        dict: Dictionary containing imports, function calls, and documentation-relevant info
    """
    dependencies = {
        "imports": [],          # Standard import tracking
        "internal_calls": [],   # Internal function calls
        "external_calls": [],   # External function/method calls
        "raises": [],          # Exceptions that might be raised
        "affects": [],         # Side effects and state modifications
        "uses": []            # Significant operations (file, db, network, etc.)
    }
    
    try:
        for child in ast.walk(node):
            # Original import collection
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for name in child.names:
                    module = child.module if isinstance(child, ast.ImportFrom) else name.name
                    dependencies["imports"].append({
                        "module": module,
                        "name": name.name,
                        "alias": name.asname,
                        "is_type_hint": name.name == 'typing' or module == 'typing'
                    })
            
            # Enhanced function call collection
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies["internal_calls"].append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    call_info = f"{child.func.value.id}.{child.func.attr}"
                    dependencies["external_calls"].append(call_info)
                    
                    # Track significant operations
                    significant_ops = ['open', 'connect', 'execute', 'write', 'read', 'send', 'recv']
                    if child.func.attr in significant_ops:
                        dependencies["uses"].append({
                            "operation": child.func.attr,
                            "context": ast.get_source_segment(content, child)
                        })
            
            # Exception tracking
            elif isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Name):
                    dependencies["raises"].append({
                        "exception": child.exc.id,
                        "context": ast.get_source_segment(content, child)
                    })
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    dependencies["raises"].append({
                        "exception": child.exc.func.id,
                        "context": ast.get_source_segment(content, child)
                    })
            
            # Side effects tracking
            elif isinstance(child, ast.Assign):
                if isinstance(child.targets[0], ast.Attribute) and \
                   isinstance(child.targets[0].value, ast.Name) and \
                   child.targets[0].value.id in ['self', 'cls']:
                    dependencies["affects"].append({
                        "attribute": child.targets[0].attr,
                        "context": ast.get_source_segment(content, child)
                    })
                    
    except Exception as e:
        logging.warning(f"Error extracting dependencies: {str(e)}")
        
    return dependencies
    
def generate_docstring_from_deps(func_name: str, doc_deps: dict) -> str:
    """
    Generate docstring sections based on dependency analysis.
    """
    docstring_parts = []
    
    # Add raises section if exceptions are found
    if doc_deps["raises"]:
        docstring_parts.append("Raises:")
        for exc in doc_deps["raises"]:
            docstring_parts.append(f"    {exc['exception']}: Based on {exc['context']}")
    
    # Add requirements section if critical imports exist
    if doc_deps["required_imports"]:
        docstring_parts.append("Requirements:")
        for imp in doc_deps["required_imports"]:
            docstring_parts.append(f"    - {imp['module']}.{imp['name']} for {imp['purpose']}")
    
    # Add important operations section
    if doc_deps["uses"]:
        docstring_parts.append("Important Operations:")
        for op in doc_deps["uses"]:
            docstring_parts.append(f"    - {op['operation']}: {op['context']}")
    
    # Add side effects section
    if doc_deps["affects"]:
        docstring_parts.append("Side Effects:")
        for effect in doc_deps["affects"]:
            docstring_parts.append(f"    - Modifies {effect['attribute']}")
    
    return "\n".join(docstring_parts)

def extract_classes_and_functions_from_ast(tree, content):
    """Extract class and function details from an AST."""
    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            try:
                # Add line count and dependencies to class info
                line_stats = count_lines(node, content)
                dependencies = extract_dependencies(node, content)
                
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node) or "",
                    "code": ast.get_source_segment(content, node),
                    "methods": [],
                    "complexity": calculate_cyclomatic_complexity(node),
                    "line_stats": line_stats,
                    "dependencies": dependencies,
                    "node": node,
                }
                
                # Process methods
                for body_item in node.body:
                    if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_line_stats = count_lines(body_item, content)
                        method_dependencies = extract_dependencies(body_item, content)
                        
                        methods.append({
								    "name": method_name,
								    "params": params,
									 "return_type": return_type,
									 "docstring": method_docstring,
								    "code": method_code,
								    "complexity": method_complexity,
								    "dependencies": extract_dependencies(body_item, content),  # Make sure this is being used
								    "node": body_item,
								})
                
                class_info["methods"] = methods
                classes.append(class_info)
                
            except Exception as e:
                logging.warning(f"Error extracting class {getattr(node, 'name', 'unknown')}: {e}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                line_stats = count_lines(node, content)
                dependencies = extract_dependencies(node, content)
                
                functions.append({
					    "name": func_name,
					    "params": params,
					    "return_type": return_type,
					    "docstring": docstring,
					    "code": function_code,
					    "complexity": function_complexity,
					    "dependencies": extract_dependencies(node, content),  # Make sure this is being used
					    "node": node,
					})
            except Exception as e:
                logging.warning(f"Error extracting function {getattr(node, 'name', 'unknown')}: {e}")

    return {"classes": classes, "functions": functions}

def extract_basic_function_info(content):
    """
    Extract basic function information when AST parsing fails.

    This function uses regular expressions to extract basic information about
    functions from the source code when AST parsing is unsuccessful.

    Args:
        content (str): The source code content.

    Returns:
        list: A list of dictionaries containing basic function information.
    """
    functions = []
    # Regex pattern to match function definitions, including optional 'async' keyword
    function_pattern = r"(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$[^)]*$\s*:"

    # Find all matches of the pattern in the content
    matches = re.finditer(function_pattern, content)
    for match in matches:
        func_name = match.group(1)  # Extract the function name
        start_idx = match.start()  # Start index of the function definition
        next_def = content.find("\ndef ", start_idx + 1)  # Find the start of the next function
        if next_def == -1:
            next_def = len(content)  # If no more functions, go to the end of the content

        # Extract the function code segment
        function_code = content[start_idx:next_def].strip()

        functions.append({
            "name": func_name,
            "params": [],  # Parameters are not extracted in this basic method
            "return_type": "Unknown",  # Return type is not extracted in this basic method
            "docstring": "",  # Docstring is not extracted in this basic method
            "code": function_code,
            "node": None,  # No AST node available
            "parsing_error": True,  # Indicate that this was a fallback extraction
        })

    return functions

def extract_functions(file_content):
    """
    Extract functions and classes with enhanced error handling.

    This function attempts to parse the source code using the AST module and
    extract function and class details. If parsing fails, it attempts to format
    the code with Black and retry parsing.

    Args:
        file_content (str): The source code content of the file.

    Returns:
        dict: A dictionary containing lists of extracted classes and functions.

    Raises:
        Exception: If an error occurs during parsing or extraction.
    """
    try:
        tree = ast.parse(file_content)
        return extract_classes_and_functions_from_ast(tree, file_content)
    except (IndentationError, SyntaxError) as e:
        logging.warning(f"Initial parsing failed, attempting black formatting: {e}")

        # Attempt to format the code with Black
        success, formatted_content = format_with_black(file_content)
        if success:
            try:
                tree = ast.parse(formatted_content)
                return extract_classes_and_functions_from_ast(tree, formatted_content)
            except (IndentationError, SyntaxError) as e:
                logging.error(f"Parsing failed even after black formatting: {e}")

        # Fallback to basic regex-based extraction
        return extract_basic_function_info(file_content)