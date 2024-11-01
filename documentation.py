"""
Documentation module for analyzing Python functions and generating structured documentation.

This module provides functions to analyze Python code using AI services, update source code
with generated documentation, and format various documentation components.
"""

import os
import ast
import json
import logging
import aiohttp
from typing import Dict, Any, Optional
from utils import get_function_hash, exponential_backoff_with_jitter

def format_parameters(params):
    """Format function parameters for documentation.

    Args:
        params (list): A list of tuples, each containing a parameter name and type.

    Returns:
        str: A formatted string representing the parameters, with type annotations if available.
    """
    if not params:
        return "None"

    formatted_params = []
    for name, param_type in params:
        if param_type and param_type != "Unknown":
            formatted_params.append(f"{name}: {param_type}")
        else:
            formatted_params.append(name)

    return ", ".join(formatted_params)

def create_error_response(function_name):
    """Create an error response for failed function analysis.

    Args:
        function_name (str): The name of the function that failed analysis.

    Returns:
        dict: A dictionary containing error messages for summary, docstring, and changelog.
    """
    return {
        "function_name": function_name,
        "complexity_score": None,
        "summary": "Error occurred during analysis",
        "docstring": "Error: Documentation generation failed",
        "changelog": "Error: Analysis failed",
    }

async def analyze_function_with_openai(function_details: Dict[str, Any], service: str) -> Dict[str, Any]:
    """
    Analyze a function using OpenAI's API or Azure OpenAI service.

    Args:
        function_details (dict): A dictionary containing the function's code and metadata.
        service (str): The AI service to use ('openai' or 'azure').

    Returns:
        dict: A dictionary containing the analysis results, including summary, docstring, and changelog.
    """
    # Define the function schema for the expected response
    function_schema = [
        {
            "name": "analyze_and_document_function",
            "description": "Analyzes a Python function and provides structured documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise description of function purpose and behavior",
                    },
                    "docstring": {
                        "type": "string",
                        "description": "Complete Google-style docstring including exceptions and side effects",
                    },
                    "changelog": {
                        "type": "string",
                        "description": "Documentation change history or 'Initial documentation'",
                    },
                },
                "required": ["summary", "docstring", "changelog"],
            },
        }
    ]

    # Process dependency information
    dependencies = function_details.get("dependencies", {})
    dependency_info = []
    
    # Add exception information
    if dependencies.get("raises"):
        dependency_info.append("Raises Exceptions:")
        for exc in dependencies["raises"]:
            dependency_info.append(f"- {exc['exception']}: {exc['context']}")
    
    # Add significant operations
    if dependencies.get("uses"):
        dependency_info.append("\nSignificant Operations:")
        for op in dependencies["uses"]:
            dependency_info.append(f"- {op['operation']}: {op['context']}")
    
    # Add side effects
    if dependencies.get("affects"):
        dependency_info.append("\nSide Effects:")
        for effect in dependencies["affects"]:
            dependency_info.append(f"- Modifies: {effect['attribute']}")
    
    # Add import dependencies
    if dependencies.get("imports"):
        dependency_info.append("\nKey Dependencies:")
        for imp in dependencies["imports"]:
            if imp.get("is_type_hint") or imp["name"] in ["typing", "dataclasses", "abc"]:
                dependency_info.append(f"- {imp['module']}.{imp['name']} (type hints)")

    # Format dependency section
    dependency_section = "\n".join(dependency_info) if dependency_info else "No significant dependencies or side effects"

    # Prepare the prompts for the AI service
    system_prompt = (
        "You are a documentation specialist that analyzes Python functions and generates structured documentation. "
        "Focus on capturing function behavior, dependencies, exceptions, and side effects in the documentation. "
        "Always return responses in the following JSON format:\n"
        "{\n"
        "    'summary': '<concise function description>',\n"
        "    'docstring': '<Google-style docstring>',\n"
        "    'changelog': '<change history>'\n"
        "}"
    )
    
    user_prompt = (
        f"Analyze and document this function:\n\n"
        f"Function Name: {function_details['name']}\n"
        f"Parameters: {format_parameters(function_details['params'])}\n"
        f"Return Type: {function_details['return_type']}\n"
        f"Existing Docstring: {function_details['docstring'] or 'None'}\n\n"
        f"Dependencies and Effects:\n{dependency_section}\n\n"
        f"Source Code:\n"
        f"```python\n{function_details['code']}\n```\n\n"
        f"Requirements:\n"
        f"1. Generate a Google-style docstring\n"
        f"2. Include type hints if present\n"
        f"3. Document any exceptions that may be raised\n"
        f"4. Document any significant side effects\n"
        f"5. Provide a clear, concise summary\n"
        f"6. Include a changelog entry"
    )

    try:
        # Configure API based on the selected service
        if service == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            endpoint = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            model = "gpt-4-0125-preview"
            
        else:  # Azure OpenAI service
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            
            if not all([api_key, endpoint, deployment_name]):
                raise ValueError("Required Azure OpenAI environment variables are not set")
            
            endpoint = f"{endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json"
            }
            model = deployment_name

        async def make_request():
            """Make an asynchronous request to the AI service."""
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "functions": function_schema,
                    "function_call": {"name": "analyze_and_document_function"},
                    "temperature": 0.2
                }
                
                # Remove functions for Azure if not supported
                if service == "azure":
                    del payload["functions"]
                    del payload["function_call"]
                
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_detail = await response.text()
                        raise ValueError(f"API request failed with status {response.status}: {error_detail}")
                    return await response.json()

        # Make API request with exponential backoff
        completion = await exponential_backoff_with_jitter(make_request)
        
        if 'error' in completion:
            logging.error(f"API returned an error: {completion['error']}")
            return create_error_response(function_details["name"])

        response_message = completion.get('choices', [{}])[0].get('message', {})
        if not response_message:
            raise KeyError("Missing 'choices' or 'message' in response")

        # Parse response based on service type
        if service == "openai" and 'function_call' in response_message:
            function_args = json.loads(response_message['function_call']['arguments'])
        else:
            try:
                content = response_message.get('content', '{}')
                function_args = json.loads(content)
            except json.JSONDecodeError:
                function_args = {
                    "summary": response_message.get('content', '').strip(),
                    "docstring": function_details.get("docstring", ""),
                    "changelog": "Initial documentation"
                }

        result = {
            "function_name": function_details["name"],
            "complexity_score": function_details.get("complexity_score"),
            "summary": function_args["summary"].strip(),
            "docstring": format_docstring(function_args["docstring"]),
            "changelog": function_args["changelog"].strip(),
        }

        return result

    except Exception as e:
        logging.error(f"An error occurred during function analysis: {str(e)}")
        return create_error_response(function_details["name"])
    
def update_function_docstring(node: ast.AST, docstring: str):
    """Update the docstring of an AST node.

    This function modifies the AST node of a function to update its docstring,
    ensuring that the new docstring is properly inserted or replaced.

    Args:
        node (ast.AST): The AST node of the function.
        docstring (str): The new docstring to insert.
    """
    if node.body:
        first_node = node.body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
            first_node.value.value = docstring
        else:
            node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
    ast.fix_missing_locations(node)

def update_source_code(filepath: str, functions_analysis: list) -> bool:
    """
    Update source code with new docstrings.

    This function parses the source code file into an AST, updates the docstrings
    of functions based on the analysis results, and writes the updated code back
    to the file if modifications were made.

    Args:
        filepath (str): Path to the source file.
        functions_analysis (list): Analysis results for functions in the file.

    Returns:
        bool: True if the source code was updated, False otherwise.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        modified = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                analysis = next(
                    (a for a in functions_analysis if a["function_name"] == node.name),
                    None
                )

                if analysis and analysis["docstring"]:
                    update_function_docstring(node, analysis["docstring"])
                    modified = True

        if modified:
            updated_source = ast.unparse(tree)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(updated_source)

            logging.info(f"Updated source code in {filepath}")
            return True
            
        return False
        
    except Exception as e:
        logging.error(f"Error updating source code in {filepath}: {str(e)}")
        return False

def format_changelog(changelog: str) -> str:
    """Format changelog entries for documentation.

    Args:
        changelog (str): A string containing changelog entries.

    Returns:
        str: A formatted string with each entry prefixed by a dash.
    """
    if not changelog:
        return "_No changelog available_"

    entries = changelog.split("\n")
    formatted_entries = []

    for entry in entries:
        entry = entry.strip()
        if entry:
            if not entry.startswith("- "):
                entry = f"- {entry}"
            formatted_entries.append(entry)

    return "\n".join(formatted_entries)

def format_docstring(docstring: str) -> str:
    """Format a docstring with proper indentation.

    Args:
        docstring (str): The raw docstring to format.

    Returns:
        str: A formatted docstring with consistent indentation.
    """
    if not docstring:
        return "No documentation available"

    lines = docstring.splitlines()
    if not lines:
        return docstring

    def get_indent(line):
        return len(line) - len(line.lstrip()) if line.strip() else float("inf")

    min_indent = min(get_indent(line) for line in lines if line.strip())
    
    cleaned_lines = []
    for line in lines:
        if line.strip():
            cleaned_lines.append(line[min_indent:] if len(line) >= min_indent else line)
        else:
            cleaned_lines.append("")

    return "\n".join(cleaned_lines)