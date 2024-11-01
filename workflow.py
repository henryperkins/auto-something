"""
Workflow module for processing files and generating documentation.

This module contains functions for processing Python files, analyzing their content
using AI services, and writing the results to a Markdown file.
"""
import os
import ast
import time
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from utils import format_with_black, get_function_hash, create_complexity_indicator
from extract import extract_classes_and_functions_from_ast, calculate_cyclomatic_complexity
from documentation import (
    analyze_function_with_openai,
    update_source_code,
    format_changelog,
    format_docstring,
)

async def analyze_with_semaphore(function_details, semaphore, service, cache):
    """
    Analyze a function with rate limiting semaphore and caching, including complexity.

    Args:
        function_details (dict): Details of the function to analyze.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.
        service (str): The AI service to use ('openai' or 'azure').
        cache (Cache): Cache instance for storing results.

    Returns:
        dict: Analysis results for the function.
    """
    function_hash = get_function_hash(function_details["code"])
    
    if cache and cache.config.enabled:
        cached_result = cache.get(function_hash)
        if cached_result:
            logging.info(f"Cache hit for function {function_details['name']}")
            return cached_result

    async with semaphore:
        try:
            result = await analyze_function_with_openai(function_details, service)
            result["complexity"] = function_details.get("complexity", None)

            if cache and cache.config.enabled:
                try:
                    cache.set(function_hash, result)
                    logging.info(f"Cached analysis result for {function_details['name']}")
                except Exception as e:
                    logging.warning(f"Failed to cache result for {function_details['name']}: {str(e)}")
            
            return result
        except Exception as e:
            logging.error(f"Analysis failed for {function_details['name']}: {str(e)}")
            return {
                "function_name": function_details["name"],
                "complexity_score": None,
                "summary": "Error occurred during analysis",
                "docstring": function_details.get("docstring", "Error: Documentation generation failed"),
                "changelog": "Error: Analysis failed"
            }

async def process_files(files_list, repo_dir, concurrency_limit, service, cache=None):
    results = {}
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    with tqdm(total=len(files_list), desc="Processing files") as pbar:
        for filepath in files_list:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                logging.info(f"Processing file: {filepath}")

                file_hash = get_function_hash(content)
                if cache and cache.config.enabled:
                    cached_result = cache.get_module(filepath, file_hash)
                    if cached_result:
                        logging.info(f"Cache hit for module: {filepath}")
                        results[filepath] = cached_result
                        pbar.update(1)
                        continue
                    else:
                        logging.info(f"Cache miss for module: {filepath}")

                # Perform analysis
                tree = ast.parse(content)
                extracted_data = extract_classes_and_functions_from_ast(tree, content)

                all_functions = extracted_data["functions"] + [
                    method for class_info in extracted_data["classes"] 
                    for method in class_info["methods"]
                ]

                tasks = []
                for function_details in all_functions:
                    function_details["complexity"] = calculate_cyclomatic_complexity(function_details["node"])
                    task = analyze_with_semaphore(function_details, semaphore, service, cache)
                    tasks.append(task)

                functions_analysis = await asyncio.gather(*tasks)

                if update_source_code(filepath, functions_analysis):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                module_docstring = ast.get_docstring(tree) if 'tree' in locals() else None
                
                result = {
                    "source_code": content,
                    "functions": functions_analysis,
                    "module_summary": module_docstring or "No module documentation available",
                    "timestamp": time.time()
                }

                if cache and cache.config.enabled:
                    cache.set_module(filepath, file_hash, result)

                results[filepath] = result
                pbar.update(1)

            except Exception as e:
                logging.error(f"Error processing file {filepath}: {e}")
                results[filepath] = {
                    "source_code": content if 'content' in locals() else "",
                    "functions": [],
                    "module_summary": f"Error: {str(e)}",
                    "error": str(e)
                }
                pbar.update(1)

    return results

def write_analysis_to_markdown(results: Dict[str, Dict], output_file_path: str, repo_dir: str):
    """
    Write analysis results to a markdown file with enhanced structure and metrics.
    
    Args:
        results: Dictionary containing analysis results
        output_file_path: Path to output markdown file
        repo_dir: Root directory of the repository
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as md_file:
            # Write header and table of contents
            md_file.write("# Code Documentation Analysis\n\n")
            md_file.write("## Table of Contents\n\n")
            
            # Generate TOC
            for filepath in results.keys():
                relative_path = os.path.relpath(filepath, repo_dir)
                file_anchor = relative_path.replace('/', '-').replace('.', '-')
                md_file.write(f"- [{relative_path}](#{file_anchor})\n")
            
            md_file.write("\n---\n\n")
            
            # Process each file
            for filepath, analysis in results.items():
                relative_path = os.path.relpath(filepath, repo_dir)
                file_anchor = relative_path.replace('/', '-').replace('.', '-')
                
                md_file.write(f"## {relative_path} {{{file_anchor}}}\n\n")
                
                # Module summary
                md_file.write("### Summary\n")
                md_file.write(f"{analysis.get('module_summary', 'No module summary available')}\n\n")
                
                # Functions and Methods
                md_file.write("### Functions and Methods\n\n")
                if analysis.get("functions"):
                    for func_analysis in analysis["functions"]:
                        name = func_analysis.get('function_name', 'Unknown')
                        md_file.write(f"#### {name}\n\n")
                        
                        # Metrics table
                        md_file.write("**Metrics:**\n\n")
                        md_file.write("| Metric | Value |\n")
                        md_file.write("|--------|-------|\n")
                        
                        # Complexity
                        complexity = func_analysis.get('complexity', 0)
                        complexity_indicator = create_complexity_indicator(complexity)
                        md_file.write(f"| Complexity | {complexity} {complexity_indicator} |\n")
                        
                        # Line counts
                        line_stats = func_analysis.get('line_stats', {})
                        if line_stats:
                            md_file.write(f"| Total Lines | {line_stats.get('total_lines', 0)} |\n")
                            md_file.write(f"| Code Lines | {line_stats.get('code_lines', 0)} |\n")
                            md_file.write(f"| Comment Lines | {line_stats.get('comment_lines', 0)} |\n")
                            md_file.write(f"| Blank Lines | {line_stats.get('blank_lines', 0)} |\n")
                        
                        md_file.write("\n")
                        
                        # Dependencies section
                        dependencies = func_analysis.get('dependencies', {})
                        if dependencies:
                            md_file.write("**Dependencies:**\n\n")
                            
                            # Imports
                            if dependencies.get('imports'):
                                md_file.write("*Imports:*\n")
                                for imp in dependencies['imports']:
                                    module = imp.get('module', '')
                                    name = imp.get('name', '')
                                    alias = imp.get('alias', '')
                                    if module:
                                        md_file.write(f"- from {module} import {name}")
                                    else:
                                        md_file.write(f"- import {name}")
                                    if alias:
                                        md_file.write(f" as {alias}")
                                    md_file.write("\n")
                                md_file.write("\n")
                            
                            # Function calls
                            if dependencies.get('internal_calls'):
                                md_file.write("*Internal Function Calls:*\n")
                                for call in dependencies['internal_calls']:
                                    md_file.write(f"- {call}\n")
                                md_file.write("\n")
                            
                            if dependencies.get('external_calls'):
                                md_file.write("*External Function Calls:*\n")
                                for call in dependencies['external_calls']:
                                    md_file.write(f"- {call}\n")
                                md_file.write("\n")
                        
                        # Function summary
                        summary = func_analysis.get('summary', 'No description available')
                        if isinstance(summary, dict):
                            summary = json.dumps(summary, indent=2)
                        md_file.write("**Description:**\n\n")
                        md_file.write(f"{summary}\n\n")
                        
                        # Function documentation
                        docstring = func_analysis.get('docstring', '')
                        if docstring:
                            md_file.write("**Documentation:**\n\n")
                            md_file.write("```python\n")
                            md_file.write(docstring)
                            md_file.write("\n```\n\n")
                        
                        # Source code
                        md_file.write("**Source Code:**\n\n")
                        md_file.write("```python\n")
                        md_file.write(func_analysis.get('code', '# Code not available'))
                        md_file.write("\n```\n\n")
                        
                        md_file.write("---\n\n")
                else:
                    md_file.write("*No functions or methods found in this file*\n\n")
                
                md_file.write("---\n\n")

        logging.info(f"Successfully wrote analysis to Markdown file: {output_file_path}")

    except Exception as e:
        logging.error(f"Error writing to Markdown file {output_file_path}: {str(e)}")
        raise