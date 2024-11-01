"""
Workflow module for processing files and generating documentation.

This module orchestrates the process of analyzing code files, generating
documentation using AI services, and writing results to markdown files.
It coordinates between different components while maintaining proper error
handling and async operations.

Functions:
    process_files: Process files and generate documentation.
    write_analysis_to_markdown: Write analysis results to a markdown file.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm
import sentry_sdk

from utils import get_function_hash
from documentation import DocumentationGenerator
from context_manager import ContextManager, ContextWindowManager
from dependency_analyzer import DependencyAnalyzer
from metadata_manager import MetadataManager
from multilang import MultiLanguageManager
from hierarchy import CodeHierarchy
from config import Config
from exceptions import WorkflowError

async def process_files(
    files_list: List[str],
    repo_dir: str,
    config: Config,
    multilang_manager: MultiLanguageManager,
    hierarchy_manager: CodeHierarchy,
    context_manager: ContextManager,
    window_manager: ContextWindowManager,
    cache: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process files and generate documentation.
    
    Args:
        files_list: List of files to process
        repo_dir: Repository directory path
        config: Configuration object
        multilang_manager: Multi-language manager instance
        hierarchy_manager: Hierarchy manager instance
        context_manager: Context manager instance
        window_manager: Context window manager instance
        cache: Optional cache instance
        
    Returns:
        Dictionary containing processing results
    """
    results = {}
    semaphore = asyncio.Semaphore(config.concurrency.limit)
    
    # Initialize documentation generator
    doc_generator = DocumentationGenerator(
        service=config.service,
        context_manager=context_manager,
        hierarchy_manager=hierarchy_manager,
        multilang_manager=multilang_manager,
        metadata_manager=MetadataManager(db_path=os.path.join(repo_dir, 'metadata.db'))
    )

    with tqdm(total=len(files_list), desc="Processing files") as pbar:
        for filepath in files_list:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                logging.info(f"Processing file: {filepath}")

                # Check cache first
                file_hash = get_function_hash(content)
                if cache and cache.config.enabled:
                    cached_result = await cache.get(file_hash)
                    if cached_result:
                        results[filepath] = cached_result
                        pbar.update(1)
                        continue

                # Process file
                async with semaphore:
                    file_result = await doc_generator.analyze_code_file(
                        filepath,
                        content,
                        config.dict()
                    )

                if file_result.get('errors'):
                    logging.warning(f"Errors processing {filepath}: {file_result['errors']}")

                # Update cache
                if cache and cache.config.enabled:
                    await cache.set(
                        file_hash,
                        file_result,
                        metadata={
                            'filepath': filepath,
                            'language': file_result.get('language'),
                            'hierarchy_path': hierarchy_manager.get_node(filepath).get_path()
                        }
                    )

                results[filepath] = file_result

            except Exception as e:
                logging.error(f"Error processing file {filepath}: {str(e)}")
                sentry_sdk.capture_exception(e)
                results[filepath] = {
                    'error': str(e),
                    'content': content if 'content' in locals() else ""
                }

            finally:
                pbar.update(1)

    # Generate cross-references
    cross_references = await doc_generator.generate_cross_references()
    results['cross_references'] = cross_references

    return results

def write_analysis_to_markdown(
    results: Dict[str, Dict],
    output_file_path: str,
    repo_dir: str
) -> None:
    """
    Write analysis results to a markdown file.

    Args:
        results: Analysis results to write
        output_file_path: Path to output file
        repo_dir: Repository directory path

    Raises:
        WorkflowError: If writing fails
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as md_file:
            # Write header and TOC
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
                
                # Write module summary
                md_file.write("### Module Summary\n")
                md_file.write(f"{analysis.get('module_summary', 'No module summary available')}\n\n")
                
                # Write functions and methods
                md_file.write("### Functions and Methods\n\n")
                
                if analysis.get("functions"):
                    for func_analysis in analysis["functions"]:
                        name = func_analysis.get('function_name', 'Unknown')
                        md_file.write(f"#### {name}\n\n")
                        
                        # Write metrics
                        complexity = func_analysis.get('complexity', 0)
                        complexity_indicator = create_complexity_indicator(complexity)
                        
                        md_file.write("**Metrics:**\n\n")
                        md_file.write("| Metric | Value |\n")
                        md_file.write("|--------|-------|\n")
                        md_file.write(f"| Complexity | {complexity} {complexity_indicator} |\n\n")
                        
                        # Write summary and documentation
                        summary = func_analysis.get('summary', 'No description available')
                        md_file.write("**Description:**\n\n")
                        md_file.write(f"{summary}\n\n")
                        
                        docstring = func_analysis.get('docstring', '')
                        if docstring:
                            md_file.write("**Documentation:**\n\n")
                            md_file.write("```python\n")
                            md_file.write(format_docstring(docstring))
                            md_file.write("\n```\n\n")
                        
                        # Write code
                        md_file.write("**Source Code:**\n\n")
                        md_file.write("```python\n")
                        md_file.write(func_analysis.get('code', '# Code not available'))
                        md_file.write("\n```\n\n")
                        
                        # Write changelog
                        changelog = func_analysis.get('changelog')
                        if changelog:
                            md_file.write("**Changelog:**\n\n")
                            md_file.write(format_changelog(changelog))
                            md_file.write("\n\n")
                        
                        md_file.write("---\n\n")
                else:
                    md_file.write("*No functions or methods found in this file*\n\n")
                
                md_file.write("---\n\n")

        logging.info(f"Successfully wrote analysis to {output_file_path}")

    except Exception as e:
        error_msg = f"Error writing to markdown file: {str(e)}"
        logging.error(error_msg)
        sentry_sdk.capture_exception(e)
        raise WorkflowError(error_msg)