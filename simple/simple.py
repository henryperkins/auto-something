import os
import sys
import shutil
import argparse
import subprocess
import logging
import ast
import hashlib
import json
from dotenv import load_dotenv
from tqdm import tqdm
import openai
import random
import time

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("error.log", mode="a", encoding="utf-8")],
)

# Configuration
DOCSTRING_CONFIG = {"style": "google", "include_types": True, "include_complexity": True, "include_changelog": True}

# Initialize prompt cache
prompt_cache = {}

# Set Azure OpenAI API key and endpoint
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = "2023-05-15"  # Use the appropriate API version

def get_all_files(directory, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []
    files_list = []
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".py"):
                files_list.append(os.path.join(root, file))
    return files_list

def format_with_black(file_content):
    """Attempt to format code using black."""
    try:
        import black

        mode = black.Mode()
        formatted_content = black.format_str(file_content, mode=mode)
        return True, formatted_content
    except Exception as e:
        logging.warning(f"Black formatting failed: {str(e)}")
        return False, file_content

def extract_classes_and_functions_from_ast(tree, content):
    """Extract class and function details from AST."""
    classes = []
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            try:
                class_name = node.name
                class_docstring = ast.get_docstring(node) or ""
                class_code = ast.get_source_segment(content, node)

                # Extract methods within the class
                methods = []
                for body_item in node.body:
                    if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = body_item.name
                        params = [
                            (
                                arg.arg,
                                ast.unparse(arg.annotation) if hasattr(arg, "annotation") and arg.annotation else "Unknown",
                            )
                            for arg in body_item.args.args
                        ]
                        return_type = ast.unparse(body_item.returns) if body_item.returns else "Unknown"
                        method_docstring = ast.get_docstring(body_item) or ""
                        method_code = ast.get_source_segment(content, body_item)

                        methods.append(
                            {
                                "name": method_name,
                                "params": params,
                                "return_type": return_type,
                                "docstring": method_docstring,
                                "code": method_code,
                                "node": body_item,
                            }
                        )

                classes.append(
                    {
                        "name": class_name,
                        "docstring": class_docstring,
                        "code": class_code,
                        "methods": methods,
                        "node": node,
                    }
                )
            except Exception as e:
                logging.warning(f"Error extracting details for class {getattr(node, 'name', 'unknown')}: {e}")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                # Skip methods inside classes
                if isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    continue

                func_name = node.name
                params = [
                    (
                        arg.arg,
                        ast.unparse(arg.annotation) if hasattr(arg, "annotation") and arg.annotation else "Unknown",
                    )
                    for arg in node.args.args
                ]
                return_type = ast.unparse(node.returns) if node.returns else "Unknown"
                docstring = ast.get_docstring(node) or ""
                function_code = ast.get_source_segment(content, node)

                functions.append(
                    {
                        "name": func_name,
                        "params": params,
                        "return_type": return_type,
                        "docstring": docstring,
                        "code": function_code,
                        "node": node,
                    }
                )
            except Exception as e:
                logging.warning(f"Error extracting details for function {getattr(node, 'name', 'unknown')}: {e}")

    return {"classes": classes, "functions": functions}

def extract_functions(file_content):
    """Extract functions and classes with enhanced error handling."""
    try:
        tree = ast.parse(file_content)
        add_parent_info(tree)
        return extract_classes_and_functions_from_ast(tree, file_content)
    except (IndentationError, SyntaxError) as e:
        logging.warning(f"Initial parsing failed, attempting black formatting: {e}")

        success, formatted_content = format_with_black(file_content)
        if success:
            try:
                tree = ast.parse(formatted_content)
                add_parent_info(tree)
                return extract_classes_and_functions_from_ast(tree, formatted_content)
            except (IndentationError, SyntaxError) as e:
                logging.error(f"Parsing failed even after black formatting: {e}")

        return extract_basic_function_info(file_content)

def extract_basic_function_info(content):
    """Extract basic function information when AST parsing fails."""
    functions = []
    import re

    # Simple regex to identify function definitions
    function_pattern = r"(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*?\):"

    matches = re.finditer(function_pattern, content)
    for match in matches:
        func_name = match.group(1)
        # Find the function body (rough approximation)
        start_idx = match.start()
        next_def = content.find("\ndef ", start_idx + 1)
        if next_def == -1:
            next_def = len(content)

        function_code = content[start_idx:next_def].strip()

        functions.append(
            {
                "name": func_name,
                "params": [],  # Cannot reliably parse parameters without AST
                "return_type": "Unknown",
                "docstring": "",
                "code": function_code,
                "node": None,
                "parsing_error": True,  # Flag to indicate this was extracted without AST
            }
        )

    return functions

def exponential_backoff_with_jitter(func, max_retries=5, base_delay=1, max_delay=60):
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except openai.error.RateLimitError as e:
            delay = min(max_delay, base_delay * 2 ** retries + random.uniform(0, 1))
            logging.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
            time.sleep(delay)
            retries += 1
        except Exception as e:
            logging.error(f"Error during OpenAI API call: {e}")
            raise e
    raise Exception("Max retries exceeded")

def get_function_hash(function_content):
    import hashlib
    return hashlib.sha256(function_content.encode("utf-8")).hexdigest()

def format_parameters(params):
    if not params:
        return "None"

    formatted_params = []
    for name, param_type in params:
        if param_type and param_type != "Unknown":
            formatted_params.append(f"{name}: {param_type}")
        else:
            formatted_params.append(name)

    return ", ".join(formatted_params)

def create_error_response(function_name, error_message):
    return {
        "function_name": function_name,
        "complexity_score": None,
        "summary": f"Error occurred during analysis: {error_message}",
        "docstring": "Error: Documentation generation failed",
        "changelog": "Error: Analysis failed",
    }

def analyze_function_with_openai(function_details):
    function_hash = get_function_hash(function_details["code"])
    if function_hash in prompt_cache:
        logging.info(f"Using cached result for function '{function_details['name']}'")
        return prompt_cache[function_hash]

    model_name = "gpt-4o-2024-08-06"  # Use the specified model name for Azure

    function_name = function_details["name"]
    if "class_name" in function_details:
        function_name = f"{function_details['class_name']}.{function_details['name']}"

    if function_details.get("parsing_error"):
        prompt = (
            f"Analyze this function with potential syntax issues:\n"
            f"Function Name: {function_name}\n"
            f"Code (may contain syntax errors):\n"
            f"```python\n{function_details['code']}\n```\n"
            f"Provide a best-effort summary and identify any visible parameters or return values."
        )
    else:
        prompt = (
            f"Analyze and document the following Python function:\n"
            f"Function Name: {function_name}\n"
            f"Parameters: {format_parameters(function_details['params'])}\n"
            f"Return Type: {function_details['return_type']}\n"
            f"Existing Docstring: {function_details['docstring'] or 'None'}\n"
            f"Source Code:\n"
            f"```python\n{function_details['code']}\n```\n"
            f"Requirements:\n"
            f"- Generate a detailed Google-style docstring.\n"
            f"- Include type hints if present.\n"
            f"- Calculate a complexity score between 0 (simple) and 10 (very complex).\n"
            f"- Provide a clear, concise summary.\n"
            f"- Include a changelog entry."
        )

    messages = [{"role": "user", "content": prompt}]

    try:
        completion = exponential_backoff_with_jitter(
            lambda: openai.ChatCompletion.create(
                engine=model_name,  # Use 'engine' instead of 'model' for Azure
                messages=messages,
                max_tokens=1000,
                temperature=0.2,
                n=1,
            )
        )

        response_content = completion.choices[0].message["content"].strip()
        logging.debug(f"OpenAI API response for '{function_name}': {response_content}")

        # Extract the different sections from the response
        # You may need to adjust this parsing based on the actual response format

        # As an example, let's assume the response includes JSON-like content
        try:
            response_data = json.loads(response_content)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to parse the content manually
            response_data = parse_response(response_content)

        result = {
            "function_name": function_name,
            "complexity_score": response_data.get("complexity_score"),
            "summary": response_data.get("summary", "").strip(),
            "docstring": response_data.get("docstring", "").strip(),
            "changelog": response_data.get("changelog", "").strip(),
        }

        if not result["docstring"]:
            result["docstring"] = function_details.get("docstring", "No documentation available.")

        prompt_cache[function_hash] = result
        return result

    except Exception as e:
        logging.error(f"Error in analyze_function_with_openai for '{function_name}': {str(e)}")
        return create_error_response(function_name, str(e))

def parse_response(response_content):
    # Simple parser to extract sections from the response
    # Adjust this function based on the actual format of the model's response
    lines = response_content.splitlines()
    response_data = {}
    current_section = None
    section_content = []

    for line in lines:
        if line.strip().lower().startswith("complexity score"):
            if current_section:
                response_data[current_section] = "\n".join(section_content).strip()
            current_section = "complexity_score"
            section_content = []
            continue
        elif line.strip().lower().startswith("summary"):
            if current_section:
                response_data[current_section] = "\n".join(section_content).strip()
            current_section = "summary"
            section_content = []
            continue
        elif line.strip().lower().startswith("docstring"):
            if current_section:
                response_data[current_section] = "\n".join(section_content).strip()
            current_section = "docstring"
            section_content = []
            continue
        elif line.strip().lower().startswith("changelog"):
            if current_section:
                response_data[current_section] = "\n".join(section_content).strip()
            current_section = "changelog"
            section_content = []
            continue

        if current_section:
            section_content.append(line)

    if current_section:
        response_data[current_section] = "\n".join(section_content).strip()

    return response_data

def update_function_docstring(node, docstring):
    docstring_node = ast.get_docstring(node, clean=False)
    if docstring_node:
        # Replace existing docstring
        node.body[0].value = ast.Constant(value=docstring)
    else:
        # Insert new docstring
        node.body.insert(0, ast.Expr(value=ast.Constant(value=docstring)))
    ast.fix_missing_locations(node)

def update_source_code(filepath, functions_analysis):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        add_parent_info(tree)
        modified = False

        # Map function names to analysis results
        analysis_map = {
            func_analysis["function_name"]: func_analysis
            for func_analysis in functions_analysis
        }

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                if isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    func_name = f"{node.parent.name}.{func_name}"

                analysis = analysis_map.get(func_name)

                if analysis and analysis["docstring"]:
                    update_function_docstring(node, analysis["docstring"])
                    modified = True

        if modified:
            updated_source = ast.unparse(tree)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(updated_source)

            logging.info(f"Updated source code in {filepath}")
            return True
    except Exception as e:
        logging.error(f"Error updating source code in {filepath}: {str(e)}")
        return False

def format_changelog(changelog):
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

def format_docstring(docstring):
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

def create_complexity_indicator(complexity):
    if complexity is None:
        return "❓"

    try:
        complexity = float(complexity)
    except (ValueError, TypeError):
        return "❓"

    indicators = {(0, 3): "🟢 Low", (3, 6): "🟡 Medium", (6, 8): "🟠 High", (8, float("inf")): "🔴 Very High"}

    for (lower, upper), indicator in indicators.items():
        if lower <= complexity < upper:
            return indicator

    return "❓ Unknown"

def write_analysis_to_markdown(results, output_file_path, repo_dir):
    with open(output_file_path, "w", encoding="utf-8") as md_file:
        md_file.write("# Code Documentation and Analysis\n\n")
        md_file.write("## Table of Contents\n\n")
        md_file.write("1. [Summary](#summary)\n")
        md_file.write("2. [Changelog](#changelog)\n")
        md_file.write("3. [Function Analysis](#function-analysis)\n")
        md_file.write("4. [Source Code](#source-code)\n\n")

        # Summary section
        md_file.write("## Summary\n\n")
        for filepath, analysis in results.items():
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f"### {relative_path}\n\n")
            for func_analysis in analysis.get("functions", []):
                if func_analysis is None:
                    continue
                md_file.write(f"- **{func_analysis['function_name']}**: {func_analysis['summary']}\n")
            md_file.write("\n")

        # Changelog section
        md_file.write("## Changelog\n\n")
        for filepath, analysis in results.items():
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f"### {relative_path}\n\n")
            for func_analysis in analysis.get("functions", []):
                if func_analysis is None:
                    continue
                md_file.write(f"#### {func_analysis['function_name']}\n")
                md_file.write(f"{format_changelog(func_analysis['changelog'])}\n\n")

        # Function analysis section
        md_file.write("## Function Analysis\n\n")
        for filepath, analysis in results.items():
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f"### {relative_path}\n\n")

            for func_analysis in analysis.get("functions", []):
                if func_analysis is None:
                    continue
                md_file.write(f"#### {func_analysis['function_name']}\n\n")
                complexity = func_analysis.get("complexity_score", None)
                complexity_indicator = create_complexity_indicator(complexity)
                md_file.write(f"**Complexity Score:** {complexity} {complexity_indicator}\n\n")
                md_file.write("**Documentation:**\n\n")
                md_file.write("```python\n")
                md_file.write(format_docstring(func_analysis.get("docstring", "")))
                md_file.write("\n```\n\n")

        # Source code section
        md_file.write("## Source Code\n\n")
        for filepath, analysis in results.items():
            relative_path = os.path.relpath(filepath, repo_dir)
            md_file.write(f"### {relative_path}\n\n")
            md_file.write("```python\n")
            md_file.write(analysis.get("source_code", ""))
            md_file.write("\n```\n\n")

def add_parent_info(node, parent=None):
    """Recursively add parent information to AST nodes."""
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_info(child, node)

def process_files(files_list, repo_dir, concurrency_limit):
    results = {}

    for filepath in tqdm(files_list, desc="Processing files"):
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            logging.info(f"Processing file: {filepath}")

            # Initial attempt to parse
            try:
                tree = ast.parse(content)
                add_parent_info(tree)
                extracted_data = extract_classes_and_functions_from_ast(tree, content)
            except (IndentationError, SyntaxError) as e:
                logging.warning(f"Initial parsing failed for {filepath}, attempting black formatting: {e}")

                # Attempt to format with black
                success, formatted_content = format_with_black(content)
                if success:
                    try:
                        tree = ast.parse(formatted_content)
                        add_parent_info(tree)
                        extracted_data = extract_classes_and_functions_from_ast(tree, formatted_content)
                    except (IndentationError, SyntaxError) as e:
                        logging.error(f"Parsing failed even after black formatting for {filepath}: {e}")
                        extracted_data = {"functions": [], "classes": []}  # Mark as skipped
                else:
                    extracted_data = {"functions": [], "classes": []}  # Mark as skipped

            # Prepare tasks for OpenAI analysis
            tasks = []
            for function_details in extracted_data["functions"]:
                tasks.append(function_details)

            # Prepare tasks for class methods analysis
            for class_details in extracted_data["classes"]:
                for method_details in class_details["methods"]:
                    # Add class name to method details
                    method_details["class_name"] = class_details["name"]
                    tasks.append(method_details)

            # Analyze functions and methods
            functions_analysis = []
            for function_details in tasks:
                analysis = analyze_function_with_openai(function_details)
                functions_analysis.append(analysis)

            # Update source code if analysis was successful
            if update_source_code(filepath, functions_analysis):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

            results[filepath] = {"source_code": content, "functions": functions_analysis}

        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")
            # Ensure source code is still included even if processing fails
            results[filepath] = {"source_code": content, "functions": [], "error": str(e)}

    return results

def clone_repo(repo_url, clone_dir):
    try:
        if os.path.exists(clone_dir):
            logging.debug(f"Removing existing directory: {clone_dir}")
            shutil.rmtree(clone_dir)

        logging.debug(f"Cloning repository {repo_url} into {clone_dir}")
        subprocess.run(["git", "clone", repo_url, clone_dir], check=True)
        logging.info(f"Cloned repository from {repo_url} into {clone_dir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Analyze a GitHub repository or local directory.")
    parser.add_argument("input_path", help="GitHub Repository URL or Local Directory Path")
    parser.add_argument("output_file", help="File to save Markdown output")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests to OpenAI")
    args = parser.parse_args()

    input_path = args.input_path
    output_file = args.output_file
    concurrency_limit = args.concurrency
    cleanup_needed = False

    if input_path.startswith("http://") or input_path.startswith("https://"):
        repo_url = input_path
        repo_dir = "cloned_repo"
        clone_repo(repo_url, repo_dir)
        cleanup_needed = True
    else:
        repo_dir = input_path
        if not os.path.isdir(repo_dir):
            logging.error(f"The directory {repo_dir} does not exist.")
            sys.exit(1)

    files_list = get_all_files(repo_dir, exclude_dirs=[".git", ".github"])
    results = process_files(files_list, repo_dir, concurrency_limit)
    write_analysis_to_markdown(results, output_file, repo_dir)

    if cleanup_needed:
        shutil.rmtree(repo_dir)
        logging.info(f"Cleaned up cloned repository at {repo_dir}")

if __name__ == "__main__":
    main()
