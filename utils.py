import os
import sys
import shutil
import logging
import subprocess
import hashlib
import random
import asyncio
from typing import Optional
from openai import OpenAIError

def get_all_files(directory, exclude_dirs=None):
    """
    Retrieve all Python files from a specified directory, excluding certain directories.

    This function traverses the given directory recursively and collects paths to all Python
    files, while excluding any directories specified in the `exclude_dirs` list.

    Args:
        directory (str): The root directory to search for Python files.
        exclude_dirs (list, optional): A list of directory names to exclude from the search.
            Defaults to None, which means no directories are excluded.

    Returns:
        list: A list of file paths to Python files found in the directory, excluding specified directories.

    Raises:
        ValueError: If the provided directory does not exist or is not accessible.
    """
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
    """
    Format Python code using the Black code formatter.

    This function attempts to format the provided Python source code string using Black,
    a popular code formatter. If formatting is successful, the formatted code is returned.

    Args:
        file_content (str): The Python source code to format.

    Returns:
        tuple: A tuple containing a boolean indicating success and the formatted content.
            If formatting fails, the original content is returned.

    Raises:
        ImportError: If the Black library is not installed.
        Exception: If any other error occurs during formatting.
    """
    try:
        import black
        mode = black.Mode()
        formatted_content = black.format_str(file_content, mode=mode)
        return True, formatted_content
    except Exception as e:
        logging.warning(f"Black formatting failed: {str(e)}")
        return False, file_content

def get_function_hash(function_content):
    """
    Generate a SHA-256 hash for a function's content.

    This function computes a SHA-256 hash of the provided function content string,
    which can be used to uniquely identify the function based on its code.

    Args:
        function_content (str): The content of the function to hash.

    Returns:
        str: A SHA-256 hash of the function content.
    """
    return hashlib.sha256(function_content.encode("utf-8")).hexdigest()

def clone_repo(repo_url, clone_dir):
    """
    Clone a Git repository to a specified local directory.

    This function uses the Git command-line tool to clone a repository from the given URL
    into the specified local directory. If the directory already exists, it is removed before cloning.

    Args:
        repo_url (str): The URL of the Git repository to clone.
        clone_dir (str): The local directory to clone the repository into.

    Raises:
        SystemExit: If the cloning process fails due to a Git error or other issues.
    """
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

async def exponential_backoff_with_jitter(func, max_retries=5, base_delay=1, max_delay=60):
    """
    Execute a function with exponential backoff and jitter for retries.

    This asynchronous function attempts to execute the provided coroutine function,
    applying an exponential backoff strategy with jitter to handle transient errors,
    such as rate limiting. The delay between retries increases exponentially, with
    a random jitter added to prevent synchronized retries.

    Args:
        func (coroutine): The coroutine function to execute.
        max_retries (int, optional): Maximum number of retries before giving up. Defaults to 5.
        base_delay (int, optional): Initial delay in seconds before the first retry. Defaults to 1.
        max_delay (int, optional): Maximum delay in seconds between retries. Defaults to 60.

    Returns:
        Any: The result of the function if successful.

    Raises:
        Exception: If the maximum number of retries is exceeded without success.
    """
    retries = 0
    while retries < max_retries:
        try:
            return await func()
        except OpenAIError as e:
            if e.http_status == 429:  # Check for rate limit error
                delay = min(max_delay, base_delay * 2 ** retries + random.uniform(0, 1))
                logging.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                retries += 1
            else:
                raise e
    raise Exception("Max retries exceeded")

def setup_logging():
    """
    Configure logging settings for the application.

    This function sets up the logging configuration to output log messages to both
    the console and a log file named 'error.log'. The log level is set to INFO, and
    the log format includes the timestamp, log level, and message.

    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("error.log", mode="a", encoding="utf-8")
        ],
    )

def create_complexity_indicator(complexity: Optional[int]) -> str:
    """Create a visual indicator for code complexity.

    Args:
        complexity (Optional[int]): An optional integer representing complexity.

    Returns:
        str: A string representing the complexity level with an emoji.
    """
    if complexity is None:
        return "❓ Unknown"

    # Define complexity ranges and their corresponding indicators
    indicators = {
        (0, 3): "🟢 Low",
        (3, 6): "🟡 Medium",
        (6, 8): "🟠 High",
        (8, float("inf")): "🔴 Very High"
    }

    # Determine the appropriate indicator based on the complexity score
    for (lower, upper), indicator in indicators.items():
        if lower <= complexity < upper:
            return indicator

    return "❓ Unknown"