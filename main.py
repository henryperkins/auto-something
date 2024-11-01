# main.py
"""
Main module for the documentation generator.

This module serves as the entry point for the application, handling command-line
arguments, configuration loading, repository management, and the workflow for
processing files and generating documentation.
"""

import os
import sys
import asyncio
import argparse
import shutil
from pathlib import Path
from typing import Optional

from config import Config, create_default_config
from cache import Cache
from validation import (
    CLIArguments,
    validate_input_files,
    validate_git_repository
)
from utils import setup_logging, get_all_files, clone_repo
from workflow import process_files, write_analysis_to_markdown
from dotenv import load_dotenv  # Import dotenv to load environment variables

def validate_service_configuration(service: str) -> None:
    """
    Validate that required environment variables are set for the selected service.
    
    Args:
        service: The selected service ('openai' or 'azure')
        
    Raises:
        ValueError: If required environment variables are missing
    """
    if service == "azure":
        required_vars = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            raise ValueError(
                f"Missing required Azure OpenAI environment variables: {', '.join(missing_vars)}\n"
                f"Please set these in your .env file or environment."
            )
            
    elif service == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Missing OPENAI_API_KEY environment variable.\n"
                "Please set this in your .env file or environment."
            )

def create_arg_parser():
    """Create and configure the argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser with all necessary options.
    """
    parser = argparse.ArgumentParser(description="Analyze a GitHub repository or local directory.")
    
    # Option to create a default configuration file
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit"
    )
    
    # Option to clear the cache
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache and exit"
    )
    
    # Input path and output file are optional if --create-config or --clear-cache is used
    parser.add_argument(
        "input_path",
        nargs="?",
        help="GitHub Repository URL or Local Directory Path"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        help="File to save Markdown output"
    )
    
    # Concurrency level for processing
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests to OpenAI"
    )
    
    # AI service selection
    parser.add_argument(
        "--service",
        choices=["openai", "azure"],
        default="openai",
        help="Select the AI service to use: 'openai' or 'azure'"
    )
    
    # Path to the configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    return parser

async def main():
    """Main entry point for the documentation generator.

    This function orchestrates the entire workflow, including argument parsing,
    configuration loading, repository handling, file processing, and documentation
    generation. It uses asynchronous programming to handle concurrent tasks efficiently.
    """
    # Set up argument parser
    parser = create_arg_parser()
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Validate service configuration
    try:
        validate_service_configuration(args.service)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nWould you like to create/edit the .env file now? (y/n)")
        if input().lower() == 'y':
            env_path = '.env'
            if os.path.exists(env_path):
                print(f"\nCurrent {env_path} content:")
                with open(env_path, 'r') as f:
                    print(f.read())
            
            print("\nPlease edit the .env file with your API keys and try again.")
            if not os.path.exists(env_path):
                with open(env_path, 'w') as f:
                    f.write("""# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
""")
                print(f"Created {env_path} file. Please edit it with your API keys.")
        sys.exit(1)

    # Handle config creation request
    if args.create_config:
        config_path = args.config if args.config else "config.yaml"
        create_default_config(config_path)
        print(f"Created default configuration file '{config_path}'")
        print("Please edit it with your API keys and preferences")
        sys.exit(0)

    # Load configuration
    config = Config.load(args.config)

    # Initialize cache
    cache = Cache(config.cache)

    # Handle cache clearing request
    if args.clear_cache:
        cache.clear()
        print("Cache cleared successfully.")
        sys.exit(0)

    # Check for required arguments
    if not args.input_path or not args.output_file:
        parser.error("input_path and output_file are required unless --create-config or --clear-cache is used")

    # Check for config file and create if it doesn't exist
    if not os.path.exists(args.config):
        print(f"Configuration file '{args.config}' not found.")
        print("Would you like to create a default configuration file? (y/n)")
        if input().lower() == 'y':
            create_default_config(args.config)
            print(f"Created default configuration file '{args.config}'")
            print("Please edit it with your API keys and preferences")
            sys.exit(0)
        else:
            print("Cannot proceed without configuration file.")
            sys.exit(1)

    # Setup logging
    setup_logging()

    try:
        # Validate CLI arguments
        cli_args = CLIArguments(
            input_path=args.input_path,
            output_file=args.output_file,
            concurrency=args.concurrency,
            service=args.service,
            config_file=args.config
        )
    except ValueError as e:
        print(f"Error in command line arguments: {str(e)}")
        sys.exit(1)

    # Load configuration
    try:
        config = Config.load(cli_args.config_file)
        config_errors = config.validate()
        if config_errors:
            print("Configuration errors:")
            for error in config_errors:
                print(f"  - {error}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)

    cleanup_needed = False
    repo_dir = None

    try:
        # Handle repository cloning if needed
        if cli_args.input_path.startswith(("http://", "https://")):
            try:
                # Validate the Git repository URL
                repo_validation = validate_git_repository(cli_args.input_path)
                repo_dir = "cloned_repo"
                # Clone the repository to a local directory
                clone_repo(str(repo_validation.url), repo_dir)
                cleanup_needed = True
            except ValueError as e:
                print(f"Error with repository URL: {str(e)}")
                sys.exit(1)
        else:
            # Use the local directory as the repository directory
            repo_dir = cli_args.input_path
            if not os.path.isdir(repo_dir):
                print(f"Error: The directory {repo_dir} does not exist.")
                sys.exit(1)

        # Get list of Python files
        try:
            # Retrieve all Python files from the repository directory
            files_list = get_all_files(repo_dir, exclude_dirs=config.exclude_dirs)
            # Validate the retrieved files
            validated_files = validate_input_files(files_list)
        except ValueError as e:
            print(f"Error with input files: {str(e)}")
            sys.exit(1)
        
        # Process files and generate documentation
        try:
            # Asynchronously process the validated files to generate documentation
            results = await process_files(
                validated_files,
                repo_dir,
                config.concurrency_limit,
                cli_args.service,
                cache
            )
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            sys.exit(1)
        
        # Write results to markdown
        try:
            # Write the analysis results to a Markdown file
            write_analysis_to_markdown(results, cli_args.output_file, repo_dir)
            print(f"\nAnalysis complete! Documentation has been written to {cli_args.output_file}")
            
            # Print cache statistics if caching is enabled
            if config.cache.enabled:
                stats = cache.get_stats()
                print("\nCache Statistics:")
                print(f"  Total Size: {stats['total_size_mb']:.2f} MB")
                print(f"  Entry Count: {stats.get('function_count', 0)}")
                print(f"  Module Count: {stats.get('module_count', 0)}")
                print(f"  Average Entry Age: {stats['avg_age_hours']:.1f} hours")
                
        except Exception as e:
            print(f"Error writing output: {str(e)}")
            sys.exit(1)

    finally:
        # Clean up cloned repository if necessary
        if cleanup_needed and repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
            print(f"Cleaned up cloned repository at {repo_dir}")

if __name__ == "__main__":
    asyncio.run(main())