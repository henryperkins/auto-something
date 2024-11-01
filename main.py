# main.py

"""
Main module for the documentation generator.

This module serves as the entry point for the application, orchestrating the
documentation generation process across multiple languages and managing hierarchical
organization of code documentation with optimized context handling.

Functions:
    main: Main entry point for the application.
    main_async: Asynchronous main function implementing the core application logic.
    create_arg_parser: Create and configure the argument parser.
    process_repository: Process a repository and generate documentation.
    clone_repository: Clone a Git repository.
    get_repository_files: Get list of files to process from repository.
    generate_documentation: Generate documentation from processing results.
    display_statistics: Display documentation generation statistics.
"""
import os
import sys
import argparse
import asyncio
import logging
import shutil
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from pydantic import ValidationError

from config import Config
from exceptions import ConfigurationError, WorkflowError
from cache import Cache
from validation import validate_input_files, validate_git_repository
from utils import setup_logging, get_all_files, clone_repo
from workflow import process_files, write_analysis_to_markdown
from hierarchy import CodeHierarchy
from multilang import MultiLanguageManager
from context_manager import ContextManager
from context_optimizer import ContextWindowManager
from metadata_manager import MetadataManager
import sentry_sdk

# Load environment variables from .env file
load_dotenv()

@dataclass
class Statistics:
    """Container for documentation generation statistics."""
    hierarchy: Dict[str, Any]
    languages: Dict[str, int]
    context: Dict[str, Any]

@dataclass
class ApplicationManager:
    """
    Manages the application's core components and lifecycle.
    
    This class coordinates between different components and ensures proper
    initialization and cleanup of resources.
    """
    config: Config
    cache: Optional[Cache] = None
    hierarchy: Optional[CodeHierarchy] = None
    multilang_manager: Optional[MultiLanguageManager] = None
    context_manager: Optional[ContextManager] = None
    window_manager: Optional[ContextWindowManager] = None
    metadata_manager: Optional[MetadataManager] = None

    async def initialize(self):
        """Initialize application components."""
        if self.config.cache.enabled:
            self.cache = Cache(self.config.cache)
            await self.cache.initialize()
            self.metadata_manager = MetadataManager(
                db_path=os.path.join(self.config.cache.directory, 'metadata.db')  # Updated from cache_dir to directory
            )
        # Initialize other components if necessary
        self.hierarchy = CodeHierarchy()
        self.multilang_manager = MultiLanguageManager(self.config.multilang.languages)
        self.context_manager = ContextManager()
        self.window_manager = ContextWindowManager(self.config.context_optimizer)
        logging.info("Application initialized successfully.")

    async def cleanup(self):
        """Cleanup application components."""
        if self.cache and self.config.cache.enabled:
            await self.cache.cleanup()
        # Cleanup other components if necessary
        logging.info("All components cleaned up successfully.")

    async def export_hierarchy(self, output_path: str):
        """Export the code hierarchy to a file."""
        if self.hierarchy:
            await self.hierarchy.export(output_path)
            logging.info(f"Hierarchy data exported to {output_path}.")

    async def get_statistics(self) -> Statistics:
        """Get documentation generation statistics."""
        hierarchy_stats = self.hierarchy.get_stats() if self.hierarchy else {}
        language_distribution = self.multilang_manager.get_language_distribution() if self.multilang_manager else {}
        context_stats = self.context_manager.get_stats() if self.context_manager else {}
        return Statistics(
            hierarchy=hierarchy_stats,
            languages=language_distribution,
            context=context_stats
        )
    
def setup_logging(config=None):
    """Configure logging with optional config parameter."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if config and hasattr(config, 'log_level'):
        logging.getLogger().setLevel(config.log_level)

def create_arg_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI-powered documentation generator with multi-language support"
    )
    
    # Basic options
    parser.add_argument(
        "input_path",
        nargs="?",
        help="GitHub Repository URL or Local Directory Path"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output documentation file path"
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    config_group.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file"
    )
    
    # Language support
    lang_group = parser.add_argument_group('Language Support')
    lang_group.add_argument(
        "--languages",
        nargs="+",
        help="Specific languages to process (default: all supported)"
    )
    
    # Hierarchy options
    hierarchy_group = parser.add_argument_group('Hierarchy')
    hierarchy_group.add_argument(
        "--hierarchy-output",
        type=str,
        help="Output path for hierarchy data"
    )
    hierarchy_group.add_argument(
        "--group-by",
        choices=["module", "language", "type"],
        default="module",
        help="Primary grouping criterion"
    )
    
    # Context management
    context_group = parser.add_argument_group('Context Management')
    context_group.add_argument(
        "--context-size",
        type=int,
        help="Override default context window size"
    )
    context_group.add_argument(
        "--optimize-context",
        action="store_true",
        help="Enable context optimization"
    )
    
    return parser

async def process_repository(
    app: ApplicationManager,
    args: argparse.Namespace
) -> None:
    """Process a repository and generate documentation."""
    cleanup_needed = False
    repo_dir = None

    try:
        # Validate input
        if not args.input_path:
            raise ConfigurationError("Input path is required.")
        
        # Clone or use local repository
        if args.input_path.startswith("http://") or args.input_path.startswith("https://"):
            repo_dir = await clone_repository(args.input_path)
            cleanup_needed = True
        else:
            repo_dir = args.input_path
            if not os.path.isdir(repo_dir):
                raise ConfigurationError(f"Directory {repo_dir} does not exist.")
        
        # Validate repository
        if not await validate_git_repository(repo_dir):
            raise ConfigurationError(f"The repository at {repo_dir} is not a valid Git repository.")
        
        # Get list of files to process
        files_list = await get_repository_files(repo_dir, app.config)
        logging.info(f"Found {len(files_list)} files to process.")

        # Process files
        results = await process_files(
            files_list=files_list,
            repo_dir=repo_dir,
            config=app.config,
            multilang_manager=app.multilang_manager,
            hierarchy_manager=app.hierarchy,
            context_manager=app.context_manager,
            window_manager=app.window_manager,
            cache=app.cache
        )

        # Generate documentation
        output_file = args.output_file or "documentation.md"
        await generate_documentation(results, output_file, app)

        # Export hierarchy if needed
        if args.hierarchy_output:
            await app.export_hierarchy(args.hierarchy_output)

        # Display statistics
        stats = await app.get_statistics()
        display_statistics(stats)

    except (ValidationError, WorkflowError, ConfigurationError) as e:
        logging.error(f"Error processing repository: {str(e)}")
        sentry_sdk.capture_exception(e)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sentry_sdk.capture_exception(e)
    finally:
        if cleanup_needed and repo_dir:
            try:
                shutil.rmtree(repo_dir)
                logging.info(f"Cleaned up cloned repository at {repo_dir}")
            except Exception as e:
                logging.error(f"Error cleaning up repository: {str(e)}")
        await app.cleanup()

async def clone_repository(url: str) -> str:
    """Clone a Git repository."""
    repo_dir = "cloned_repo"
    if os.path.exists(repo_dir):
        try:
            shutil.rmtree(repo_dir)
            logging.info(f"Removed existing directory at {repo_dir}")
        except Exception as e:
            logging.error(f"Error removing existing directory: {str(e)}")
            raise WorkflowError(f"Failed to remove existing directory {repo_dir}: {str(e)}")
    await clone_repo(url, repo_dir)
    logging.info(f"Cloned repository {url} to {repo_dir}")
    return repo_dir

async def get_repository_files(repo_dir: str, config: Config) -> List[str]:
    """Get list of files to process from repository."""
    files = await get_all_files(repo_dir, config.exclude_dirs)
    validated_files = validate_input_files(files)
    return validated_files

async def generate_documentation(
    results: Dict[str, Any],
    output_file: str,
    app: ApplicationManager
) -> None:
    """Generate documentation from processing results."""
    try:
        write_analysis_to_markdown(results, output_file, app.config.repo_dir)
        logging.info(f"Documentation generated at {output_file}")
    except WorkflowError as e:
        logging.error(f"Workflow error: {str(e)}")
        sentry_sdk.capture_exception(e)
    except Exception as e:
        logging.error(f"Unexpected error during documentation generation: {str(e)}")
        sentry_sdk.capture_exception(e)

def display_statistics(stats: Statistics) -> None:
    """Display documentation generation statistics."""
    print("\nDocumentation Generation Statistics:")
    print("\nHierarchy Information:")
    print(f"  Total Nodes: {stats.hierarchy.get('total_nodes', 'N/A')}")
    print(f"  Maximum Depth: {stats.hierarchy.get('max_depth', 'N/A')}")
    
    print("\nLanguage Distribution:")
    for lang, count in stats.languages.items():
        print(f"  {lang}: {count} files")
    
    print("\nContext Statistics:")
    print(f"  Total Segments: {stats.context.get('total_segments', 'N/A')}")
    print(f"  Token Usage: {stats.context.get('total_tokens', 'N/A')}/{stats.context.get('max_tokens', 'N/A')}")
    utilization = stats.context.get('utilization', 0)
    print(f"  Utilization: {utilization:.2%}")

def setup_signal_handlers(app: ApplicationManager):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}. Shutting down gracefully...")
        asyncio.create_task(app.cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main_async(args: argparse.Namespace) -> None:
    """Asynchronous main function."""
    try:
        config = Config.load(args.config)
        app = ApplicationManager(config=config)
        setup_signal_handlers(app)
        await app.initialize()
        await process_repository(app, args)
    except ConfigurationError as e:
        logging.error(f"Configuration error: {str(e)}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during initialization: {str(e)}")
        sentry_sdk.capture_exception(e)
        sys.exit(1)

def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()
    setup_logging(args.config)  # Pass the config parameter

    if args.create_config:
        create_default_config(args.config)
        logging.info(f"Created default configuration at {args.config}")
        sys.exit(0)

    # Setup logging
    setup_logging(args.config)

    # Initialize Sentry if configured
    if args.config.sentry.enabled:
        sentry_sdk.init(
            dsn=args.config.sentry.dsn.get_secret_value(),
            traces_sample_rate=args.config.sentry.traces_sample_rate,
            environment=args.config.sentry.environment,
            release=args.config.sentry.release,
            debug=args.config.sentry.debug,
            attach_stacktrace=args.config.sentry.attach_stacktrace,
            send_default_pii=args.config.sentry.send_default_pii
        )

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()