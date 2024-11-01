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
import asyncio
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import signal
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import sentry_sdk

from config import Config, create_default_config
from cache import Cache
from validation import validate_input_files, validate_git_repository, ValidationError
from utils import setup_logging, get_all_files, clone_repo
from workflow import process_files
from hierarchy import CodeHierarchy
from multilang import MultiLanguageManager
from context_manager import ContextManager, ContextWindowManager
from metadata_manager import MetadataManager
from exceptions import ConfigurationError

# Initialize Sentry
sentry_sdk.init(
    dsn="your-sentry-dsn-here",
    traces_sample_rate=1.0,
    environment="production",
    release="my-project-name@2.3.12",
    debug=True,
    attach_stacktrace=True,
    send_default_pii=False
)

class ApplicationManager:
    """
    Manages the application's core components and lifecycle.
    
    This class coordinates between different components and ensures proper
    initialization and cleanup of resources.
    """
    
    def __init__(self, config: Config):
        """Initialize the application manager."""
        self.config = config
        self.cache = None
        self.hierarchy = None
        self.multilang = None
        self.context = None
        self.window_manager = None
        self.metadata_manager = None
        
    async def initialize(self):
        """Initialize all components."""
        # Initialize cache
        if self.config.cache.enabled:
            self.cache = Cache(self.config.cache)
            
        # Initialize managers
        self.hierarchy = CodeHierarchy()
        self.multilang = MultiLanguageManager(self.config.multilang)
        self.context = ContextManager(
            context_size_limit=self.config.model.context_size_limit,
            max_tokens=self.config.model.max_tokens,
            model_name=self.config.model.model
        )
        self.window_manager = ContextWindowManager(
            model_name=self.config.model.model,
            max_tokens=self.config.model.max_tokens,
            target_token_usage=self.config.context_optimizer.target_token_usage
        )
        self.metadata_manager = MetadataManager(db_path=".metadata_store.db")
        
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.cache:
                await self.cache.clear()
            if self.context:
                await self.context.cleanup()
            if self.multilang:
                self.multilang.cleanup()
            if self.window_manager:
                await self.window_manager.cleanup()
            if self.metadata_manager:
                self.metadata_manager.cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            raise

    async def export_hierarchy(self, output_path: str):
        """Export the documentation hierarchy."""
        if self.hierarchy:
            self.hierarchy.save_to_file(output_path)
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the documentation process."""
        stats = {
            'hierarchy': {
                'total_nodes': len(list(self.hierarchy.iterate_nodes())),
                'max_depth': max(node.get_path().count('.') for node in 
                               self.hierarchy.iterate_nodes())
            },
            'languages': {},
            'context': await self.context.get_context_stats()
        }
        
        # Collect language statistics
        for filepath, parsed in self.multilang.parsed_files.items():
            lang = parsed.language
            if lang not in stats['languages']:
                stats['languages'][lang] = 0
            stats['languages'][lang] += 1
            
        return stats

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
        # Handle repository
        if args.input_path.startswith(("http://", "https://")):
            repo_dir = await clone_repository(args.input_path)
            cleanup_needed = True
        else:
            repo_dir = args.input_path
            
        # Get and validate files
        files = await get_repository_files(repo_dir, app.config)
        
        # Process files
        results = await process_files(
            files,
            repo_dir,
            app.config,
            multilang_manager=app.multilang,
            hierarchy_manager=app.hierarchy,
            context_manager=app.context,
            window_manager=app.window_manager,
            cache=app.cache
        )
        
        # Generate documentation
        await generate_documentation(results, args.output_file, app)
        
        # Export hierarchy if requested
        if args.hierarchy_output:
            await app.export_hierarchy(args.hierarchy_output)
            
        # Display statistics
        stats = await app.get_statistics()
        display_statistics(stats)
        
    finally:
        if cleanup_needed and repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

async def clone_repository(url: str) -> str:
    """Clone a Git repository."""
    repo_dir = "cloned_repo"
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    await clone_repo(url, repo_dir)
    return repo_dir

async def get_repository_files(repo_dir: str, config: Config) -> List[str]:
    """Get list of files to process from repository."""
    files = await get_all_files(repo_dir, config.exclude_dirs)
    return validate_input_files(files)

async def generate_documentation(
    results: Dict[str, Any],
    output_file: str,
    app: ApplicationManager
) -> None:
    """Generate documentation from processing results."""
    from workflow import write_analysis_to_markdown
    
    # Organize results by hierarchy
    organized_results = app.hierarchy.organize_results(results)
    
    # Generate cross-references
    cross_refs = await app.multilang.get_cross_references()
    
    # Write documentation
    await write_analysis_to_markdown(
        organized_results,
        output_file,
        cross_references=cross_refs,
        hierarchy=app.hierarchy
    )

def display_statistics(stats: Dict[str, Any]) -> None:
    """Display documentation generation statistics."""
    print("\nDocumentation Generation Statistics:")
    print("\nHierarchy Information:")
    print(f"  Total Nodes: {stats['hierarchy']['total_nodes']}")
    print(f"  Maximum Depth: {stats['hierarchy']['max_depth']}")
    
    print("\nLanguage Distribution:")
    for lang, count in stats['languages'].items():
        print(f"  {lang}: {count} files")
    
    print("\nContext Statistics:")
    print(f"  Total Segments: {stats['context']['total_segments']}")
    print(f"  Token Usage: {stats['context']['total_tokens']}/{stats['context']['max_tokens']}")
    print(f"  Utilization: {stats['context']['utilization']:.2%}")

def setup_signal_handlers(app: ApplicationManager):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}. Starting cleanup...")
        asyncio.run(app.cleanup())
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main_async():
    """Asynchronous main function."""
    app = None
    try:
        # Load environment variables
        load_dotenv()
        
        # Parse arguments
        parser = create_arg_parser()
        args = parser.parse_args()
        
        # Handle configuration
        if args.create_config:
            create_default_config(args.config)
            print(f"Created default configuration at: {args.config}")
            return
            
        # Load configuration
        config = Config.load(args.config)
        
        # Validate configuration
        validate_configuration(config)
        
        # Setup logging
        setup_logging(config)
        
        # Initialize application
        app = ApplicationManager(config)
        await app.initialize()
        
        # Setup signal handlers
        setup_signal_handlers(app)
        
        try:
            # Start Sentry profiler
            sentry_sdk.profiler.start_profiler()
            
            # Process repository
            await process_repository(app, args)
            
        finally:
            # Stop Sentry profiler
            sentry_sdk.profiler.stop_profiler()
            
            # Cleanup
            await app.cleanup()
            
    except ConfigurationError as e:
        logging.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.debug("Stack trace:", exc_info=True)
        sys.exit(1)

def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()