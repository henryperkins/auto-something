"""
Documentation module for analyzing code and generating documentation.

This module provides functionality to analyze code using AI services,
managing different AI providers through a common interface. It handles
response parsing, error management, and documentation formatting, with
support for multi-language processing and context optimization.

Classes:
    AIServiceInterface: Abstract base class for AI service implementations.
    OpenAIService: OpenAI-specific implementation of the AI service interface.
    AzureOpenAIService: Azure OpenAI-specific implementation.
    DocumentationGenerator: Main class for generating documentation.
    AIResponseParser: Parser for AI service responses.
"""

import os
import json
import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sentry_sdk
from utils import exponential_backoff_with_jitter
from context_optimizer import ContextWindowManager
from context_manager import ContextManager
from hierarchy import CodeHierarchy
from multilang import MultiLanguageManager
from metadata_manager import MetadataManager
from exceptions import AIServiceError, AIServiceConfigError, AIServiceResponseError

@dataclass
class AnalysisResult:
    """Container for function analysis results."""
    name: str
    summary: str
    docstring: str
    changelog: str
    complexity_score: Optional[int] = None

class AIServiceInterface(ABC):
    """Abstract base class defining the interface for AI services."""

    @abstractmethod
    async def analyze_function(
        self,
        function_details: Dict[str, Any],
        context_segments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyze a function using the AI service.

        Args:
            function_details: Dictionary containing function information
            context_segments: Optional list of relevant code segments
            metadata: Optional metadata about the function

        Returns:
            AnalysisResult containing the analysis

        Raises:
            AIServiceError: If analysis fails
        """
        pass

    @abstractmethod
    async def validate_configuration(self) -> None:
        """
        Validate service configuration.

        Raises:
            AIServiceConfigError: If configuration is invalid
        """
        pass

class AIResponseParser:
    """Parser for AI service responses."""

    @staticmethod
    def parse_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate an AI service response.

        Args:
            response: Raw response from the AI service

        Returns:
            Dictionary containing parsed response data

        Raises:
            AIServiceResponseError: If parsing fails
        """
        try:
            if 'choices' not in response:
                raise AIServiceResponseError("Invalid response format: missing 'choices'")

            message = response['choices'][0].get('message', {})
            
            # Handle function call format
            if 'function_call' in message:
                try:
                    args = json.loads(message['function_call']['arguments'])
                    return {
                        'summary': args.get('summary', ''),
                        'docstring': args.get('docstring', ''),
                        'changelog': args.get('changelog', '')
                    }
                except json.JSONDecodeError as e:
                    raise AIServiceResponseError(f"Failed to parse function call arguments: {str(e)}")
            
            # Handle direct content format
            elif 'content' in message:
                try:
                    content = message['content']
                    if isinstance(content, str):
                        # Attempt to parse as JSON
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            # Fall back to basic content parsing
                            return {
                                'summary': content.strip(),
                                'docstring': '',
                                'changelog': 'Initial documentation'
                            }
                    elif isinstance(content, dict):
                        return content
                except Exception as e:
                    raise AIServiceResponseError(f"Failed to parse message content: {str(e)}")
            
            raise AIServiceResponseError("Response format not recognized")
            
        except Exception as e:
            if not isinstance(e, AIServiceResponseError):
                raise AIServiceResponseError(f"Failed to parse response: {str(e)}")
            raise

class OpenAIService(AIServiceInterface):
    """OpenAI service implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def validate_configuration(self) -> None:
        if not self.api_key:
            raise AIServiceConfigError("OpenAI API key is not set")

    async def analyze_function(
        self,
        function_details: Dict[str, Any],
        context_segments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        try:
            await self.validate_configuration()
            
            prompt = self._create_prompt(function_details, context_segments, metadata)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a code documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                "functions": [self._get_function_schema()],
                "function_call": {"name": "analyze_and_document_function"},
                "temperature": 0.2
            }

            async with aiohttp.ClientSession() as session:
                completion = await exponential_backoff_with_jitter(
                    lambda: self._make_request(session, payload)
                )

            parsed_response = AIResponseParser.parse_response(completion)
            
            return AnalysisResult(
                name=function_details["name"],
                summary=parsed_response["summary"],
                docstring=parsed_response["docstring"],
                changelog=parsed_response["changelog"],
                complexity_score=function_details.get("complexity")
            )

        except AIServiceError:
            raise
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise AIServiceError(f"OpenAI service error: {str(e)}")

    async def _make_request(self, session: aiohttp.ClientSession, payload: Dict) -> Dict:
        """Make a request to the OpenAI API."""
        async with session.post(self.endpoint, json=payload, headers=self.headers) as response:
            if response.status != 200:
                error_detail = await response.text()
                raise AIServiceError(f"OpenAI API request failed with status {response.status}: {error_detail}")
            return await response.json()

    def _get_function_schema(self) -> Dict[str, Any]:
        """Get the function schema for the API."""
        return {
            "name": "analyze_and_document_function",
            "description": "Analyzes a Python function and provides structured documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Concise description of function purpose"
                    },
                    "docstring": {
                        "type": "string",
                        "description": "Complete Google-style docstring"
                    },
                    "changelog": {
                        "type": "string",
                        "description": "Documentation change history"
                    }
                },
                "required": ["summary", "docstring", "changelog"]
            }
        }

    def _create_prompt(
        self,
        function_details: Dict[str, Any],
        context_segments: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Create the prompt for the AI service."""
        parts = []
        
        # Add context if available
        if context_segments:
            context = "\n\n".join(context_segments)
            parts.append(f"Additional Context:\n```python\n{context}\n```\n")
        
        # Add metadata if available
        if metadata:
            meta_str = json.dumps(metadata, indent=2)
            parts.append(f"Metadata:\n```json\n{meta_str}\n```\n")
        
        # Add function details
        parts.extend([
            f"Function Name: {function_details['name']}",
            f"Parameters: {', '.join(f'{p[0]}: {p[1]}' for p in function_details.get('params', []))}",
            f"Return Type: {function_details.get('return_type', 'None')}",
            f"Existing Docstring: {function_details.get('docstring', 'None')}",
            "",
            "Source Code:",
            f"```python\n{function_details['code']}\n```"
        ])
        
        return "\n".join(parts)

class AzureOpenAIService(AIServiceInterface):
    """Azure OpenAI service implementation."""

    def __init__(self, api_key: str, endpoint: str, deployment_name: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.headers = {
            "api-key": api_key,
            "Content-Type": "application/json"
        }

    async def validate_configuration(self) -> None:
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise AIServiceConfigError("Azure OpenAI configuration is incomplete")

    async def analyze_function(
        self,
        function_details: Dict[str, Any],
        context_segments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        try:
            await self.validate_configuration()
            
            prompt = OpenAIService._create_prompt(self, function_details, context_segments, metadata)
            endpoint = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version=2024-02-15-preview"
            
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a code documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }

            async with aiohttp.ClientSession() as session:
                completion = await exponential_backoff_with_jitter(
                    lambda: self._make_request(session, endpoint, payload)
                )

            parsed_response = AIResponseParser.parse_response(completion)
            
            return AnalysisResult(
                name=function_details["name"],
                summary=parsed_response["summary"],
                docstring=parsed_response["docstring"],
                changelog=parsed_response["changelog"],
                complexity_score=function_details.get("complexity")
            )

        except AIServiceError:
            raise
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise AIServiceError(f"Azure OpenAI service error: {str(e)}")

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        payload: Dict
    ) -> Dict:
        """Make a request to the Azure OpenAI API."""
        async with session.post(endpoint, json=payload, headers=self.headers) as response:
            if response.status != 200:
                error_detail = await response.text()
                raise AIServiceError(
                    f"Azure OpenAI API request failed with status {response.status}: {error_detail}"
                )
            return await response.json()

class DocumentationGenerator:
    """Main class for generating documentation using AI services."""

    def __init__(self, 
                 service: AIServiceInterface,
                 context_manager: ContextManager,
                 hierarchy_manager: CodeHierarchy,
                 multilang_manager: MultiLanguageManager,
                 metadata_manager: MetadataManager):
        """
        Initialize the documentation generator.
        
        Args:
            service: AI service interface
            context_manager: Context management instance
            hierarchy_manager: Hierarchy management instance
            multilang_manager: Multi-language support instance
            metadata_manager: Metadata management instance
        """
        self.service = service
        self.context_manager = context_manager
        self.hierarchy_manager = hierarchy_manager
        self.multilang_manager = multilang_manager
        self.metadata_manager = metadata_manager
        self.window_manager = ContextWindowManager()

    async def analyze_function(
        self,
        function_details: Dict[str, Any],
        context_segments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyze a function and generate documentation.

        Args:
            function_details: Dictionary containing function information
            context_segments: Optional list of relevant code segments
            metadata: Optional metadata about the function

        Returns:
            AnalysisResult containing the analysis

        Raises:
            AIServiceError: If analysis fails
        """
        try:
            # Detect language if not specified
            if 'language' not in metadata:
                detected_lang = await self.multilang_manager.detect_language(
                    function_details['code']
                )
                metadata['language'] = detected_lang

            # Update hierarchy
            hierarchy_path = self.hierarchy_manager.add_node(
                path=function_details['name'],
                node_type='function',
                documentation=function_details.get('docstring'),
                metadata=metadata
            ).get_path()

            # Optimize context window
            context = await self.window_manager.optimize_window()
            if context_segments:
                optimized_segments = await self.context_manager.get_semantic_relevant_segments(
                    context_segments,
                    target_tokens=self.window_manager.target_tokens
                )
                context_segments = optimized_segments

            # Generate documentation
            result = await self.service.analyze_function(
                function_details,
                context_segments,
                {
                    **metadata,
                    'hierarchy_path': hierarchy_path,
                    'language': metadata['language']
                }
            )

            # Synchronize metadata with context information
            await self.metadata_manager.sync_with_context(
                function_details['name'],
                self.window_manager.get_segment_score(function_details['name']),
                time.time()
            )

            return result

        except Exception as e:
            logging.error(f"Error analyzing function {function_details['name']}: {str(e)}")
            sentry_sdk.capture_exception(e)
            raise

    async def analyze_code_file(
        self,
        filepath: str,
        content: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a complete code file.

        Args:
            filepath: Path to the file
            content: File content
            config: Configuration dictionary

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Process file with multi-language support
            parsed_file = await self.multilang_manager.process_file(filepath, content)
            
            if parsed_file.errors:
                logging.warning(f"Errors parsing {filepath}: {parsed_file.errors}")
                return {'errors': parsed_file.errors}

            # Create hierarchy nodes for file structure
            file_node = self.hierarchy_manager.add_node(
                path=filepath,
                node_type='file',
                metadata={'language': parsed_file.language}
            )

            results = {
                'language': parsed_file.language,
                'elements': []
            }

            for element in parsed_file.content.get('functions', []) + \
                        parsed_file.content.get('classes', []):
                try:
                    # Optimize context for each element
                    context = await self.window_manager.get_optimized_context()
                    
                    # Generate documentation
                    element_result = await self.analyze_function(
                        element,
                        context_segments=context,
                        metadata={
                            'file_path': filepath,
                            'language': parsed_file.language,
                            'parent_node': file_node.get_path()
                        }
                    )
                    
                    results['elements'].append(element_result)
                    
                except Exception as e:
                    logging.error(f"Error analyzing element {element.get('name')}: {str(e)}")
                    sentry_sdk.capture_exception(e)
                    results['elements'].append({
                        'name': element.get('name'),
                        'error': str(e)
                    })

            return results

        except Exception as e:
            logging.error(f"Error analyzing code file {filepath}: {str(e)}")
            sentry_sdk.capture_exception(e)
            return {'error': str(e)}

    async def generate_cross_references(self) -> Dict[str, List[str]]:
        """
        Generate cross-references between documented elements.

        Returns:
            Dictionary of cross-references
        """
        try:
            return await self.multilang_manager.get_cross_references()
        except Exception as e:
            logging.error(f"Error generating cross-references: {str(e)}")
            sentry_sdk.capture_exception(e)
            return {}