"""
Context Management Module.

This module provides dynamic context management, adjusting the context window
based on code relevance, token limits, and semantic similarity. It integrates
with the multi-language support and hierarchy systems for comprehensive
context handling across different programming languages.

Classes:
    ContextSegment: Represents a segment in the context window.
    ContextManager: Manages code segments and their relevance for context window optimization.
"""

import logging
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Set
import tiktoken
from sentence_transformers import SentenceTransformer, util
import torch
import async_timeout
from dataclasses import dataclass
import sentry_sdk

@dataclass
class ContextSegment:
    """
    Represents a segment in the context window.
    
    Attributes:
        content: The code content
        language: Programming language
        hierarchy_path: Path in documentation hierarchy
        relevance_score: Semantic relevance score
        token_count: Number of tokens in segment
        last_access: Timestamp of last access
        access_count: Number of times accessed
    """
    content: str
    language: str
    hierarchy_path: Optional[str] = None
    relevance_score: float = 0.0
    token_count: int = 0
    last_access: float = 0.0
    access_count: int = 0

class ContextManager:
    """
    Manages code segments and their relevance for context window optimization.
    
    This class tracks access frequency, timestamps, and embeddings for code segments,
    providing methods to update access information and retrieve the most relevant
    code segments based on recency, frequency, and semantic similarity.
    """
    
    def __init__(self, 
                 context_size_limit: int = 10,
                 max_tokens: int = 2048,
                 model_name: str = 'gpt-4',
                 embedding_batch_size: int = 32):
        """
        Initialize the context manager.
        
        Args:
            context_size_limit: Maximum number of segments in context
            max_tokens: Maximum tokens allowed
            model_name: Model name for tokenization
            embedding_batch_size: Batch size for embedding generation
        """
        self.context_size_limit = context_size_limit
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.embedding_batch_size = embedding_batch_size
        
        self.segments: Dict[str, ContextSegment] = {}
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.language_segments: Dict[str, Set[str]] = {}
        
        # Initialize components
        self.lock = asyncio.Lock()
        self.embedding_model = None
        self._embedding_model_initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()
        self._init_timeout = 30.0
        self._initialization_task = None
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception as e:
            logging.error(f"Failed to initialize tokenizer: {str(e)}")
            sentry_sdk.capture_exception(e)
            raise

    async def add_or_update_segment(self,
                                    segment_id: str,
                                    content: str,
                                    metadata: Dict[str, Any]) -> None:
        """
        Add or update a code segment with metadata.
        
        Args:
            segment_id: Unique identifier for the segment
            content: Code content
            metadata: Metadata including language and hierarchy info
        """
        async with self.lock:
            # Create or update segment
            if segment_id in self.segments:
                segment = self.segments[segment_id]
                segment.content = content
                segment.access_count += 1
                segment.last_access = time.time()
                # Update language tracking if changed
                if segment.language != metadata['language']:
                    old_lang = segment.language
                    new_lang = metadata['language']
                    if old_lang in self.language_segments:
                        self.language_segments[old_lang].remove(segment_id)
                    self._add_to_language_tracking(segment_id, new_lang)
            else:
                segment = ContextSegment(
                    content=content,
                    language=metadata['language'],
                    hierarchy_path=metadata.get('hierarchy_path'),
                    last_access=time.time()
                )
                self.segments[segment_id] = segment
                self._add_to_language_tracking(segment_id, metadata['language'])
            
            # Update token count
            segment.token_count = len(self.tokenizer.encode(content))
            
            # Generate embedding
            embedding = await self._generate_embedding(content)
            if embedding is not None:
                self.embeddings[segment_id] = embedding
                
            await self._optimize_context()

    async def get_semantic_relevant_segments(self,
                                             query: str,
                                             language: Optional[str] = None,
                                             top_k: int = 5,
                                             target_tokens: Optional[int] = None) -> List[str]:
        """
        Retrieve the most semantically relevant code segments.
        
        Args:
            query: Query string for relevance matching
            language: Optional language filter
            top_k: Maximum number of segments to retrieve
            target_tokens: Optional token limit target
            
        Returns:
            List of relevant code segments
        """
        query_embedding = await self._generate_embedding(query)
        if query_embedding is None:
            return []
            
        async with self.lock:
            # Filter segments by language if specified
            segment_ids = (
                self.language_segments.get(language, set())
                if language else set(self.segments.keys())
            )
            
            # Calculate similarities
            similarities = []
            for segment_id in segment_ids:
                if segment_id in self.embeddings:
                    similarity = util.cos_sim(
                        query_embedding,
                        self.embeddings[segment_id]
                    )[0][0]
                    similarities.append((segment_id, similarity.item()))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Select top segments within token limit
            selected_segments = []
            total_tokens = 0
            
            for segment_id, similarity in similarities[:top_k]:
                segment = self.segments[segment_id]
                if target_tokens and total_tokens + segment.token_count > target_tokens:
                    break
                selected_segments.append(segment.content)
                total_tokens += segment.token_count
                
                # Update segment metadata
                segment.relevance_score = similarity
                segment.access_count += 1
                segment.last_access = time.time()
                
            return selected_segments

    async def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the current context state."""
        async with self.lock:
            total_segments = len(self.segments)
            total_tokens = sum(segment.token_count for segment in self.segments.values())
            avg_relevance = sum(segment.relevance_score for segment in self.segments.values()) / total_segments if total_segments > 0 else 0
            
            language_stats = {
                lang: len(segments)
                for lang, segments in self.language_segments.items()
            }
            
            return {
                'total_segments': total_segments,
                'total_tokens': total_tokens,
                'available_tokens': max(0, self.max_tokens - total_tokens),
                'average_relevance': avg_relevance,
                'language_distribution': language_stats
            }

    async def _optimize_context(self) -> None:
        """Optimize the context window based on relevance and constraints."""
        async with self.lock:
            total_tokens = sum(segment.token_count for segment in self.segments.values())
            
            if (total_tokens > self.max_tokens or 
                len(self.segments) > self.context_size_limit):
                # Score segments
                scored_segments = [
                    (segment_id, self._calculate_segment_score(segment))
                    for segment_id, segment in self.segments.items()
                ]
                scored_segments.sort(key=lambda x: x[1], reverse=True)
                
                # Keep highest scoring segments within limits
                kept_segments = set()
                current_tokens = 0
                
                for segment_id, score in scored_segments:
                    segment = self.segments[segment_id]
                    if (len(kept_segments) < self.context_size_limit and
                        current_tokens + segment.token_count <= self.max_tokens):
                        kept_segments.add(segment_id)
                        current_tokens += segment.token_count
                
                # Remove other segments
                for segment_id in list(self.segments.keys()):
                    if segment_id not in kept_segments:
                        await self.remove_segment(segment_id)

    def _calculate_segment_score(self, segment: ContextSegment) -> float:
        """Calculate a segment's importance score."""
        recency_factor = 1.0 / (time.time() - segment.last_access + 1)
        frequency_factor = segment.access_count / (max(s.access_count for s in self.segments.values()) + 1)
        relevance_factor = segment.relevance_score
        
        return (0.4 * recency_factor + 
                0.3 * frequency_factor +
                0.3 * relevance_factor)

    def _add_to_language_tracking(self, segment_id: str, language: str) -> None:
        """Add a segment to language-specific tracking."""
        if language not in self.language_segments:
            self.language_segments[language] = set()
        self.language_segments[language].add(segment_id)

    async def remove_segment(self, segment_id: str) -> None:
        """Remove a segment from the context manager."""
        async with self.lock:
            if segment_id in self.segments:
                segment = self.segments[segment_id]
                if segment.language in self.language_segments:
                    self.language_segments[segment.language].remove(segment_id)
                del self.segments[segment_id]
                self.embeddings.pop(segment_id, None)

    async def clear(self) -> None:
        """Clear all segments from the context manager."""
        async with self.lock:
            self.segments.clear()
            self.embeddings.clear()
            self.language_segments.clear()

    async def _init_embedding_model(self) -> None:
        """Initialize the embedding model."""
        async with async_timeout.timeout(self._init_timeout):
            if not self._initialization_task:
                self._initialization_task = asyncio.create_task(self._load_model())
            await self._initialization_task

    async def _load_model(self) -> None:
        """Load the embedding model."""
        async with self.lock:
            if not self._embedding_model_initialized.is_set():
                try:
                    loop = asyncio.get_event_loop()
                    self.embedding_model = await loop.run_in_executor(
                        None,
                        lambda: SentenceTransformer('all-MiniLM-L6-v2')
                    )
                    self._embedding_model_initialized.set()
                except Exception as e:
                    logging.error(f"Failed to initialize embedding model: {str(e)}")
                    sentry_sdk.capture_exception(e)
                    raise

    async def _generate_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Generate embedding for text."""
        await self._init_embedding_model()
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                text,
                {'convert_to_tensor': True}
            )
            return embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            sentry_sdk.capture_exception(e)
            return None

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.clear()
            if self.embedding_model:
                # Clean up embedding model resources
                del self.embedding_model
                self.embedding_model = None
                self._embedding_model_initialized.clear()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            sentry_sdk.capture_exception(e)
