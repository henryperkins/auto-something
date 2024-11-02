import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

import torch
import tiktoken
from dataclasses import dataclass
from typing import Optional, Union
from asyncio import Lock, Event


@dataclass
class ContextSegment:
    """
    Represents a code segment with associated metadata.
    
    Attributes:
        content: The actual code content.
        language: Programming language of the code segment.
        hierarchy_path: Path in the documentation hierarchy.
        access_count: Number of times the segment has been accessed.
        last_access: Timestamp of the last access.
        token_count: Number of tokens in the segment.
        relevance_score: Semantic relevance score.
    """
    content: str
    language: str
    hierarchy_path: Optional[str] = None
    access_count: int = 0
    last_access: float = time.time()
    token_count: int = 0
    relevance_score: float = 0.0


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
        self.lock = Lock()
        self.embedding_model = None
        self._embedding_model_initialized = Event()
        self._init_lock = Lock()
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
        # Ensure segment_id is a string
        if not isinstance(segment_id, str):
            segment_id = self._serialize_segment_id(segment_id)
        
        async with self.lock:
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

    def _add_to_language_tracking(self, segment_id: str, language: str) -> None:
        """Add a segment to language-specific tracking."""
        if language not in self.language_segments:
            self.language_segments[language] = set()
        self.language_segments[language].add(segment_id)

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

    def _serialize_segment_id(self, segment_id: Any) -> str:
        """Serialize segment_id to a string."""
        if isinstance(segment_id, dict):
            return str(tuple(sorted(segment_id.items())))
        return str(segment_id)

    async def _init_embedding_model(self):
        """Initialize the embedding model."""
        async with self._init_lock:
            if not self._embedding_model_initialized.is_set():
                try:
                    self.embedding_model = SomeEmbeddingModel(self.model_name)
                    self._embedding_model_initialized.set()
                except Exception as e:
                    logging.error(f"Failed to initialize embedding model: {str(e)}")
                    sentry_sdk.capture_exception(e)
                    raise

    async def _optimize_context(self):
        """Optimize the context by enforcing size and token limits."""
        # Implementation for context optimization
        pass

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

    async def clear(self) -> None:
        """Clear all segments from the context manager."""
        async with self.lock:
            self.segments.clear()
            self.embeddings.clear()
            self.language_segments.clear()

    async def cleanup(self) -> None:
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