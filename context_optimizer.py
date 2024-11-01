"""
Context Window Optimizer Module.

This module provides advanced context window management and optimization for AI-driven
code documentation. It implements dynamic context adjustment, predictive token management,
and intelligent context prioritization to maximize the effectiveness of limited context windows.

Classes:
    TokenPredictor: Predicts token usage for code segments.
    ContextPrioritizer: Prioritizes code segments based on relevance and dependencies.
    ContextWindowManager: Manages and optimizes context windows dynamically.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import tiktoken
from collections import defaultdict
import numpy as np
from sentence_transformers import util

@dataclass
class TokenUsageStats:
    """Container for token usage statistics."""
    total_tokens: int
    code_tokens: int
    comment_tokens: int
    avg_tokens_per_line: float
    prediction_confidence: float

class TokenPredictor:
    """
    Predicts token usage for code segments using historical data and heuristics.
    
    This class analyzes code characteristics to predict token consumption,
    helping optimize context window usage before actual tokenization.
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the token predictor."""
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.usage_history: Dict[str, TokenUsageStats] = {}
        self.lock = asyncio.Lock()

    async def predict_tokens(self, code: str) -> TokenUsageStats:
        """
        Predict token usage for a code segment.
        
        Args:
            code: The code segment to analyze
            
        Returns:
            TokenUsageStats containing predicted token usage statistics
        """
        lines = code.split('\n')
        num_lines = len(lines)
        
        # Basic statistics
        num_chars = len(code)
        num_words = len(code.split())
        
        # Calculate actual tokens for calibration
        actual_tokens = len(self.tokenizer.encode(code))
        
        # Calculate token statistics
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = num_lines - comment_lines
        
        # Predict based on characteristics
        predicted_tokens = int(num_chars * 0.3 + num_words * 0.7)  # Simple heuristic
        confidence = 1.0 - abs(predicted_tokens - actual_tokens) / actual_tokens
        
        stats = TokenUsageStats(
            total_tokens=actual_tokens,
            code_tokens=int(actual_tokens * (code_lines / num_lines)),
            comment_tokens=int(actual_tokens * (comment_lines / num_lines)),
            avg_tokens_per_line=actual_tokens / num_lines if num_lines > 0 else 0,
            prediction_confidence=confidence
        )
        
        # Update history
        async with self.lock:
            self.usage_history[hash(code)] = stats
        
        return stats

    async def get_historical_stats(self, code_hash: str) -> Optional[TokenUsageStats]:
        """Retrieve historical token usage statistics for a code segment."""
        async with self.lock:
            return self.usage_history.get(code_hash)

class ContextPrioritizer:
    """
    Prioritizes code segments based on relevance and dependencies.
    
    This class analyzes code relationships and importance to determine
    which segments should be included in limited context windows.
    """
    
    def __init__(self):
        """Initialize the context prioritizer."""
        self.dependency_graph = defaultdict(set)
        self.importance_scores = {}
        self.lock = asyncio.Lock()

    async def add_dependency(self, source: str, target: str) -> None:
        """Add a dependency relationship between code segments."""
        async with self.lock:
            self.dependency_graph[source].add(target)

    async def calculate_importance(self, segment_id: str, 
                                 usage_frequency: int,
                                 modification_recency: float,
                                 complexity_score: float) -> float:
        """
        Calculate importance score for a code segment.
        
        Args:
            segment_id: Unique identifier for the code segment
            usage_frequency: How often the segment is used
            modification_recency: How recently the segment was modified
            complexity_score: Complexity measure of the segment
            
        Returns:
            Importance score between 0 and 1
        """
        # Normalize inputs
        norm_frequency = min(usage_frequency / 100, 1.0)
        norm_recency = 1.0 / (1.0 + modification_recency)  # More recent = higher score
        norm_complexity = min(complexity_score / 10, 1.0)
        
        # Calculate base score
        base_score = (
            0.4 * norm_frequency +
            0.3 * norm_recency +
            0.3 * norm_complexity
        )
        
        # Adjust for dependencies
        async with self.lock:
            num_dependents = len([d for d in self.dependency_graph.values() 
                                if segment_id in d])
            dependency_factor = min(num_dependents / 10, 1.0)
        
        final_score = 0.7 * base_score + 0.3 * dependency_factor
        
        async with self.lock:
            self.importance_scores[segment_id] = final_score
        
        return final_score

    async def get_priority_ordering(self, 
                                  segments: List[str],
                                  max_segments: Optional[int] = None) -> List[str]:
        """
        Get ordered list of segments based on importance scores.
        
        Args:
            segments: List of segment IDs to order
            max_segments: Optional maximum number of segments to return
            
        Returns:
            Ordered list of segment IDs
        """
        async with self.lock:
            scored_segments = [
                (s, self.importance_scores.get(s, 0.0))
                for s in segments
            ]
        
        # Sort by importance score
        sorted_segments = sorted(
            scored_segments,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply limit if specified
        if max_segments is not None:
            sorted_segments = sorted_segments[:max_segments]
        
        return [s[0] for s in sorted_segments]

class ContextWindowManager:
    """
    Manages and optimizes context windows dynamically.
    
    This class handles the selection and organization of code segments
    within token limits, using prediction and prioritization to maximize
    context window utility.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 max_tokens: int = 8192,
                 target_token_usage: float = 0.9):
        """
        Initialize the context window manager.
        
        Args:
            model_name: Name of the model for tokenization
            max_tokens: Maximum tokens allowed in context
            target_token_usage: Target proportion of max_tokens to use
        """
        self.max_tokens = max_tokens
        self.target_tokens = int(max_tokens * target_token_usage)
        self.predictor = TokenPredictor(model_name)
        self.prioritizer = ContextPrioritizer()
        self.current_segments: Dict[str, str] = {}
        self.current_token_count = 0
        self.lock = asyncio.Lock()

    async def add_segment(self,
                         segment_id: str,
                         code: str,
                         importance_params: Dict[str, Any]) -> bool:
        """
        Add a code segment to the context window if possible.
        
        Args:
            segment_id: Unique identifier for the code segment
            code: The code segment content
            importance_params: Parameters for calculating importance
            
        Returns:
            bool indicating whether the segment was added successfully
        """
        # Predict token usage
        stats = await self.predictor.predict_tokens(code)
        
        async with self.lock:
            # Check if addition would exceed token limit
            if self.current_token_count + stats.total_tokens > self.max_tokens:
                return False
            
            # Calculate importance score
            importance = await self.prioritizer.calculate_importance(
                segment_id,
                importance_params.get('usage_frequency', 1),
                importance_params.get('modification_recency', 0.0),
                importance_params.get('complexity_score', 1.0)
            )
            
            # Add segment
            self.current_segments[segment_id] = code
            self.current_token_count += stats.total_tokens
            
            return True

    async def remove_segment(self, segment_id: str) -> None:
        """Remove a segment from the context window."""
        async with self.lock:
            if segment_id in self.current_segments:
                code = self.current_segments[segment_id]
                stats = await self.predictor.predict_tokens(code)
                self.current_token_count -= stats.total_tokens
                del self.current_segments[segment_id]

    async def optimize_window(self) -> None:
        """
        Optimize the current context window based on priorities.
        
        This method reorganizes segments to maximize the utility
        of the available token space while respecting dependencies
        and importance scores.
        """
        async with self.lock:
            # Get current segments and their importances
            segments = list(self.current_segments.keys())
            ordered_segments = await self.prioritizer.get_priority_ordering(segments)
            
            # Temporarily store all segments
            temp_segments = self.current_segments.copy()
            
            # Clear current context
            self.current_segments.clear()
            self.current_token_count = 0
            
            # Add segments back in priority order
            for segment_id in ordered_segments:
                code = temp_segments[segment_id]
                stats = await self.predictor.predict_tokens(code)
                
                if self.current_token_count + stats.total_tokens <= self.target_tokens:
                    self.current_segments[segment_id] = code
                    self.current_token_count += stats.total_tokens
                else:
                    break

    async def get_window_stats(self) -> Dict[str, Any]:
        """Get statistics about the current context window."""
        async with self.lock:
            return {
                'num_segments': len(self.current_segments),
                'total_tokens': self.current_token_count,
                'available_tokens': self.max_tokens - self.current_token_count,
                'utilization': self.current_token_count / self.max_tokens
            }

    async def get_optimized_context(self) -> List[Tuple[str, str]]:
        """
        Get the optimized context as a list of (segment_id, code) tuples.
        
        Returns:
            List of tuples containing segment IDs and their code
        """
        await self.optimize_window()
        async with self.lock:
            return [(sid, code) for sid, code in self.current_segments.items()]