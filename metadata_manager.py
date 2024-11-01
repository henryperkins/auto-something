"""
Metadata Manager Module.

This module provides functionality for tagging and managing metadata for code
segments across multiple programming languages. It supports hierarchical
organization and efficient querying of metadata.
"""

import os
import json
import sqlite3
import logging
import threading
import time
import atexit
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass
from contextlib import contextmanager
import asyncio
from pathlib import Path

@dataclass
class MetadataEntry:
    """
    Container for metadata information.
    
    Attributes:
        segment_id: Unique identifier for the code segment
        metadata: Dictionary of metadata
        language: Programming language
        hierarchy_path: Path in documentation hierarchy
        timestamps: Dictionary of timestamps
        usage_metrics: Dictionary of usage metrics
    """
    segment_id: str
    metadata: Dict[str, Any]
    language: str
    hierarchy_path: Optional[str] = None
    timestamps: Dict[str, float] = None
    usage_metrics: Dict[str, Union[int, float]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.timestamps is None:
            self.timestamps = {
                'created': time.time(),
                'last_updated': time.time(),
                'last_accessed': time.time()
            }
        if self.usage_metrics is None:
            self.usage_metrics = {
                'access_count': 0,
                'update_count': 0,
                'reference_count': 0
            }

class MetadataManager:
    """High-level interface for metadata management."""
    
    def __init__(self, db_path: str, batch_size: int = 50):
        """
        Initialize the metadata manager.
        
        Args:
            db_path: Path to SQLite database
            batch_size: Size of batch operations
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.cache: Dict[str, MetadataEntry] = {}
        self.pending_updates: List[Tuple[str, MetadataEntry]] = []
        self._init_db()
        self._start_background_flush()
        atexit.register(self.cleanup)
        
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    segment_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    language TEXT NOT NULL,
                    hierarchy_path TEXT,
                    timestamps_json TEXT NOT NULL,
                    usage_metrics_json TEXT NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_language ON metadata(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hierarchy ON metadata(hierarchy_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_updated ON metadata(last_updated)")
            
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _start_background_flush(self):
        """Start background thread for flushing updates."""
        self.should_stop = threading.Event()
        self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self.flush_thread.start()

    def _background_flush(self):
        """Background thread for periodic flushing of updates."""
        while not self.should_stop.is_set():
            try:
                with self.lock:
                    if len(self.pending_updates) >= self.batch_size:
                        self.flush()
                time.sleep(1)  # Check every second
            except Exception as e:
                logging.error(f"Error in background flush: {str(e)}")

    def add_or_update_entry(self, entry: MetadataEntry) -> None:
        """Add or update a metadata entry."""
        with self.lock:
            self.cache[entry.segment_id] = entry
            self.pending_updates.append((entry.segment_id, entry))
            if len(self.pending_updates) >= self.batch_size:
                self.flush()

    def get_entry(self, segment_id: str) -> Optional[MetadataEntry]:
        """Retrieve a metadata entry."""
        # Check cache first
        with self.lock:
            if segment_id in self.cache:
                entry = self.cache[segment_id]
                entry.usage_metrics['access_count'] += 1
                entry.timestamps['last_accessed'] = time.time()
                return entry

        # Query database
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT * FROM metadata WHERE segment_id = ?",
                (segment_id,)
            ).fetchone()

            if result:
                entry = self._row_to_entry(result)
                with self.lock:
                    self.cache[segment_id] = entry
                return entry

        return None

    def _row_to_entry(self, row) -> MetadataEntry:
        """Convert database row to MetadataEntry."""
        return MetadataEntry(
            segment_id=row[0],
            metadata=json.loads(row[1]),
            language=row[2],
            hierarchy_path=row[3],
            timestamps=json.loads(row[4]),
            usage_metrics=json.loads(row[5])
        )

    def flush(self) -> None:
        """Flush pending updates to database."""
        with self.lock:
            updates = self.pending_updates[:]
            self.pending_updates.clear()

        if not updates:
            return

        with self._get_connection() as conn:
            for segment_id, entry in updates:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata
                    (segment_id, metadata_json, language, hierarchy_path,
                     timestamps_json, usage_metrics_json, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        segment_id,
                        json.dumps(entry.metadata),
                        entry.language,
                        entry.hierarchy_path,
                        json.dumps(entry.timestamps),
                        json.dumps(entry.usage_metrics),
                        time.time()
                    )
                )

    async def async_flush(self):
        """Async version of flush operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.flush)

    def cleanup(self):
        """Clean up resources."""
        self.should_stop.set()
        if hasattr(self, 'flush_thread'):
            self.flush_thread.join(timeout=5.0)
        self.flush()

    def query_by_language(self, language: str) -> List[MetadataEntry]:
        """Query entries by programming language."""
        with self._get_connection() as conn:
            results = conn.execute(
                "SELECT * FROM metadata WHERE language = ?",
                (language,)
            ).fetchall()
            return [self._row_to_entry(row) for row in results]

    def query_by_hierarchy(self, path_prefix: str) -> List[MetadataEntry]:
        """Query entries by hierarchy path prefix."""
        with self._get_connection() as conn:
            results = conn.execute(
                "SELECT * FROM metadata WHERE hierarchy_path LIKE ?",
                (f"{path_prefix}%",)
            ).fetchall()
            return [self._row_to_entry(row) for row in results]

    async def tag_code_segment(self,
                               segment_id: str,
                               metadata: Dict[str, Any],
                               language: str,
                               hierarchy_path: Optional[str] = None) -> None:
        """Tag a code segment with metadata."""
        entry = MetadataEntry(
            segment_id=segment_id,
            metadata=metadata,
            language=language,
            hierarchy_path=hierarchy_path
        )
        self.add_or_update_entry(entry)
        await self.async_flush()

    async def get_metadata(self, segment_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a code segment."""
        entry = self.get_entry(segment_id)
        return entry.metadata if entry else None

    async def update_metadata(self,
                              segment_id: str,
                              metadata: Dict[str, Any]) -> None:
        """Update metadata for a code segment."""
        entry = self.get_entry(segment_id)
        if entry:
            entry.metadata.update(metadata)
            entry.timestamps['last_updated'] = time.time()
            entry.usage_metrics['update_count'] += 1
            self.add_or_update_entry(entry)
            await self.async_flush()

    async def get_language_segments(self, language: str) -> List[Dict[str, Any]]:
        """Get all segments for a specific language."""
        entries = self.query_by_language(language)
        return [
            {
                'segment_id': entry.segment_id,
                'metadata': entry.metadata,
                'hierarchy_path': entry.hierarchy_path
            }
            for entry in entries
        ]

    async def get_hierarchy_segments(self, 
                                     path_prefix: str) -> List[Dict[str, Any]]:
        """Get all segments under a hierarchy path."""
        entries = self.query_by_hierarchy(path_prefix)
        return [
            {
                'segment_id': entry.segment_id,
                'metadata': entry.metadata,
                'language': entry.language
            }
            for entry in entries
        ]

    def _recover_corrupted_entry(self, segment_id: str) -> Optional[MetadataEntry]:
        """Attempt to recover corrupted metadata entry."""
        try:
            with self._get_connection() as conn:
                # Try to recover basic information
                basic_info = conn.execute(
                    "SELECT language, hierarchy_path FROM metadata WHERE segment_id = ?",
                    (segment_id,)
                ).fetchone()
                
                if basic_info:
                    return MetadataEntry(
                        segment_id=segment_id,
                        metadata={},
                        language=basic_info[0],
                        hierarchy_path=basic_info[1]
                    )
        except Exception as e:
            logging.error(f"Failed to recover entry {segment_id}: {str(e)}")
        return None