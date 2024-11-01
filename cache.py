# cache.py
import os
import json
import sqlite3
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("cache.log", mode="a", encoding="utf-8")  # Log to file
    ],
)

@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Dict[str, Any]
    timestamp: float
    size_bytes: int

class Cache:
    """SQLite-based persistent cache implementation with module-level caching."""
    
    def __init__(self, config):
        """Initialize the cache with configuration."""
        self.config = config
        if not self.config.enabled:
            logging.info("Cache is disabled.")
            return

        # Setup cache directory and database
        self.cache_dir = Path(self.config.directory)  # Using 'directory' as per Config
        self.db_path = self.cache_dir / "cache.db"
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            gitignore_path = self.cache_dir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text("*\n")
            
            self._init_db()
            self._cleanup_old_entries()
            self._cleanup_by_size()
            logging.info(f"Cache initialized at {self.db_path}")
        except Exception as e:
            logging.error(f"Failed to initialize cache: {str(e)}")
            self.config.enabled = False

    def _init_db(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS function_cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS module_cache (
                    filepath TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    size_bytes INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_func_timestamp ON function_cache(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_module_timestamp ON module_cache(timestamp)")
            conn.commit()

    async def initialize(self):
        """
        Initialize the cache by setting up the database and performing initial cleanup.
        """
        if not self.config.enabled:
            logging.info("Cache is disabled.")
            return

        try:
            await asyncio.to_thread(self._init_db)
            await asyncio.to_thread(self._cleanup_old_entries)
            await asyncio.to_thread(self._cleanup_by_size)
            logging.info("Cache initialization complete.")
        except Exception as e:
            logging.error(f"Error during cache initialization: {str(e)}")
            raise

    async def cleanup(self):
        """
        Cleanup the cache by clearing all entries.
        """
        if not self.config.enabled:
            logging.info("Cache is disabled. No cleanup needed.")
            return

        try:
            await self.clear()  # Properly await the async clear method
            logging.info("Cache cleanup complete.")
        except Exception as e:
            logging.error(f"Error during cache cleanup: {str(e)}")
            raise

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a function analysis from the cache."""
        if not self.config.enabled:
            return None
        try:
            def fetch():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT data, timestamp FROM function_cache WHERE key = ?", (key,)
                    )
                    return cursor.fetchone()

            row = await asyncio.to_thread(fetch)
            if row:
                data_str, timestamp = row
                if time.time() - timestamp < self.config.ttl_hours * 3600:
                    logging.info(f"Cache hit for key: {key}")
                    return json.loads(data_str)
                else:
                    logging.info(f"Cache expired for key: {key}")
                    await self.set(key, None)  # Optionally remove the expired entry
            else:
                logging.info(f"Cache miss for key: {key}")
            return None
        except Exception as e:
            logging.error(f"Error retrieving from cache: {str(e)}")
            return None

    async def set(self, key: str, value: Optional[Dict[str, Any]]):
        """Store a function analysis in the cache."""
        if not self.config.enabled:
            return
        try:
            if value is None:
                def delete_entry():
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            "DELETE FROM function_cache WHERE key = ?", (key,)
                        )
                        conn.commit()
                await asyncio.to_thread(delete_entry)
                logging.info(f"Removed expired cache entry for key: {key}")
                return

            data_str = json.dumps(value)
            size_bytes = len(data_str.encode('utf-8'))
            def insert():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO function_cache (key, data, timestamp, size_bytes) VALUES (?, ?, ?, ?)",
                        (key, data_str, time.time(), size_bytes)
                    )
                    conn.commit()
            await asyncio.to_thread(insert)
            logging.info(f"Stored data in cache for key: {key}")
            await asyncio.to_thread(self._cleanup_by_size)  # Ensure async cleanup
        except Exception as e:
            logging.error(f"Error setting cache: {str(e)}")

    async def get_module(self, filepath: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve module analysis from cache."""
        if not self.config.enabled:
            return None
        try:
            def fetch_module():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT data, timestamp, hash FROM module_cache WHERE filepath = ?",
                        (filepath,)
                    )
                    return cursor.fetchone()

            row = await asyncio.to_thread(fetch_module)
            if row:
                data_str, timestamp, stored_hash = row
                if stored_hash != file_hash:
                    logging.info(f"Cache invalidated for module: {filepath}")
                    def delete_module():
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("DELETE FROM module_cache WHERE filepath = ?", (filepath,))
                            conn.commit()
                    await asyncio.to_thread(delete_module)
                    return None
                if time.time() - timestamp < self.config.ttl_hours * 3600:
                    logging.info(f"Cache hit for module: {filepath}")
                    return json.loads(data_str)
                else:
                    logging.info(f"Cache expired for module: {filepath}")
                    def delete_expired_module():
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("DELETE FROM module_cache WHERE filepath = ?", (filepath,))
                            conn.commit()
                    await asyncio.to_thread(delete_expired_module)
            else:
                logging.info(f"Cache miss for module: {filepath}")
            return None
        except Exception as e:
            logging.error(f"Error retrieving module from cache: {str(e)}")
            return None

    async def set_module(self, filepath: str, file_hash: str, value: Dict[str, Any]):
        """Store module analysis in cache."""
        if not self.config.enabled:
            return
        try:
            data_str = json.dumps(value)
            size_bytes = len(data_str.encode('utf-8'))
            def insert_module():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO module_cache (filepath, hash, data, timestamp, size_bytes) VALUES (?, ?, ?, ?, ?)",
                        (filepath, file_hash, data_str, time.time(), size_bytes)
                    )
                    conn.commit()
            await asyncio.to_thread(insert_module)
            logging.info(f"Stored module in cache: {filepath}")
            await asyncio.to_thread(self._cleanup_by_size)  # Ensure async cleanup
        except Exception as e:
            logging.error(f"Error setting module cache: {str(e)}")

    async def clear(self):
        """Clear all cached entries."""
        try:
            def clear_all():
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM function_cache")
                    conn.execute("DELETE FROM module_cache")
                    conn.commit()
            await asyncio.to_thread(clear_all)
            logging.info("Cleared all cache entries.")
        except Exception as e:
            logging.warning(f"Cache clear error: {str(e)}")

    def _cleanup_old_entries(self):
        """Remove entries older than TTL."""
        cutoff_time = time.time() - (self.config.ttl_hours * 3600)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM function_cache WHERE timestamp < ?", (cutoff_time,))
                conn.execute("DELETE FROM module_cache WHERE timestamp < ?", (cutoff_time,))
                conn.commit()
            logging.info("Performed cleanup of old cache entries.")
        except Exception as e:
            logging.error(f"Error during cache cleanup: {str(e)}")

    def _cleanup_by_size(self):
        """Remove oldest entries if cache size exceeds limit."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(size_bytes) FROM function_cache")
                func_size = cursor.fetchone()[0] or 0
                cursor.execute("SELECT SUM(size_bytes) FROM module_cache")
                module_size = cursor.fetchone()[0] or 0
                total_size = func_size + module_size
                max_size_bytes = self.config.max_size_mb * 1024 * 1024
                if total_size > max_size_bytes:
                    logging.info("Cache size exceeded limit. Initiating size-based cleanup.")
                    size_to_remove = total_size - max_size_bytes
                    for table in ['function_cache', 'module_cache']:
                        if table == 'function_cache':
                            key_field = 'key'
                        else:
                            key_field = 'filepath'
                        cursor.execute(f"SELECT {key_field} FROM {table} ORDER BY timestamp ASC")
                        rows = cursor.fetchall()
                        for row in rows:
                            key = row[0]
                            # Get size_bytes before deletion
                            cursor.execute(f"SELECT size_bytes FROM {table} WHERE {key_field} = ?", (key,))
                            removed_size = cursor.fetchone()[0]
                            cursor.execute(f"DELETE FROM {table} WHERE {key_field} = ?", (key,))
                            size_to_remove -= removed_size
                            if size_to_remove <= 0:
                                break
                    conn.commit()
                    logging.info("Size-based cache cleanup completed.")
        except Exception as e:
            logging.error(f"Error during size-based cache cleanup: {str(e)}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.config.enabled:
            return {
                'total_size_mb': 0,
                'function_count': 0,
                'module_count': 0
            }
        try:
            def fetch_stats():
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM function_cache")
                    func_size = cursor.fetchone()[0]
                    cursor.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM module_cache")
                    module_size = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM function_cache")
                    func_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM module_cache")
                    module_count = cursor.fetchone()[0]
                    return {
                        'total_size_mb': (func_size + module_size) / (1024 * 1024),
                        'function_count': func_count,
                        'module_count': module_count
                    }

            stats = await asyncio.to_thread(fetch_stats)
            return stats
        except Exception as e:
            logging.error(f"Error retrieving cache stats: {str(e)}")
            return {
                'total_size_mb': 0,
                'function_count': 0,
                'module_count': 0
            }

async def monitor_cache(cache: Cache, interval: int = 60, size_threshold_mb: int = 500):
    """Periodically log cache statistics and alert on unusual behavior."""
    while True:
        stats = await cache.get_stats()
        if stats['total_size_mb'] > size_threshold_mb:
            logging.warning(f"Cache size {stats['total_size_mb']:.2f} MB exceeds threshold of {size_threshold_mb} MB.")
        logging.info(f"Cache Stats - Total Size: {stats['total_size_mb']:.2f} MB, "
                     f"Function Entries: {stats['function_count']}, "
                     f"Module Entries: {stats['module_count']}")
        await asyncio.sleep(interval)

# Example usage:
# cache_instance = Cache(config)
# asyncio.create_task(monitor_cache(cache_instance))