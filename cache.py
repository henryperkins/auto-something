import os
import json
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading

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
        if not config.enabled:
            logging.info("Cache is disabled.")
            return

        # Setup cache directory and database
        cache_dir = Path(config.directory)
        self.db_path = cache_dir / "cache.db"
        
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            gitignore_path = cache_dir / ".gitignore"
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

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a function analysis from the cache."""
        if not self.config.enabled:
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("SELECT data, timestamp FROM function_cache WHERE key = ?", (key,)).fetchone()
                if result is None:
                    logging.info(f"Cache miss for key: {key}")
                    return None
                logging.info(f"Cache hit for key: {key}")
                data_str, timestamp = result
                if time.time() - timestamp > self.config.ttl_hours * 3600:
                    conn.execute("DELETE FROM function_cache WHERE key = ?", (key,))
                    logging.info(f"Cache entry expired for key: {key}")
                    return None
                return json.loads(data_str)
        except Exception as e:
            logging.warning(f"Cache read error: {str(e)}")
            return None

    def set(self, key: str, value: Dict[str, Any]):
        """Store a function analysis in the cache."""
        if not self.config.enabled:
            return
        try:
            data_str = json.dumps(value)
            size_bytes = len(data_str.encode('utf-8'))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO function_cache (key, data, timestamp, size_bytes) VALUES (?, ?, ?, ?)",
                    (key, data_str, time.time(), size_bytes)
                )
            logging.info(f"Stored key in cache: {key}")
            self._cleanup_old_entries()
            self._cleanup_by_size()
        except Exception as e:
            logging.warning(f"Cache write error: {str(e)}")

    def get_module(self, filepath: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve module analysis from cache."""
        if not self.config.enabled:
            return None
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT data, timestamp, hash FROM module_cache WHERE filepath = ?",
                    (filepath,)
                ).fetchone()
                if result is None:
                    logging.info(f"Cache miss for module: {filepath}")
                    return None
                data_str, timestamp, stored_hash = result
                if stored_hash != file_hash:
                    conn.execute("DELETE FROM module_cache WHERE filepath = ?", (filepath,))
                    logging.info(f"Cache entry invalidated for module: {filepath}")
                    return None
                if time.time() - timestamp > self.config.ttl_hours * 3600:
                    conn.execute("DELETE FROM module_cache WHERE filepath = ?", (filepath,))
                    logging.info(f"Cache entry expired for module: {filepath}")
                    return None
                logging.info(f"Cache hit for module: {filepath}")
                return json.loads(data_str)
        except Exception as e:
            logging.warning(f"Cache read error for module {filepath}: {str(e)}")
            return None
    
    def set_module(self, filepath: str, file_hash: str, value: Dict[str, Any]):
        """Store module analysis in cache."""
        if not self.config.enabled:
            return
        try:
            data_str = json.dumps(value)
            size_bytes = len(data_str.encode('utf-8'))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO module_cache (filepath, hash, data, timestamp, size_bytes) VALUES (?, ?, ?, ?, ?)",
                    (filepath, file_hash, data_str, time.time(), size_bytes)
                )
            logging.info(f"Stored module in cache: {filepath}")
            self._cleanup_old_entries()
            self._cleanup_by_size()
        except Exception as e:
            logging.warning(f"Cache write error for module {filepath}: {str(e)}")
    
def clear(self):
    """Clear all cached entries."""
    if not self.config.enabled:
        return
    try:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM function_cache")
            conn.execute("DELETE FROM module_cache")
        logging.info("Cleared all cache entries.")
    except Exception as e:
        logging.warning(f"Cache clear error: {str(e)}")
    
    def _cleanup_old_entries(self):
        """Remove entries older than TTL."""
        cutoff_time = time.time() - (self.config.ttl_hours * 3600)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM function_cache WHERE timestamp < ?", (cutoff_time,))
            conn.execute("DELETE FROM module_cache WHERE timestamp < ?", (cutoff_time,))
        logging.info("Performed cleanup of old cache entries.")

    def _cleanup_by_size(self):
        """Remove oldest entries if cache size exceeds limit."""
        with sqlite3.connect(self.db_path) as conn:
            total_size = conn.execute(
                "SELECT COALESCE((SELECT SUM(size_bytes) FROM function_cache) + (SELECT SUM(size_bytes) FROM module_cache), 0)"
            ).fetchone()[0]
            max_size_bytes = self.config.max_size_mb * 1024 * 1024
            if total_size > max_size_bytes:
                logging.info("Cache size exceeded limit, performing size-based cleanup.")
                size_to_remove = total_size - max_size_bytes
                for table in ['function_cache', 'module_cache']:
                    entries = conn.execute(f"SELECT size_bytes FROM {table} ORDER BY timestamp ASC").fetchall()
                    removed_size = 0
                    for (entry_size,) in entries:
                        conn.execute(f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} ORDER BY timestamp ASC LIMIT 1)")
                        removed_size += entry_size
                        if removed_size >= size_to_remove / 2:
                            break

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.config.enabled:
            return {
                'total_size_mb': 0,
                'entry_count': 0,
                'module_count': 0,
                'avg_age_hours': 0
            }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                total_size = conn.execute(
                    "SELECT COALESCE((SELECT SUM(size_bytes) FROM function_cache) + (SELECT SUM(size_bytes) FROM module_cache), 0)"
                ).fetchone()[0]
                stats['total_size_mb'] = total_size / (1024 * 1024)
                stats['function_count'] = conn.execute("SELECT COUNT(*) FROM function_cache").fetchone()[0]
                stats['module_count'] = conn.execute("SELECT COUNT(*) FROM module_cache").fetchone()[0]
                current_time = time.time()
                total_age = conn.execute(
                    "SELECT COALESCE((SELECT SUM(? - timestamp) FROM function_cache UNION ALL SELECT SUM(? - timestamp) FROM module_cache), 0)",
                    (current_time, current_time)
                ).fetchone()[0]
                total_entries = stats['function_count'] + stats['module_count']
                stats['avg_age_hours'] = (total_age / total_entries / 3600) if total_entries > 0 else 0
                logging.info(f"Cache stats - Total Size: {stats['total_size_mb']:.2f} MB, "
                             f"Function Entries: {stats['function_count']}, "
                             f"Module Entries: {stats['module_count']}, "
                             f"Average Age: {stats['avg_age_hours']:.2f} hours")
                return stats
        except Exception as e:
            logging.warning(f"Cache stats error: {str(e)}")
            return {
                'total_size_mb': 0,
                'entry_count': 0,
                'module_count': 0,
                'avg_age_hours': 0
            }

def monitor_cache(cache, interval=60, size_threshold_mb=500):
    """Periodically log cache statistics and alert on unusual behavior."""
    while True:
        stats = cache.get_stats()
        if stats['total_size_mb'] > size_threshold_mb:
            logging.warning(f"Cache size exceeded threshold: {stats['total_size_mb']:.2f} MB")
        time.sleep(interval)

# Example usage:
# cache_instance = Cache(config)
# monitor_thread = threading.Thread(target=monitor_cache, args=(cache_instance,))
# monitor_thread.daemon = True
# monitor_thread.start()