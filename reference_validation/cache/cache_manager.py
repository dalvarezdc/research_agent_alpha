"""
Cache manager for validation results.
Supports multiple backends: SQLite, JSON, in-memory.
"""

import json
import sqlite3
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path

from ..models import ValidationResult


class CacheManager:
    """Manages caching of validation results"""

    def __init__(self, backend: str = "sqlite", cache_path: str = "./cache/reference_validation.db", ttl_days: int = 30):
        """
        Initialize cache manager.

        Args:
            backend: Cache backend (sqlite, json, memory, none)
            cache_path: Path for persistent cache
            ttl_days: Time-to-live for cache entries in days
        """
        self.backend = backend
        self.cache_path = cache_path
        self.ttl_days = ttl_days
        self.memory_cache = {}  # For memory backend

        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "json":
            self._init_json()
        elif backend in ["memory", "none"]:
            pass  # No initialization needed
        else:
            raise ValueError(f"Unknown cache backend: {backend}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database"""
        # Create directory if needed
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_cache (
                key TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL
            )
        """)

        # Create index on expiration
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON validation_cache(expires_at)
        """)

        conn.commit()
        conn.close()

    def _init_json(self) -> None:
        """Initialize JSON file cache"""
        # Create directory if needed
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)

        if not Path(self.cache_path).exists():
            with open(self.cache_path, 'w') as f:
                json.dump({}, f)

    def get(self, key: str) -> Optional[ValidationResult]:
        """
        Get validation result from cache.

        Args:
            key: Cache key

        Returns:
            ValidationResult if found and not expired, None otherwise
        """
        if self.backend == "none":
            return None

        if self.backend == "memory":
            return self._get_memory(key)
        elif self.backend == "sqlite":
            return self._get_sqlite(key)
        elif self.backend == "json":
            return self._get_json(key)

        return None

    def set(self, key: str, result: ValidationResult) -> None:
        """
        Store validation result in cache.

        Args:
            key: Cache key
            result: ValidationResult to cache
        """
        if self.backend == "none":
            return

        if self.backend == "memory":
            self._set_memory(key, result)
        elif self.backend == "sqlite":
            self._set_sqlite(key, result)
        elif self.backend == "json":
            self._set_json(key, result)

    def clear(self) -> None:
        """Clear all cache entries"""
        if self.backend == "memory":
            self.memory_cache.clear()
        elif self.backend == "sqlite":
            conn = sqlite3.connect(self.cache_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM validation_cache")
            conn.commit()
            conn.close()
        elif self.backend == "json":
            with open(self.cache_path, 'w') as f:
                json.dump({}, f)

    def size(self) -> int:
        """Get number of cached entries"""
        if self.backend == "memory":
            return len(self.memory_cache)
        elif self.backend == "sqlite":
            conn = sqlite3.connect(self.cache_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM validation_cache WHERE expires_at > ?", (datetime.now(),))
            count = cursor.fetchone()[0]
            conn.close()
            return count
        elif self.backend == "json":
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                return len(data)

        return 0

    def _get_memory(self, key: str) -> Optional[ValidationResult]:
        """Get from memory cache"""
        entry = self.memory_cache.get(key)

        if entry:
            expires_at = entry['expires_at']
            if datetime.fromisoformat(expires_at) > datetime.now():
                return ValidationResult(**entry['result'])

            # Expired, remove it
            del self.memory_cache[key]

        return None

    def _set_memory(self, key: str, result: ValidationResult) -> None:
        """Set in memory cache"""
        expires_at = datetime.now() + timedelta(days=self.ttl_days)

        self.memory_cache[key] = {
            'result': result.model_dump(),
            'expires_at': expires_at.isoformat()
        }

    def _get_sqlite(self, key: str) -> Optional[ValidationResult]:
        """Get from SQLite cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT result FROM validation_cache WHERE key = ? AND expires_at > ?",
            (key, datetime.now())
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            result_dict = json.loads(row[0])
            return ValidationResult(**result_dict)

        return None

    def _set_sqlite(self, key: str, result: ValidationResult) -> None:
        """Set in SQLite cache"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        created_at = datetime.now()
        expires_at = created_at + timedelta(days=self.ttl_days)

        cursor.execute(
            """
            INSERT OR REPLACE INTO validation_cache (key, result, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, json.dumps(result.model_dump()), created_at, expires_at)
        )

        conn.commit()
        conn.close()

    def _get_json(self, key: str) -> Optional[ValidationResult]:
        """Get from JSON file cache"""
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)

            entry = data.get(key)

            if entry:
                expires_at = datetime.fromisoformat(entry['expires_at'])
                if expires_at > datetime.now():
                    return ValidationResult(**entry['result'])

                # Expired, remove it
                del data[key]
                with open(self.cache_path, 'w') as f:
                    json.dump(data, f)

        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return None

    def _set_json(self, key: str, result: ValidationResult) -> None:
        """Set in JSON file cache"""
        try:
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        expires_at = datetime.now() + timedelta(days=self.ttl_days)

        data[key] = {
            'result': result.model_dump(),
            'expires_at': expires_at.isoformat()
        }

        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        if self.backend == "sqlite":
            conn = sqlite3.connect(self.cache_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM validation_cache WHERE expires_at <= ?", (datetime.now(),))
            count = cursor.rowcount

            conn.commit()
            conn.close()

            return count

        elif self.backend == "json":
            with open(self.cache_path, 'r') as f:
                data = json.load(f)

            original_count = len(data)
            now = datetime.now()

            data = {
                k: v for k, v in data.items()
                if datetime.fromisoformat(v['expires_at']) > now
            }

            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)

            return original_count - len(data)

        return 0
