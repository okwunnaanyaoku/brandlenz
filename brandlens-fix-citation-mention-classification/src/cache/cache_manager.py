# coding: ascii
"""Caching utilities for BrandLens.""" 

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class CacheError(Exception):
    """Raised when cache operations fail."""


@dataclass
class _CacheEntry:
    value: Any
    expires_at: Optional[float]

    def is_expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


class CacheManager:
    """Three-tier cache (memory + disk) for BrandLens analytics."""

    def __init__(
        self,
        *,
        cache_dir: str,
        default_ttl: Optional[int] = 3600,
        enable_disk: bool = True,
        time_func: Optional[callable] = None,
    ) -> None:
        self._default_ttl = default_ttl
        self._enable_disk = enable_disk
        self._time = time_func or time.time
        self._memory: Dict[str, _CacheEntry] = {}
        self._cache_dir = Path(cache_dir)
        if self._enable_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        now = self._time()
        entry = self._memory.get(key)
        if entry:
            if entry.is_expired(now):
                self.invalidate(key)
            else:
                return entry.value

        if self._enable_disk:
            disk_entry = self._load_from_disk(key)
            if disk_entry and not disk_entry.is_expired(now):
                self._memory[key] = disk_entry
                return disk_entry.value
            elif disk_entry:
                self.invalidate(key)

        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = self._compute_expiry(ttl)
        entry = _CacheEntry(value=value, expires_at=expires_at)
        self._memory[key] = entry

        if self._enable_disk:
            self._persist_to_disk(key, entry)

    def get_many(self, keys: Iterable[str]) -> Dict[str, Any]:
        return {key: value for key in keys if (value := self.get(key)) is not None}

    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> None:
        for key, value in mapping.items():
            self.set(key, value, ttl=ttl)

    def invalidate(self, key: str) -> None:
        self._memory.pop(key, None)
        if self._enable_disk:
            path = self._path_for(key)
            if path.exists():
                path.unlink(missing_ok=True)

    def clear(self) -> None:
        self._memory.clear()
        if self._enable_disk and self._cache_dir.exists():
            for item in self._cache_dir.glob('*'):
                if item.is_file():
                    item.unlink(missing_ok=True)

    def _compute_expiry(self, ttl: Optional[int]) -> Optional[float]:
        if ttl is None:
            ttl = self._default_ttl
        if ttl is None:
            return None
        if ttl <= 0:
            return None
        return self._time() + ttl

    def _path_for(self, key: str) -> Path:
        safe_key = ''.join(ch for ch in key if ch.isalnum() or ch in ('-', '_'))
        if not safe_key:
            safe_key = str(abs(hash(key)))
        return self._cache_dir / f"{safe_key}.json"

    def _persist_to_disk(self, key: str, entry: _CacheEntry) -> None:
        path = self._path_for(key)
        try:
            payload = {
                "value": entry.value,
                "expires_at": entry.expires_at,
            }
            with path.open('w', encoding='utf-8') as handle:
                json.dump(payload, handle)
        except Exception as exc:  # pragma: no cover - defensive
            raise CacheError(f"Failed to persist cache entry for key '{key}'") from exc

    def _load_from_disk(self, key: str) -> Optional[_CacheEntry]:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            with path.open('r', encoding='utf-8') as handle:
                payload = json.load(handle)
            return _CacheEntry(value=payload.get('value'), expires_at=payload.get('expires_at'))
        except Exception:  # pragma: no cover - defensive
            return None


__all__ = ["CacheManager", "CacheError"]
