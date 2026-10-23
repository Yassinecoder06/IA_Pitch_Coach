"""In-memory exponential backoff for secret guessing (passphrase, profile PINs)."""

from __future__ import annotations

import time
from typing import Dict, Tuple


class Backoff:
    """After `free_attempts` failures for a key, each further failure doubles the lockout."""

    def __init__(self, free_attempts: int = 5, max_lockout_seconds: int = 15 * 60):
        self.free_attempts = free_attempts
        self.max_lockout_seconds = max_lockout_seconds
        self._state: Dict[str, Tuple[int, float]] = {}  # key -> (failures, locked_until)

    def retry_after(self, key: str = "") -> int:
        """Seconds until the next attempt is allowed (0 when allowed now)."""
        _, locked_until = self._state.get(key, (0, 0.0))
        remaining = locked_until - time.monotonic()
        return int(remaining) + 1 if remaining > 0 else 0

    def fail(self, key: str = "") -> None:
        failures, locked_until = self._state.get(key, (0, 0.0))
        failures += 1
        if failures >= self.free_attempts:
            lockout = min(self.max_lockout_seconds, 2 ** (failures - self.free_attempts + 1))
            locked_until = time.monotonic() + lockout
        self._state[key] = (failures, locked_until)

    def succeed(self, key: str = "") -> None:
        self._state.pop(key, None)

    def reset(self) -> None:
        self._state.clear()
