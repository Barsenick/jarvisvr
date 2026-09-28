"""Spatial Memory Store for JarvisVR.
Handles the persistence and retrieval of real-world object coordinates.
"""
from __future__ import annotations

import sqlite3
import json
import logging
from dataclasses import dataclass
from typing import Optional, Any

log = logging.getLogger("jarvis.memory.spatial")

@dataclass
class SpatialRecord:
    name: str
    position: list[float]
    anchor: str
    source: str
    confidence: float
    timestamp: float

class SpatialMemory:
    def __init__(self, db_path: str = "spatial_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    position TEXT NOT NULL,
                    anchor TEXT NOT NULL,
                    source TEXT,
                    confidence REAL,
                    timestamp REAL
                )
            """)
            conn.commit()

    def remember(self, name: str, position: list[float], anchor: str = "world", source: str = "vision", confidence: float = 1.0, timestamp: float = 0.0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO objects (name, position, anchor, source, confidence, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (name, json.dumps(position), anchor, source, confidence, timestamp)
            )
            conn.commit()

    def recall(self, name: str) -> Optional[SpatialRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name, position, anchor, source, confidence, timestamp FROM objects WHERE name = ? ORDER BY timestamp DESC LIMIT 1",
                (name,)
            )
            row = cursor.fetchone()
            if row:
                return SpatialRecord(
                    name=row[0],
                    position=json.loads(row[1]),
                    anchor=row[2],
                    source=row[3],
                    confidence=row[4],
                    timestamp=row[5]
                )
        return None

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM objects")
            conn.commit()

# Global instance for the agent to use
spatial_db = SpatialMemory()
