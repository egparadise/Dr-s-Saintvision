"""
Database Manager for DR-Saintvision
Handles storage and retrieval of debate results
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for storing debate history and results"""

    def __init__(self, db_path: str = "dr_saintvision.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Debates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS debates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT UNIQUE NOT NULL,
                    query TEXT NOT NULL,
                    search_result TEXT,
                    reasoning_result TEXT,
                    synthesis_result TEXT,
                    confidence_scores TEXT,
                    debate_time REAL,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'anonymous'
                )
            ''')

            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_debates_timestamp
                ON debates(timestamp DESC)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_debates_user
                ON debates(user_id)
            ''')

            # Metrics table for tracking performance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (debate_id) REFERENCES debates(id)
                )
            ''')

            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id INTEGER,
                    rating INTEGER,
                    comment TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (debate_id) REFERENCES debates(id)
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_debate(
        self,
        query_id: str,
        query: str,
        search_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        synthesis_result: Dict[str, Any],
        confidence_scores: Dict[str, float],
        debate_time: float,
        status: str,
        user_id: str = "anonymous"
    ) -> int:
        """Save a debate result to the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO debates (
                    query_id, query, search_result, reasoning_result,
                    synthesis_result, confidence_scores, debate_time,
                    status, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query_id,
                query,
                json.dumps(search_result, ensure_ascii=False),
                json.dumps(reasoning_result, ensure_ascii=False),
                json.dumps(synthesis_result, ensure_ascii=False),
                json.dumps(confidence_scores),
                debate_time,
                status,
                user_id
            ))

            conn.commit()
            logger.info(f"Saved debate: {query_id}")
            return cursor.lastrowid

    def get_debate(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific debate by query_id"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM debates WHERE query_id = ?
            ''', (query_id,))

            row = cursor.fetchone()

            if row:
                return self._row_to_dict(row)
            return None

    def get_debate_by_id(self, debate_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific debate by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM debates WHERE id = ?
            ''', (debate_id,))

            row = cursor.fetchone()

            if row:
                return self._row_to_dict(row)
            return None

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary"""
        result = dict(row)

        # Parse JSON fields
        for field in ['search_result', 'reasoning_result', 'synthesis_result', 'confidence_scores']:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    pass

        return result

    def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get debate history for a user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT query_id, query, confidence_scores, debate_time,
                       status, timestamp
                FROM debates
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('confidence_scores'):
                    try:
                        result['confidence_scores'] = json.loads(result['confidence_scores'])
                    except json.JSONDecodeError:
                        pass
                results.append(result)

            return results

    def get_recent_debates(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent debates"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT query_id, query, confidence_scores, debate_time,
                       status, timestamp, user_id
                FROM debates
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('confidence_scores'):
                    try:
                        result['confidence_scores'] = json.loads(result['confidence_scores'])
                    except json.JSONDecodeError:
                        pass
                results.append(result)

            return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total debates
            cursor.execute("SELECT COUNT(*) FROM debates")
            total = cursor.fetchone()[0]

            # Completed debates
            cursor.execute("SELECT COUNT(*) FROM debates WHERE status = 'completed'")
            completed = cursor.fetchone()[0]

            # Average metrics
            cursor.execute('''
                SELECT
                    AVG(debate_time) as avg_time,
                    MIN(debate_time) as min_time,
                    MAX(debate_time) as max_time
                FROM debates
                WHERE status = 'completed'
            ''')
            time_stats = cursor.fetchone()

            # Average confidence
            cursor.execute('''
                SELECT confidence_scores FROM debates
                WHERE status = 'completed'
            ''')

            confidences = []
            for row in cursor.fetchall():
                if row[0]:
                    try:
                        scores = json.loads(row[0])
                        if 'overall' in scores:
                            confidences.append(scores['overall'])
                    except json.JSONDecodeError:
                        pass

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return {
                "total_debates": total,
                "completed_debates": completed,
                "failed_debates": total - completed,
                "success_rate": completed / total if total > 0 else 0,
                "average_time": time_stats[0] or 0,
                "min_time": time_stats[1] or 0,
                "max_time": time_stats[2] or 0,
                "average_confidence": avg_confidence
            }

    def save_feedback(
        self,
        debate_id: int,
        rating: int,
        comment: str = ""
    ):
        """Save user feedback for a debate"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO feedback (debate_id, rating, comment)
                VALUES (?, ?, ?)
            ''', (debate_id, rating, comment))

            conn.commit()

    def save_metric(
        self,
        debate_id: int,
        metric_name: str,
        metric_value: float
    ):
        """Save a metric for a debate"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO metrics (debate_id, metric_name, metric_value)
                VALUES (?, ?, ?)
            ''', (debate_id, metric_name, metric_value))

            conn.commit()

    def search_debates(
        self,
        keyword: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search debates by keyword"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT query_id, query, confidence_scores, debate_time,
                       status, timestamp
                FROM debates
                WHERE query LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (f"%{keyword}%", limit))

            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('confidence_scores'):
                    try:
                        result['confidence_scores'] = json.loads(result['confidence_scores'])
                    except json.JSONDecodeError:
                        pass
                results.append(result)

            return results

    def delete_debate(self, query_id: str) -> bool:
        """Delete a debate by query_id"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                DELETE FROM debates WHERE query_id = ?
            ''', (query_id,))

            conn.commit()
            return cursor.rowcount > 0

    def cleanup_old_debates(self, days: int = 30) -> int:
        """Delete debates older than specified days"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                DELETE FROM debates
                WHERE timestamp < datetime('now', ?)
            ''', (f'-{days} days',))

            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} old debates")
            return deleted
