import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path("data/garmin_coach.db")

def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Table for raw data and analysis results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                garmin_data TEXT,
                metrics_summary TEXT,
                physiology_summary TEXT,
                activity_summary TEXT,
                synthesis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for training plans
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                plan_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully at %s", DB_PATH)
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
    finally:
        conn.close()

def save_analysis_run(
    execution_id: str,
    user_id: str,
    date: str,
    garmin_data: dict[str, Any] | None = None,
    metrics_summary: str | None = None,
    physiology_summary: str | None = None,
    activity_summary: str | None = None,
    synthesis_result: str | None = None
) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO analysis_runs 
            (execution_id, user_id, date, garmin_data, metrics_summary, physiology_summary, activity_summary, synthesis_result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution_id,
            user_id,
            date,
            json.dumps(garmin_data) if garmin_data else None,
            metrics_summary,
            physiology_summary,
            activity_summary,
            synthesis_result
        ))
        conn.commit()
        logger.info("Saved analysis run %s for user %s", execution_id, user_id)
    except Exception as e:
        logger.error("Failed to save analysis run: %s", e)
    finally:
        conn.close()

def save_training_plan(user_id: str, plan_type: str, content: str) -> None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_plans (user_id, plan_type, content)
            VALUES (?, ?, ?)
        """, (user_id, plan_type, content))
        conn.commit()
        logger.info("Saved %s plan for user %s", plan_type, user_id)
    except Exception as e:
        logger.error("Failed to save training plan: %s", e)
    finally:
        conn.close()

def load_latest_training_plan(user_id: str, plan_type: str) -> str | None:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content FROM training_plans 
            WHERE user_id = ? AND plan_type = ? 
            ORDER BY created_at DESC LIMIT 1
        """, (user_id, plan_type))
        row = cursor.fetchone()
        if row:
            return row["content"]
        return None
    except Exception as e:
        logger.error("Failed to load training plan: %s", e)
        return None
    finally:
        conn.close()

# Initialize database on module import
init_db()
