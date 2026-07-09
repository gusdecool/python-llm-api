from dbm import sqlite3
import sqlite3
from app.config import DATABASE_URL
from app.log import logger
from contextlib import contextmanager


def init_db(conn: sqlite3.Connection) -> None:
    """
    Initializes the SQLite database. Create table jobs if it doesn't exist
    """
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT,
                status varcar(10) DEFAULT done,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if table is empty to seed it
        cursor.execute("SELECT COUNT(*) FROM llm_jobs")
        if cursor.fetchone()[0] == 0:
            cursor.executemany("""
                INSERT INTO llm_jobs (prompt, response, status)
                VALUES (?, ?, ?)
            """, [
                ("what is E=mc2", None, "queue"),
                ("who is the genous", "Its you Budi", "done")
            ])
            conn.commit()
            logger.info("Database initialized and seeded.")
        else:
            logger.info("Database already initialized.")
    finally:
        conn.close()


@contextmanager
def get_db_conn():
    """
    Get db connection
    """
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()