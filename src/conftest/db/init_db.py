"""
Database Initializer.

Creates all database tables and verifies connectivity.
Can be executed directly: `python -m conftest.db.init_db`
"""

import sys
from conftest.db.base import Base
from conftest.db.session import engine
from conftest.config import settings
from conftest.logging_config import get_logger

logger = get_logger(__name__)


def init_db() -> bool:
    """
    Initialize all database tables defined in the Base metadata.

    Returns:
        bool: True if initialization was successful.
    """
    try:
        # Ensure parent directory for SQLite file exists
        if settings.database_url.startswith("sqlite:///"):
            sqlite_path = settings.database_url.replace("sqlite:///", "")
            import os
            os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)

        logger.info(f"Initializing database at: {settings.database_url}")
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")
        return True
    except Exception as exc:
        logger.error(f"Failed to initialize database schema: {exc}", exc_info=True)
        return False


if __name__ == "__main__":
    success = init_db()
    sys.exit(0 if success else 1)
