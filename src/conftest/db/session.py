"""
Database Session and Engine Management.

Configures connection pooling, SQLite WAL journal mode, and FastAPI dependency injectors.
"""

from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from conftest.config import settings
from conftest.logging_config import get_logger

logger = get_logger(__name__)

# Determine if the connection is SQLite
is_sqlite = settings.database_url.startswith("sqlite")

# Engine connection arguments
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Create central engine
engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,
    connect_args=connect_args,
    pool_pre_ping=True,
)

# Enable SQLite foreign key enforcement and WAL journal mode
if is_sqlite:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a database session per request.
    Ensures safe commit/rollback and automatic session closure.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        logger.error(f"Database session rolled back due to error: {exc}")
        db.rollback()
        raise
    finally:
        db.close()
