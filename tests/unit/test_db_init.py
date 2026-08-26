"""
Unit tests for database initialization, engine connectivity, and session generator.
"""

from sqlalchemy.orm import Session
from sqlalchemy import text
from conftest.db.init_db import init_db
from conftest.db.session import get_db


def test_init_db_execution():
    """Verify that init_db() successfully runs without errors."""
    success = init_db()
    assert success is True


def test_get_db_session_lifecycle(db_session: Session):
    """Verify that the get_db generator yields a valid session and allows queries."""
    result = db_session.execute(text("SELECT 1")).scalar()
    assert result == 1


def test_session_generator_context():
    """Verify standard get_db generator usage pattern."""
    gen = get_db()
    session = next(gen)
    assert isinstance(session, Session)
    # Ensure cleanup
    try:
        next(gen)
    except StopIteration:
        pass
