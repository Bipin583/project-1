"""
ConfTest Database Layer.

Exports base declarative metadata, session factories, ORM models, and CRUD operations.
"""

from conftest.db.base import Base
from conftest.db.session import engine, get_db, SessionLocal
from conftest.db.models import (
    Repository,
    Commit,
    ChangedFile,
    TestCase,
    TestRun,
    FeatureRecord,
    Prediction,
    SelectionDecision,
    Outcome,
)
from conftest.db import crud

__all__ = [
    "Base",
    "engine",
    "get_db",
    "SessionLocal",
    "Repository",
    "Commit",
    "ChangedFile",
    "TestCase",
    "TestRun",
    "FeatureRecord",
    "Prediction",
    "SelectionDecision",
    "Outcome",
    "crud",
]
