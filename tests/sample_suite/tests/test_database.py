"""
Unit tests for In-Memory Database Module
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src_app.database import SimpleDB

def test_db_set_and_get():
    db = SimpleDB()
    db.set("user:1", {"name": "Alice", "role": "admin"})
    assert db.get("user:1") == {"name": "Alice", "role": "admin"}
    assert db.get("user:2") is None

def test_db_delete():
    db = SimpleDB()
    db.set("session:abc", "token_123")
    assert db.delete("session:abc") is True
    assert db.get("session:abc") is None
    assert db.delete("session:abc") is False
