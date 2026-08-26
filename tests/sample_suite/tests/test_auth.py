"""
Unit tests for Auth Module
"""
import pytest
import sys
import os

# Ensure sample_suite root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src_app.auth import AuthService

def test_password_hashing():
    auth = AuthService()
    h1 = auth.hash_password("test")
    h2 = auth.hash_password("test")
    assert h1 == h2
    assert len(h1) == 64

def test_authentication_success():
    auth = AuthService()
    assert auth.authenticate("alice", "secret123") is True

def test_authentication_failure():
    auth = AuthService()
    assert auth.authenticate("alice", "wrongpass") is False
    assert auth.authenticate("nonexistent", "secret123") is False

def test_token_validation():
    auth = AuthService()
    assert auth.validate_token("Bearer_xyz123") is True
    assert auth.validate_token("Basic_xyz123") is False
    assert auth.validate_token("") is False
