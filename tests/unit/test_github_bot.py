"""
Unit and Integration tests for GitHub PR Bot and Webhook Handler.
"""

import hmac
import hashlib
import json
from fastapi.testclient import TestClient
import pytest

from conftest.config import settings
from conftest.github.pr_bot import GitHubPRBot


def test_hmac_signature_verification():
    """Verify HMAC SHA-256 signature calculation and verification."""
    secret = "my_super_secret_key_123"
    bot = GitHubPRBot(webhook_secret=secret)

    payload = b'{"action": "opened", "number": 42}'
    mac = hmac.new(secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
    valid_sig = f"sha256={mac.hexdigest()}"

    # 1. Valid signature
    assert bot.verify_webhook_signature(payload, valid_sig) is True

    # 2. Tampered payload
    assert bot.verify_webhook_signature(b'{"action": "opened", "number": 99}', valid_sig) is False

    # 3. Invalid signature header format
    assert bot.verify_webhook_signature(payload, "invalid_header") is False
    assert bot.verify_webhook_signature(payload, None) is False


def test_github_pr_bot_graceful_missing_token():
    """Verify bot safely handles missing GitHub token without crashing."""
    bot = GitHubPRBot(github_token=None)
    res = bot.post_or_update_pr_comment("owner/repo", 1, "## Test Report")
    assert res is None


def test_webhook_ping_event(client: TestClient):
    """Verify POST /api/v1/github/webhook handles 'ping' event with signature."""
    secret = settings.github_webhook_secret or "test_secret"
    payload = json.dumps({"zen": "Keep it logically awesome."}).encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
    sig_header = f"sha256={mac.hexdigest()}"

    resp = client.post(
        "/api/v1/github/webhook",
        headers={
            "X-GitHub-Event": "ping",
            "X-Hub-Signature-256": sig_header,
            "Content-Type": "application/json",
        },
        content=payload,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pong"


def test_webhook_pull_request_event(client: TestClient):
    """Verify POST /api/v1/github/webhook processes pull_request opened event."""
    secret = settings.github_webhook_secret or "test_secret"
    payload_dict = {
        "action": "opened",
        "number": 105,
        "pull_request": {
            "number": 105,
            "title": "feat: add secure password hashing",
            "head": {"sha": "c0ffee1234567890abcdef"},
        },
        "repository": {
            "full_name": "owner/sample-pr-repo",
            "html_url": "https://github.com/owner/sample-pr-repo",
        },
    }
    payload = json.dumps(payload_dict).encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), msg=payload, digestmod=hashlib.sha256)
    sig_header = f"sha256={mac.hexdigest()}"

    resp = client.post(
        "/api/v1/github/webhook",
        headers={
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": sig_header,
            "Content-Type": "application/json",
        },
        content=payload,
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "processed"
    assert data["pull_number"] == 105
    assert "decision_mode" in data
    assert data["decision_mode"] in ("FAST_SELECTED", "SAFE_FULL_SUITE")
