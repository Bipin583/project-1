"""
ConfTest GitHub PR Bot & Webhook Security Subsystem.

Handles HMAC SHA-256 webhook signature verification, GitHub REST API client interactions,
and automatic PR comment generation with idempotency (updating existing bot comments).
"""

import hmac
import hashlib
from typing import Any, Dict, List, Optional
import httpx

from conftest.config import settings
from conftest.logging_config import get_logger

logger = get_logger(__name__)

CONFTEST_BOT_SIGNATURE = "<!-- conftest-ci-bot-comment -->"


class GitHubPRBot:
    """Interacts with GitHub REST API to post test selection decisions directly on PRs."""

    def __init__(self, github_token: Optional[str] = None, webhook_secret: Optional[str] = None):
        self.github_token = github_token or settings.github_token
        self.webhook_secret = webhook_secret or settings.github_webhook_secret

    def verify_webhook_signature(self, payload_bytes: bytes, signature_header: Optional[str]) -> bool:
        """
        Verify GitHub HMAC SHA-256 webhook payload signature.

        Args:
            payload_bytes: Raw request body in bytes.
            signature_header: X-Hub-Signature-256 header (e.g. 'sha256=abcdef...').

        Returns:
            True if signature matches, False otherwise.
        """
        if not self.webhook_secret:
            # If no secret configured in test/dev environment, allow
            logger.warning("No GITHUB_WEBHOOK_SECRET configured. Skipping signature verification in dev mode.")
            return True

        if not signature_header or not signature_header.startswith("sha256="):
            return False

        expected_sig = signature_header.split("sha256=")[-1].strip()
        mac = hmac.new(
            self.webhook_secret.encode("utf-8"),
            msg=payload_bytes,
            digestmod=hashlib.sha256,
        )
        calculated_sig = mac.hexdigest()
        return hmac.compare_digest(calculated_sig, expected_sig)

    def post_or_update_pr_comment(
        self,
        repo_full_name: str,
        pull_number: int,
        markdown_body: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Post a new comment or update an existing ConfTest comment on a Pull Request.

        Args:
            repo_full_name: e.g. 'owner/repo'
            pull_number: PR issue number
            markdown_body: Formatted Markdown report

        Returns:
            JSON response from GitHub API or None if token not configured.
        """
        if not self.github_token:
            logger.info(f"GITHUB_TOKEN not configured. Skipping live PR comment for {repo_full_name}#{pull_number}.")
            return None

        # Add unique bot watermark for idempotency
        body_with_tag = f"{CONFTEST_BOT_SIGNATURE}\n{markdown_body}"
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        base_url = f"https://api.github.com/repos/{repo_full_name}/issues/{pull_number}/comments"

        try:
            with httpx.Client(timeout=10.0) as client:
                # 1. Search for existing ConfTest comment to update
                list_resp = client.get(base_url, headers=headers)
                if list_resp.status_code == 200:
                    comments = list_resp.json()
                    for c in comments:
                        if CONFTEST_BOT_SIGNATURE in c.get("body", ""):
                            # Update existing comment
                            comment_id = c["id"]
                            update_url = f"https://api.github.com/repos/{repo_full_name}/issues/comments/{comment_id}"
                            patch_resp = client.patch(update_url, headers=headers, json={"body": body_with_tag})
                            if patch_resp.status_code == 200:
                                logger.info(f"Updated ConfTest comment #{comment_id} on {repo_full_name}#{pull_number}")
                                return patch_resp.json()

                # 2. Create new comment if none exists
                post_resp = client.post(base_url, headers=headers, json={"body": body_with_tag})
                if post_resp.status_code == 201:
                    logger.info(f"Created ConfTest comment on {repo_full_name}#{pull_number}")
                    return post_resp.json()
                else:
                    logger.error(f"Failed to post PR comment: {post_resp.status_code} {post_resp.text}")
                    return None
        except Exception as exc:
            logger.error(f"Error calling GitHub API: {exc}")
            return None
