"""
ConfTest GitHub Webhook API Route.

Endpoint: POST /api/v1/github/webhook
Receives GitHub pull_request events, executes selective test selection,
and posts automated rationale reports to PRs.
"""

from typing import Any, Dict
from fastapi import APIRouter, Header, HTTPException, Request, Response, status, Depends
from sqlalchemy.orm import Session

from conftest.config import settings
from conftest.db.session import get_db
from conftest.db import crud
from conftest.engine.selector_engine import ConfTestEngine
from conftest.github.pr_bot import GitHubPRBot
from conftest.explainability.rules import RuleBasedExplainer
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/github", tags=["GitHub Integration"])

_bot_instance = GitHubPRBot()
_engine_instance: Dict[str, ConfTestEngine] = {}


def get_engine_for_path(repo_path: str = "./tests/sample_suite") -> ConfTestEngine:
    if repo_path not in _engine_instance:
        _engine_instance[repo_path] = ConfTestEngine(
            repo_root=repo_path,
            ensemble_path="./models/ensembles/5_seed_lgbm",
            calibrator_path="./models/calibrator.joblib",
            policy_config_path="./models/policy_config.json",
        )
    return _engine_instance[repo_path]


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def handle_github_webhook(
    request: Request,
    db: Session = Depends(get_db),
    x_github_event: str = Header(default="ping", alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header(default="", alias="X-Hub-Signature-256"),
) -> Dict[str, Any]:
    """
    Process incoming GitHub webhooks.

    Validates HMAC signature and processes 'pull_request' actions (opened, synchronize, reopened).
    """
    body_bytes = await request.body()

    # 1. Verify HMAC SHA-256 signature
    if settings.github_webhook_secret and not _bot_instance.verify_webhook_signature(body_bytes, x_hub_signature_256):
        logger.warning("Rejected webhook request: Invalid HMAC SHA-256 signature.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Hub-Signature-256 signature.",
        )

    # 2. Handle Ping event
    if x_github_event == "ping":
        logger.info("Received GitHub ping event. Webhook configured successfully.")
        return {"status": "pong", "message": "ConfTest webhook active."}

    # 3. Handle Pull Request events
    if x_github_event == "pull_request":
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload.")

        action = payload.get("action")
        if action not in ("opened", "synchronize", "reopened"):
            return {"status": "ignored", "action": action, "message": f"Action '{action}' does not require test selection."}

        pr = payload.get("pull_request", {})
        repo_info = payload.get("repository", {})
        repo_full_name = repo_info.get("full_name", "unknown/repo")
        pull_number = payload.get("number") or pr.get("number")
        head_sha = pr.get("head", {}).get("sha", "HEAD")
        pr_title = pr.get("title", "Pull Request")

        logger.info(f"Processing PR #{pull_number} ({pr_title}) for {repo_full_name} at commit {head_sha[:8]}")

        # Register repository if not existing
        repo_record = crud.get_or_create_repository(
            db=db,
            full_name=repo_full_name,
            url=repo_info.get("html_url", f"https://github.com/{repo_full_name}"),
            local_path="./tests/sample_suite",
        )

        # Mock / extract changed files
        changed_files = [
            {"file_path": "src_app/auth.py", "change_type": "M", "lines_added": 20, "lines_deleted": 5}
        ]

        engine = get_engine_for_path("./tests/sample_suite")
        outcome = engine.analyze_and_select(
            commit_sha=head_sha,
            changed_files=changed_files,
            commit_message=pr_title,
            budget_ratio=0.25,
            db=db,
            repository_id=repo_record.id,
            execute=False,
        )

        # Generate PR Markdown summary
        rule_explainer = RuleBasedExplainer()
        markdown_body = rule_explainer.generate_commit_markdown_summary(
            commit_sha=head_sha,
            decision_dict=outcome,
            top_tests=outcome.get("ranked_tests", []),
        )

        # Post comment to GitHub PR
        bot_resp = _bot_instance.post_or_update_pr_comment(
            repo_full_name=repo_full_name,
            pull_number=pull_number,
            markdown_body=markdown_body,
        )

        return {
            "status": "processed",
            "pull_number": pull_number,
            "commit_sha": head_sha,
            "decision_mode": outcome["decision_mode"],
            "selected_count": outcome["selected_count"],
            "total_count": outcome["total_count"],
            "abstained": outcome["abstained"],
            "comment_posted": bot_resp is not None,
        }

    return {"status": "ignored", "event": x_github_event}
