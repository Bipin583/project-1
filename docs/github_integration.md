# ConfTest GitHub Actions & Webhook Integration

## 1. Overview
ConfTest integrates directly with GitHub Pull Requests and CI/CD pipelines via two complementary modes:
1. **GitHub Actions Workflow (`.github/workflows/conftest.yml`):** Native in-runner test selection executing pytest in PR jobs.
2. **GitHub Webhooks Bot (`/api/v1/github/webhook`):** Cloud backend receiving pull request events and posting automated developer Markdown rationale comments.

---

## 2. GitHub Actions Integration

Place `.github/workflows/conftest.yml` in your repository:
```yaml
name: ConfTest RTS CI
on: [pull_request]
jobs:
  conftest-rts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python scripts/select_tests.py --repo-path . --commit-sha "${{ github.event.pull_request.head.sha }}" --budget 0.25 --execute
```

---

## 3. GitHub Webhook Setup

### Webhook Configuration in GitHub Repository Settings:
- **Payload URL:** `https://your-domain.com/api/v1/github/webhook`
- **Content type:** `application/json`
- **Secret:** Generate a random 32-character string and save as `GITHUB_WEBHOOK_SECRET` in `.env`.
- **Events to trigger:** Select *"Pull requests"*.

### Security & Signature Verification:
Every incoming webhook is verified against `X-Hub-Signature-256` using HMAC SHA-256 before any code execution or database mutation occurs.

---

## 4. Pull Request Report Preview

When a developer opens or updates a Pull Request, ConfTest posts an automated summary card:

```markdown
## 🛡️ ConfTest CI Regression Test Selection Report
**Commit:** `c0ffee12` | **Status:** `FAST RTS (75.0% SAVINGS)`

### 📊 Decision Summary
- **Execution Mode:** `FAST_SELECTED`
- **Selected Tests:** **3 / 12** (75.0% test execution reduction)
- **Model Confidence:** `92.4%` | **Epistemic Uncertainty:** `0.0084`

### 🔍 Top High-Risk Selected Tests & Rationale
| Test Case | Risk Level | Calibrated Confidence | Primary Rationale |
| :--- | :---: | :---: | :--- |
| `tests/test_auth.py::test_jwt_login` | `HIGH` | `92.4%` | Direct Dependency: Test imports modified module `src/auth.py`<br>Recent Regression: Failed in 30% of last 10 runs |
```
