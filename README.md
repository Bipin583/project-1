# ConfTest: Confidence-Calibrated Regression Test Selection with Selective Prediction for CI/CD Optimization

**APJ Abdul Kalam Technological University (KTU) | Final-Year B.Tech CSE Major Project**

---

## 📄 Key Documentation & Proposal Artifacts
- **HTML Printable / PDF-Ready Proposal:** [`ConfTest_Concept_Document_KTU.html`](./ConfTest_Concept_Document_KTU.html)
- **Markdown Specification:** [`ConfTest_Concept_Document_KTU.md`](./ConfTest_Concept_Document_KTU.md)
- **Google Colab Training Pipeline:** [`notebooks/ConfTest_Colab_Training_Pipeline.ipynb`](./notebooks/ConfTest_Colab_Training_Pipeline.ipynb)
- **Master Specification Artifact:** `conftest_complete_project_specification.md`

---

## ☁️ Google Colab Training & Artifact Generation

The training and calibration pipeline is designed to run seamlessly on Google Colab (with free GPU/CPU):
1. Open [`notebooks/ConfTest_Colab_Training_Pipeline.ipynb`](./notebooks/ConfTest_Colab_Training_Pipeline.ipynb) in **Google Colab**.
2. Run all cells to:
   - Train the LightGBM test failure scoring model on 25,000+ commit-test samples.
   - Fit post-hoc Temperature Scaling ($T \approx 1.04$) and Venn-Abers uncertainty intervals.
   - Run the 8-baseline comparison experiment.
   - Export trained model weights (`conftest_model.pkl`) and parameters (`conftest_metadata.json`).
3. Download the exported artifacts and place them in the project root to power the local CI runner.

Alternatively, execute the standalone training script locally or in cloud terminals:
```bash
python src/models/colab_trainer.py
```

---

## 🚀 Quickstart & Local CI Execution

### 1. Environment Setup
```powershell
# Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Run Test Suite
```powershell
python -m pytest -v
```

### 3. Start FastAPI Backend (Port 8000)
```powershell
python -m uvicorn conftest.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser for Swagger documentation.

### 4. Launch Interactive Streamlit Research Dashboard (Port 8501)
```powershell
python -m streamlit run dashboard/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 👥 4-Member Technical Module Ownership
- **Member 1 (Data Engineering & CI/CD):** GitHub Actions webhook, runner CLI (`conftest_cli/`), SQLite/PostgreSQL schema, temporal data splitting.
- **Member 2 (AST & Feature Engineering):** Tree-sitter AST diffing (`src/features/ast_parser.py`), dependency call-graph extraction, cyclomatic complexity delta.
- **Member 3 (ML, Calibration & Abstention):** LightGBM classifier, temperature scaling, Venn-Abers calibration (`src/models/calibration.py`), Colab training notebook (`notebooks/`).
- **Member 4 (Dashboard & Statistics):** FastAPI analytics backend (`src/dashboard/`), PR comment bot, SHAP explainability, Wilcoxon signed-rank & Cliff's delta statistical evaluation.

---

## 💻 Hardware Compatibility
- Tested and verified for **ASUS ROG Strix G16** & **Google Colab Free Tier**.
- **Budget Tier:** ₹0.00 (Zero cloud compute / API dependency).
