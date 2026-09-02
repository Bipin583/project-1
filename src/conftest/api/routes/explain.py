"""
ConfTest Explainability API Route.

Endpoint: POST /api/v1/explain
Returns exact SHAP TreeExplainer feature attributions and developer reason cards.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from fastapi import APIRouter, HTTPException, status

from conftest.api.schemas import ExplainRequestSchema, ExplainResponseSchema, FeatureDriverSchema
from conftest.features.pipeline import FEATURE_NAMES
from conftest.models.lightgbm_model import LightGBMTestPredictor
from conftest.explainability.shap_explainer import ShapExplainer
from conftest.explainability.rules import RuleBasedExplainer
from conftest.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/explain", tags=["Model Explainability"])

_explainer_cache: Dict[str, ShapExplainer] = {}


def get_shap_explainer() -> ShapExplainer:
    """Load or retrieve cached SHAP explainer instance."""
    model_path = Path("./models/ensembles/5_seed_lgbm/member_1_seed_42.joblib")
    if not model_path.exists():
        fallback = Path("./models/ensembles/lgbm_v1.0.0_seed42.joblib")
        if fallback.exists():
            model_path = fallback
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trained ML model artifact not found. Run train_model.py or train_ensemble.py first.",
            )

    key = str(model_path)
    if key not in _explainer_cache:
        predictor = LightGBMTestPredictor.load(str(model_path))
        _explainer_cache[key] = ShapExplainer(predictor)
    return _explainer_cache[key]


@router.post("", response_model=ExplainResponseSchema, status_code=status.HTTP_200_OK)
def explain_test_prediction(payload: ExplainRequestSchema) -> ExplainResponseSchema:
    """
    Explain why a specific test received its failure risk score.

    Computes local Shapley values (SHAP) across all 32 features and formats natural language reasons.
    """
    try:
        explainer = get_shap_explainer()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize explainer: {str(exc)}",
        )

    # Convert features dict to ordered vector
    x_vec = np.array([float(payload.features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=np.float32)

    shap_breakdown = explainer.explain_instance(x_vec, top_k=payload.top_k)
    top_pos = [
        FeatureDriverSchema(
            feature=f["feature"],
            feature_value=f["feature_value"],
            shap_attribution=f["shap_attribution"],
            impact=f["impact"],
        )
        for f in shap_breakdown["top_risk_increasing_features"]
    ]
    top_neg = [
        FeatureDriverSchema(
            feature=f["feature"],
            feature_value=f["feature_value"],
            shap_attribution=f["shap_attribution"],
            impact=f["impact"],
        )
        for f in shap_breakdown["top_risk_decreasing_features"]
    ]

    rule_explainer = RuleBasedExplainer()
    card = rule_explainer.generate_test_reason_card(
        test_id=payload.test_id,
        feature_dict=payload.features,
        shap_drivers=shap_breakdown["top_risk_increasing_features"],
        confidence=shap_breakdown["predicted_probability"],
    )

    return ExplainResponseSchema(
        test_id=payload.test_id,
        predicted_probability=shap_breakdown["predicted_probability"],
        base_expected_value=shap_breakdown["base_expected_value"],
        risk_level=card["risk_level"],
        primary_reasons=card["primary_reasons"],
        top_risk_increasing_features=top_pos,
        top_risk_decreasing_features=top_neg,
    )
