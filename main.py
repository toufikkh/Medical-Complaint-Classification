"""
Medical Complaint Classifier - FastAPI service
==============================================
Run with:   uvicorn main:app --reload
Docs:       http://127.0.0.1:8000/docs   (interactive Swagger UI)

Endpoints
---------
GET  /            short info about the service
GET  /health      is the model loaded?
GET  /categories  list of categories the model can predict
POST /predict     body: {"text": "...", "top_k": 3} -> top-k categories with probabilities

This is NOT a diagnosis tool. It classifies text into predefined categories only.
Inference only: the model is trained in notebooks/analysis.ipynb.
"""

import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pipeline.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"

LOW_CONFIDENCE = 0.50
MAX_CHARS = 2000

DISCLAIMER = (
    "Educational/research prototype. It classifies text into predefined medical categories "
    "and does not provide a medical diagnosis, emergency triage, or medical advice."
)

# The model is loaded once at start-up and stored here.
state = {"model": None, "meta": {}}


def clean_text(text: str) -> str:
    """Same cleaning as in the notebook and app.py (keep identical!)."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s.,'\-/%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load the trained pipeline."""
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            state["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        print(f"WARNING: model not found at {MODEL_PATH}. Run notebooks/analysis.ipynb first.")
    yield
    state["model"] = None


app = FastAPI(
    title="Medical Complaint Classifier API",
    description="Classifies a short medical complaint into a predefined medical category "
    "(top-k with probabilities). Educational prototype - not a diagnosis system.",
    version="1.0.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# Request / response schemas
# ----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        max_length=MAX_CHARS,
        description="The patient's complaint in free text.",
        examples=["I have red patches on my skin and severe itching."],
    )
    top_k: int = Field(3, ge=1, le=10, description="How many categories to return.")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not clean_text(value):
            raise ValueError("text must contain readable characters")
        return value


class CategoryScore(BaseModel):
    category: str
    probability: float = Field(..., description="Model score between 0 and 1")
    percentage: float = Field(..., description="Same score as a percentage")


class PredictResponse(BaseModel):
    input_text: str
    predictions: List[CategoryScore]
    low_confidence: bool
    disclaimer: str


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
def require_model():
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run notebooks/analysis.ipynb to create models/pipeline.joblib, "
            "then restart the server.",
        )
    return state["model"]


@app.get("/")
def root():
    return {
        "service": "Medical Complaint Classifier API",
        "docs": "/docs",
        "model_loaded": state["model"] is not None,
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/categories")
def categories():
    model = require_model()
    return {"categories": [str(c) for c in model.classes_], "metrics": state["meta"].get("metrics", {})}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = require_model()

    cleaned = clean_text(request.text)
    proba = model.predict_proba([cleaned])[0]

    k = min(request.top_k, len(model.classes_))
    top_idx = np.argsort(proba)[::-1][:k]

    predictions = [
        CategoryScore(
            category=str(model.classes_[i]),
            probability=round(float(proba[i]), 4),
            percentage=round(float(proba[i]) * 100, 2),
        )
        for i in top_idx
    ]

    return PredictResponse(
        input_text=request.text,
        predictions=predictions,
        low_confidence=predictions[0].probability < LOW_CONFIDENCE,
        disclaimer=DISCLAIMER,
    )
