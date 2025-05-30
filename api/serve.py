#!/usr/bin/env python3
"""FastAPI server for Mistral NER model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from src.config import Config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.inference import load_model_for_inference, predict_entities
from src.config import Config
from src.utils import setup_logging


# Request/Response models
class NERRequest(BaseModel):
    texts: list[str]
    batch_size: int | None = 1


class Entity(BaseModel):
    text: str
    type: str
    start: int
    end: int


class NERResponse(BaseModel):
    text: str
    words: list[str]
    labels: list[str]
    entities: list[Entity]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Global variables
app = FastAPI(title="Mistral NER API", version="1.0.0")
model: PreTrainedModel | None = None
tokenizer: PreTrainedTokenizerBase | None = None
config: Config | None = None
device: str = "cuda" if torch.cuda.is_available() else "cpu"
logger: Any = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    global model, tokenizer, config, logger

    logger = setup_logging()
    logger.info("Starting Mistral NER API server...")

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config = Config.from_yaml(str(config_path))

    # Model path from environment variable or default
    import os

    model_path = os.getenv("NER_MODEL_PATH", "./mistral-ner-finetuned-final")
    base_model = os.getenv("NER_BASE_MODEL", "mistralai/Mistral-7B-v0.3")

    try:
        logger.info(f"Loading model from {model_path}")
        model, tokenizer = load_model_for_inference(model_path=model_path, base_model=base_model, config=config)
        model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup, but model will be None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy", model_loaded=model is not None, device=device
    )


@app.post("/predict", response_model=list[NERResponse])
async def predict(request: NERRequest):
    """Predict named entities in texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per request")

    try:
        # Run prediction
        if config is None:
            raise HTTPException(status_code=503, detail="Config not loaded")

        predictions = predict_entities(
            model=model,
            tokenizer=tokenizer,
            texts=request.texts,
            label_names=config.data.label_names,
            device=device,
            batch_size=request.batch_size or 1,
        )

        # Convert to response format
        responses = []
        for pred in predictions:
            responses.append(
                NERResponse(
                    text=pred["text"],
                    words=pred["words"],
                    labels=pred["labels"],
                    entities=[Entity(**e) for e in pred["entities"]],
                )
            )

        return responses

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict_simple")
async def predict_simple(texts: list[str]) -> list[dict[str, Any]]:
    """Simple prediction endpoint that returns only entities."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if config is None:
            raise HTTPException(status_code=503, detail="Config not loaded")

        predictions = predict_entities(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            label_names=config.data.label_names,
            device=device,
            batch_size=1,
        )

        # Return simplified format
        results = []
        for pred in predictions:
            results.append({"text": pred["text"], "entities": pred["entities"]})

        return results

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def main() -> None:
    """Run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Mistral NER API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    uvicorn.run("api.serve:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
