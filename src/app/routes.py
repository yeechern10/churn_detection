import json
from fastapi import Depends, FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import Response
import pandas as pd
from typing import Annotated
from prometheus_client import generate_latest
from pydantic import BaseModel

from app.logger import logger
from app.request_model import create_request_model
from configuration import SCHEMA_FILE

router = APIRouter()

async def get_validated_data(request: Request) -> BaseModel:
    try:
        body = await request.json()
        
        if not hasattr(request.app.state, "request_model_class"):
            logger.error("Request model class not found in app state")
            raise HTTPException(status_code=500, detail="Request model not initialized")
        
        model_class = request.app.state.request_model_class
        return model_class(**body)
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

@router.post("/predict")
async def predict( request: Request, data: BaseModel = Depends(get_validated_data)):
    data_dict = {
        k: (v.value if hasattr(v, "value") else v)
        for k, v in data.model_dump().items()
    }
    df = pd.DataFrame([data_dict])

    model = request.app.state.model
    threshold = request.app.state.threshold

    # determine positive class index
    try:
        classes = model.classes_
        pos_index = list(classes).index("Yes")
        logger.info('Positive class index found at: %d', pos_index)
    except Exception:
        pos_index = 1
        logger.warning('Positive class index not found, defaulting to index 1')
        
    # predict
    try:
        prob = float(model.predict(df)[:, pos_index][0])
        pred = int(prob >= threshold)
        logger.info(f"Predicted probability: {prob}, Threshold: {threshold}, Prediction: {pred}")
    except Exception as e:
        logger.error("Error during prediction", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction error")

    return {
        "prediction": "Yes" if pred else "No",
        "probability": prob,
        "threshold": threshold
    }

@router.get("/")
async def health_check():
    return {"status": "ok"}

@router.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")