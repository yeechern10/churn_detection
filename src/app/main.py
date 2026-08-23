from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager
import mlflow
from mlflow.tracking import MlflowClient
import json
import tempfile
import pandas as pd
from prometheus_client import generate_latest
import uvicorn
from pydantic import create_model
from enum import Enum
import os
from pathlib import Path
import time

from app.logger import logger
from app.middleware import PrometheusMiddleware
from app.routes import router
from app.request_model import create_request_model
from configuration import MLFLOW_TRACKING_URI, MODEL_NAME
from app.prom_metrics import SERVICE_INITIALIZATION_TIME


logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    with SERVICE_INITIALIZATION_TIME.time():
        logger.info("Loading model...")

        client = MlflowClient()
        model_version = None
        model_uri = None
        for i in range(10):
            try:
                logger.info(f"Trying search model in MLflow... attempt {i+1}")
                models = client.search_registered_models(filter_string=f"name='{MODEL_NAME}'")
                
                if not models:
                    logger.warning(f"Model '{MODEL_NAME}' not found in MLflow")
                    time.sleep(2)
                    continue
                
                registered_model = models[0]
                aliases = registered_model.aliases
                
                champion_version = aliases.get("champion")
                
                if champion_version is None:
                    logger.error(f"No version with 'champion' alias found for model '{MODEL_NAME}'")
                    continue

                logger.info(f"Found champion version: {champion_version}")
                
                model_version = client.get_model_version(MODEL_NAME, champion_version)
                model_uri = model_version.source
                break
            except Exception as e:
                logger.warning("Retrying MLflow...", e)
                time.sleep(2)
                
        else:
            logger.error(f"Failed to find model '{MODEL_NAME}' with 'champion' alias after multiple attempts")
            raise RuntimeError("Failed to get model from MLflow")
        
        
        
        for i in range(10):
            try:
                logger.info(f"Trying to load model from MLflow attempt {i+1}, uri: {model_uri}")
                model = mlflow.pyfunc.load_model(model_uri)
                break
            except Exception as e:
                logger.warning("Retrying MLflow...", e)
                time.sleep(2)

        if model is None:
            logger.error("Failed to load model from MLflow after multiple attempts")
            raise RuntimeError("Failed to load model from MLflow")
        
        # load schema
        run_id = model_version.run_id

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Downloading schema.json from MLflow run_id: {run_id} to temporary directory: {tmpdir}")
            schema_path = client.download_artifacts(run_id, "schema.json", dst_path=tmpdir)
            logger.info(f"Schema downloaded to: {schema_path}")
            with open(schema_path) as f:
                schema = json.load(f)
                logger.info(f"Schema loaded: {schema}")
                

        # store in app state for global access
        app.state.model = model
        app.state.schema = schema
        app.state.threshold = schema.get("threshold", 0.5)
        app.state.RequestModel = create_request_model(schema)

        logger.info("Model loaded")
    
    @app.post("/predict")
    async def predict(data: app.state.RequestModel):
        data_dict = {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in data.model_dump().items()
        }
        df = pd.DataFrame([data_dict])

        model = app.state.model
        threshold = app.state.threshold

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

    @app.get("/")
    async def health_check():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(), media_type="text/plain")
    
    yield

    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(PrometheusMiddleware)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()