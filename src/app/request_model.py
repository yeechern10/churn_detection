from fastapi import FastAPI, HTTPException
from pydantic import create_model
from enum import Enum

from app.logger import logger

def create_request_model(schema):
    logger.info("Creating request model from schema...")
    fields = {}

    for col in schema["features"]["numerical"]:
        if 'int' in col["type"]:
            fields[col["name"]] = (int, ...)
        elif 'float' in col["type"]:
            fields[col["name"]] = (float, ...)

    for col in schema["features"]["categorical"]:
        name = col["name"]
        values = col.get("values")
        if values:
            enum_cls = Enum(
                name,
                {f"{name}_{v}": v for v in values}
            )
            fields[name] = (enum_cls, ...)
        else:
            fields[name] = (str, ...)
    try:
        model = create_model("ChurnRequest", **fields)
        logger.info("Request model created successfully")
        return model
    except Exception as e:
        logger.error("Error creating request model", exc_info=True)
        raise e