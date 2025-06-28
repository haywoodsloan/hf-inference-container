import base64
import logging
import os

from optimum.pipelines import pipeline
from azure.monitor.opentelemetry import configure_azure_monitor
from fastapi import FastAPI, Response, status, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exception_handlers import http_exception_handler

task_type = os.environ.get("TASK_TYPE")
model_name = os.environ.get("MODEL_NAME")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Setup logging to app insights
try:
    configure_azure_monitor(logger_name="app")
except Exception as e:
    log.warning(f"Failed to configure azure monitoring: {e}")

# Preload the model
try:
    log.info(f"Loading inference model, task: {task_type}, model: {model_name}")
    inference = pipeline(task=task_type, model=model_name, accelerator="ort")
    log.info(f"Model loaded successfully")
except Exception as e:
    log.error(f"Loading model failed: ${e}")
    raise e

# Initialize the server
app = FastAPI()


@app.options("/invoke")
def invoke_options():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/invoke")
def invoke_post(body=Body()):
    log.info(f"Invoking transformer pipeline, task: {task_type}, model: {model_name}")

    if isinstance(body, bytes):
        input = base64.b64encode(body).decode("ascii")
        log.info("Using binary input as a base64 string for model")
    else:
        input = body
        log.info("Using JSON input for model")

    try:
        output = inference(input)
        return JSONResponse(output, status_code=200)
    except Exception as e:
        log.warning(f"Inference failed: ${e}")
        return PlainTextResponse(f"[INFERENCE FAILED]: {e}", status_code=500)
