import base64
import logging
import os

from transformers import pipeline
from fastapi import FastAPI, Response, Request, status, Body, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exception_handlers import http_exception_handler

task_type = os.environ.get("TASK_TYPE")
model_name = os.environ.get("MODEL_NAME")
api_key = os.environ.get("API_KEY")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

app = FastAPI()

try:
    log.info(f"Loading inference model, task: {task_type}, model: {model_name}")
    inference = pipeline(task=task_type, model=model_name, device_map="auto")
    log.info(f"Model loaded successfully")
except Exception as e:
    log.info(f"Loading model failed: ${e}")
    raise e


@app.middleware("http")
async def check_auth(request: Request, call_next):
    if "Authorization" not in request.headers:
        exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return await http_exception_handler(request, exc)

    if request.headers["Authorization"] != f"Bearer {api_key}":
        exc = HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return await http_exception_handler(request, exc)

    return await call_next(request)


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
        log.info(f"Inference failed: ${e}")
        return PlainTextResponse(f"[INFERENCE FAILED]: {e}", status_code=500)
