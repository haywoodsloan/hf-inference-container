from transformers import pipeline

import json
import base64
import azure.functions as func
import logging
import os

task_type = os.environ.get("TASK_TYPE")
model_name = os.environ.get("MODEL_NAME")

inference = pipeline(task=task_type, model=model_name, use_fast=True)
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="invoke", methods=["POST"])
def invoke(req: func.HttpRequest) -> func.HttpResponse:
    logging.info(
        f"Invoking Transformer pipeline, task: {task_type}, model: {model_name}"
    )

    try:
        input = req.get_json()
        logging.info("Using JSON input for model")
    except:
        input = base64.b64encode(req.get_body()).decode("ascii")
        logging.info("Using binary input as a base64 string for model")

    try:
        output = inference(input)
        return func.HttpResponse(json.dumps(output), status_code=200)
    except Exception as e:
        return func.HttpResponse(f"{e}", status_code=500)
