import json
import base64
import azure.functions as func
import logging
import os

from transformers import pipeline

task_type = os.environ.get("TASK_TYPE")
model_name = os.environ.get("MODEL_NAME")

inference = None
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

    global inference
    if inference is None:
        try:
            logging.info("Loading model")
            inference = pipeline(task=task_type, model=model_name, device_map="auto")
            logging.info("Model loaded")
        except Exception as e:
            logging.info(f"Model loading failed: ${e}")
            return func.HttpResponse(f"[MODEL LOADING FAILED]: ${e}", status_code=500)
    else:
        logging.info("Using cached model")

    try:
        output = inference(input)
        return func.HttpResponse(json.dumps(output), status_code=200)
    except Exception as e:
        logging.info(f"Inference failed: ${e}")
        return func.HttpResponse(f"[INFERENCE FAILED]: {e}", status_code=500)
