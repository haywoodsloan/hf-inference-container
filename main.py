import base64
import logging
import os
import asyncio

from optimum.onnxruntime import ORTModelForImageClassification, ORTQuantizer
from optimum.onnxruntime.configuration import (
    QuantizationConfig,
    QuantFormat,
    QuantizationMode,
    QuantType,
)
from transformers import AutoImageProcessor
from optimum.pipelines import pipeline
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI, Response, status, Body
from fastapi.responses import JSONResponse, PlainTextResponse

model_name = os.environ.get("MODEL_NAME")
batch_size = int(os.environ.get("BATCH_SIZE"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Setup logging to app insights
try:
    configure_azure_monitor(logger_name="app")
except Exception as e:
    log.warning(f"Failed to configure azure monitoring: {e}")

# Preload the model
try:
    log.info(f"Loading inference model, model: {model_name}")
    model = ORTModelForImageClassification.from_pretrained(
        model_id=model_name, export=True
    )

    optimizer = ORTQuantizer.from_pretrained(model)
    config = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,
        weights_dtype=QuantType.QInt8,
        nodes_to_exclude=["/swinv2/embeddings/patch_embeddings/projection/Conv"],
    )

    optimizer.quantize(save_dir=".optimized/1", quantization_config=config)
    optimizer = ORTQuantizer.from_pretrained(".optimized/1")
    config = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,
        weights_dtype=QuantType.QUInt8,
        nodes_to_quantize=["/swinv2/embeddings/patch_embeddings/projection/Conv"],
    )

    optimizer.quantize(save_dir=".optimized/2", quantization_config=config)
    model = ORTModelForImageClassification.from_pretrained(".optimized/2")

    inference = pipeline(
        task="image-classification",
        model=model,
        image_processor=AutoImageProcessor.from_pretrained(model_name),
        accelerator="ort",
    )

    model = None
    optimizer = None
    config = None

    log.info(f"Model loaded successfully")
except Exception as e:
    log.error(f"Loading model failed: {e}")
    raise e

# Initialize the server
app = FastAPI()

# Setup for batch processing
loop = asyncio.get_event_loop()
queue = asyncio.Queue()


@app.options("/invoke")
async def invoke_options():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/invoke")
async def invoke_post(body=Body()):
    log.info(f"Invoking transformer pipeline, model: {model_name}")
    input = base64.b64encode(body).decode("ascii")

    try:
        future = loop.create_future()
        queueInvoke(input, future)
        return JSONResponse(await future, status_code=200)
    except Exception as e:
        log.warning(f"Inference failed: {e}")
        return PlainTextResponse(f"[INFERENCE FAILED]: {e}", status_code=500)


def queueInvoke(input, future):
    queue.put_nowait([input, future])
    log.info(f"Queued invocation, size: {queue.qsize()} ")


async def processQueue():
    while True:
        head = await queue.get()

        batch = [queue.get_nowait() for _ in range(min(batch_size - 1, queue.qsize()))]
        batch.append(head)

        inputs = [item[0] for item in batch]
        log.info(f"Invoking batch inference, size: {len(inputs)}")
        outputs = inference(inputs, batch_size=len(inputs))

        for idx, item in enumerate(batch):
            item[1].set_result(outputs[idx])


FastAPIInstrumentor.instrument_app(app)
loop.create_task(processQueue())
