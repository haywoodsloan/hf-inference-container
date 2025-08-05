import base64
import logging
import os
import asyncio

from datetime import datetime
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
from fastapi import FastAPI, Response, status, Body, Request
from fastapi.responses import JSONResponse, PlainTextResponse

model_name = os.environ.get("MODEL_NAME")
batch_size = int(os.environ.get("BATCH_SIZE"))
max_queue = int(os.environ.get("MAX_QUEUE"))

timeout = 30
heartbeat = 0.5

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
event = asyncio.Event()
lock = asyncio.Lock()
queue = []


@app.options("/invoke")
async def invoke_options():
    return Response(status_code=status.HTTP_200_OK)


@app.post("/invoke")
async def invoke_post(request:Request, body=Body()):
    if len(queue) >= max_queue:
        return PlainTextResponse(f"[INFERENCE FAILED]: Overloaded", status_code=503)

    log.info(f"Invoking transformer pipeline, model: {model_name}")
    input = base64.b64encode(body).decode("ascii")

    try:
        future = loop.create_future()
        await queueInvoke(input, future)

        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            if await request.is_disconnected():
                await dequeueInvoke(future)
                return

            try:
                shielded = asyncio.shield(future)
                result = await asyncio.wait_for(shielded, heartbeat)
                return JSONResponse(result, status_code=200)
            except asyncio.TimeoutError:
                pass

        await dequeueInvoke(future)
        raise Exception("Timeout")
    except Exception as e:
        log.warning(f"Inference failed: {e}")
        return PlainTextResponse(f"[INFERENCE FAILED]: {e}", status_code=500)


def find(arr, pred):
    for i in range(len(arr)):
        if pred(arr[i]):
            return i
    return -1


async def queueInvoke(input, future):
    async with lock:
        queue.append([input, future])
        log.info(f"Queued invocation, size: {len(queue)}")
    event.set()


async def dequeueInvoke(future):
    async with lock:
        idx = find(queue, lambda item: item[1] == future)
        if idx >= 0:
            del queue[idx]
            log.info(f"Dequeued invocation, size: {len(queue)}")


async def processQueue():
    while True:
        if not queue:
            event.clear()
            await event.wait()

        async with lock:
            batch = [queue.pop() for _ in range(min(batch_size, len(queue)))]

        inputs = [item[0] for item in batch]
        log.info(f"Invoking batch inference, size: {len(inputs)}")
        outputs = inference(inputs, batch_size=len(inputs))

        for idx, item in enumerate(batch):
            if not item[1].done():
                item[1].set_result(outputs[idx])


FastAPIInstrumentor.instrument_app(app)
loop.create_task(processQueue())
