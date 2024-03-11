import os
import time
import traceback
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config import load_config_from_yaml
from prototype.logger import setup_logging
from prototype.observability.elastic_search_logging import elastic_search_setup
from prototype.pipelines.preprocessing import get_inputs
from prototype.utils import load_classifier

config = load_config_from_yaml("../config.yaml", cli_args=None)
logger = setup_logging(config)

if config.elasticsearch.enabled:
    client = elastic_search_setup(config=config)

model = load_classifier(
    config=config,
    directory=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

# App ini
app = FastAPI()

# Prometheus metrics
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds", "Inference latency in seconds", ["endpoint"]
)
ERROR_COUNT = Counter(
    "http_errors_total", "Total number of HTTP errors", ["method", "endpoint"]
)


class Product(BaseModel):
    name: str
    description: str
    product_id: int


class InputData(BaseModel):
    data: list[Product]


class OutputItem(BaseModel):
    pattern: str
    product_id: int


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()
        request_body = await request.body()
        response = await call_next(request)
        end_time = datetime.utcnow()

        try:
            if hasattr(response, "body"):
                response_body = await response.body()
                response_body_str = response_body.decode()
            else:
                response_body_str = ""
        except Exception as e:
            response_body_str = ""
            logger.error(f"Error decoding response body: {e}")

        log_entry = {
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "request_body": request_body,
            "status_code": response.status_code,
            "response_body": response_body_str,
            "duration_ms": (end_time - start_time).total_seconds() * 1000,
        }

        logger.info(f"incoming_request: {log_entry}")
        if config.elasticsearch.enabled:
            client.index(index="fastapi-logs", body=log_entry)
        return response


app.add_middleware(LoggingMiddleware)


@app.get("/docs")
async def get_docs():
    return RedirectResponse(url="/docs")


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/infer")
def infer(products: InputData):
    # TODO would be better to rewrite the preprocessing to operate on series
    start_time = time.time()
    column_names = Product.__fields__.keys()
    data = [
        {name: getattr(product, name) for name in column_names}
        for product in products.data
    ]
    input_df = pd.DataFrame(data)

    x = get_inputs(input_df)
    predictions = model.predict(x)

    output_items = [
        OutputItem(pattern=pattern, product_id=product_id)
        for pattern, product_id in zip(predictions, input_df["product_id"])
    ]

    log_entry = {"infer_response": output_items}
    logger.info(log_entry)
    if config.elasticsearch.enabled:
        client.index(index="fastapi-inference-logs", body=log_entry)

    INFERENCE_LATENCY.labels(endpoint="/infer").observe(time.time() - start_time)
    return output_items


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    start_time = datetime.utcnow()
    end_time = datetime.utcnow()

    log_entry = {
        "timestamp": start_time.isoformat(),
        "method": request.method,
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "request_body": await request.body(),
        "status_code": 500,  # Since it's an exception, set the status code to 500
        "response_body": {"message": "Internal Server Error"},
        "duration_ms": (end_time - start_time).total_seconds() * 1000,
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }

    logger.error(f"exception_handler: {log_entry}")
    if config.elasticsearch.enabled:
        client.index(index="fastapi-logs", body=log_entry)

    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})


@app.middleware("http")
async def record_request_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    request_latency = time.time() - start_time
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(request_latency)
    return response


@app.middleware("http")
async def record_request_count(request: Request, call_next):
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(
            request.method, request.url.path, str(response.status_code)
        ).inc()
        return response
    except Exception as _:
        REQUEST_COUNT.labels(request.method, request.url.path, "500").inc()
        raise


@app.middleware("http")
async def record_error_count(request: Request, call_next):
    try:
        response = await call_next(request)
        if 400 <= response.status_code < 600:
            ERROR_COUNT.labels(request.method, request.url.path).inc()
        return response
    except Exception as _:
        ERROR_COUNT.labels(request.method, request.url.path).inc()
        raise


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
