import asyncio
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from flow_grpo.harmbench_utils import predict


class PredictRequest(BaseModel):
    generations: str
    behavior: str
    device: str = "cuda"
    return_logprob: bool = False


class PredictResponse(BaseModel):
    is_harmful: bool | None = None
    success_logprob: float | None = None
    completion: str | None = None
    message: str


app = FastAPI(
    title="HarmBench API Server",
    description="ASO reward helper for HarmBench classification.",
    version="1.0.0",
)

sem = asyncio.Semaphore(int(os.environ.get("HARM_BENCH_CONCURRENCY", "1")))


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    async with sem:
        try:
            result = predict(
                generations=request.generations,
                behavior=request.behavior,
                device=request.device,
                return_logprob=request.return_logprob,
            )
            if request.return_logprob:
                return PredictResponse(
                    is_harmful=result["is_harmful"],
                    success_logprob=result["success_logprob"],
                    completion=str(result["completion"]),
                    message="prediction succeeded",
                )
            return PredictResponse(
                is_harmful=bool(result),
                message="prediction succeeded",
            )
        except Exception as exc:
            return PredictResponse(message=f"prediction failed: {exc}")


@app.get("/")
async def root():
    return {
        "message": "HarmBench API server is running",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - classify a generation against a behavior",
            "/docs": "OpenAPI documentation",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.environ.get("HARM_BENCH_HOST", "0.0.0.0"),
        port=int(os.environ.get("HARM_BENCH_PORT", "5000")),
        reload=False,
        log_level=os.environ.get("HARM_BENCH_LOG_LEVEL", "info"),
    )
