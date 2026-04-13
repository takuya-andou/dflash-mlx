#!/usr/bin/env python3
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dflash_mlx import DFlashGenerator


DEFAULT_TARGET_MODEL = "mlx-community/Qwen3.5-4B-MLX-bf16"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-4B-DFlash"


class GenerateRequest(BaseModel):
    prompt: str
    images: list[str] = Field(default_factory=list)
    max_new_tokens: int = 256
    temperature: float = 0.0
    speculative_tokens: int | None = None
    verify_mode: str = "parallel-replay"
    verify_chunk_size: int = 4
    skip_special_tokens: bool = True
    profile: bool = False


@lru_cache(maxsize=1)
def get_runner() -> DFlashGenerator:
    return DFlashGenerator(
        target_model=os.environ.get("TARGET_MODEL", DEFAULT_TARGET_MODEL),
        draft_model=os.environ.get("DRAFT_MODEL", DEFAULT_DRAFT_MODEL),
        seed=int(os.environ.get("SEED", "0")),
    )


app = FastAPI(title="dflash-mlx Qwen3.5 Image API")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    runner = get_runner()
    return {
        "ok": True,
        "target_model": str(runner.target_model_path),
        "draft_model": str(runner.draft_path),
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    runner = get_runner()
    try:
        result = runner.generate(
            prompt_text=request.prompt,
            images=request.images or None,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            speculative_tokens=request.speculative_tokens,
            verify_mode=request.verify_mode,
            verify_chunk_size=request.verify_chunk_size,
            skip_special_tokens=request.skip_special_tokens,
            profile=request.profile,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "text": result.text,
        "output_tokens": result.output_tokens,
        "generated_tokens": result.generated_tokens,
        "metrics": result.metrics,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
