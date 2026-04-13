#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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


def run_generation(
    *,
    prompt: str,
    images: list[str] | None,
    max_new_tokens: int,
    temperature: float,
    speculative_tokens: int | None,
    verify_mode: str,
    verify_chunk_size: int,
    skip_special_tokens: bool,
    profile: bool,
) -> dict[str, Any]:
    runner = get_runner()
    try:
        result = runner.generate(
            prompt_text=prompt,
            images=images or None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "text": result.text,
        "output_tokens": result.output_tokens,
        "generated_tokens": result.generated_tokens,
        "metrics": result.metrics,
    }


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
    return run_generation(
        prompt=request.prompt,
        images=request.images,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        speculative_tokens=request.speculative_tokens,
        verify_mode=request.verify_mode,
        verify_chunk_size=request.verify_chunk_size,
        skip_special_tokens=request.skip_special_tokens,
        profile=request.profile,
    )


@app.post("/generate-upload")
async def generate_upload(
    prompt: str = Form(...),
    images: list[UploadFile] = File(...),
    max_new_tokens: int = Form(256),
    temperature: float = Form(0.0),
    speculative_tokens: int | None = Form(None),
    verify_mode: str = Form("parallel-replay"),
    verify_chunk_size: int = Form(4),
    skip_special_tokens: bool = Form(True),
    profile: bool = Form(False),
) -> dict[str, Any]:
    if not images:
        raise HTTPException(status_code=400, detail="At least one image upload is required.")

    with tempfile.TemporaryDirectory(prefix="dflash-qwen35-upload-") as tmpdir:
        image_paths: list[str] = []
        tmpdir_path = Path(tmpdir)

        for index, upload in enumerate(images):
            suffix = Path(upload.filename or f"image-{index}").suffix or ".bin"
            target_path = tmpdir_path / f"image-{index}{suffix}"
            data = await upload.read()
            target_path.write_bytes(data)
            image_paths.append(str(target_path))

        return run_generation(
            prompt=prompt,
            images=image_paths,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=speculative_tokens,
            verify_mode=verify_mode,
            verify_chunk_size=verify_chunk_size,
            skip_special_tokens=skip_special_tokens,
            profile=profile,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
