from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .prompting import PromptContext


def _to_mlx(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if value is None or isinstance(value, mx.array):
            result[key] = value
        elif isinstance(value, np.ndarray):
            result[key] = mx.array(value)
        elif isinstance(value, list):
            try:
                result[key] = mx.array(np.array(value))
            except (TypeError, ValueError):
                result[key] = value
        else:
            result[key] = value
    return result


def _load_chat_template(tokenizer: Any, model_path: Path) -> None:
    chat_template_json = model_path / "chat_template.json"
    chat_template_jinja = model_path / "chat_template.jinja"

    if chat_template_json.exists():
        template_data = json.loads(chat_template_json.read_text())
        tokenizer.chat_template = template_data["chat_template"]
    elif chat_template_jinja.exists():
        tokenizer.chat_template = chat_template_jinja.read_text()


def _load_image_source(image: str | Path | Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Image input requires Pillow to be installed in the runtime environment."
        ) from exc

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, Path):
        return Image.open(image).convert("RGB")

    if isinstance(image, str):
        if image.startswith(("http://", "https://")):
            req = urllib.request.Request(image, headers={"User-Agent": "dflash-mlx"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return Image.open(BytesIO(response.read())).convert("RGB")
        return Image.open(image).convert("RGB")

    return image


@dataclass
class Qwen35ProcessorBundle:
    tokenizer: Any
    image_processor: Any
    image_token: str


def load_qwen35_processor_bundle(model_path: Path) -> Qwen35ProcessorBundle:
    from transformers import AutoImageProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    _load_chat_template(tokenizer, model_path)

    image_processor_overrides: dict[str, Any] = {}
    proc_cfg_path = model_path / "processor_config.json"
    if proc_cfg_path.exists():
        proc_cfg = json.loads(proc_cfg_path.read_text())
        image_cfg = proc_cfg.get("image_processor", {})
        for key in ("patch_size", "size", "merge_size", "temporal_patch_size"):
            if key in image_cfg:
                image_processor_overrides[key] = image_cfg[key]

    try:
        image_processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            use_fast=False,
            **image_processor_overrides,
        )
    except ValueError:
        image_processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            **image_processor_overrides,
        )

    image_token = (
        tokenizer.image_token
        if getattr(tokenizer, "image_token", None)
        else "<|image_pad|>"
    )
    return Qwen35ProcessorBundle(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token=image_token,
    )


def build_qwen35_multimodal_prompt_context(
    model_path: Path,
    prompt_text: str,
    images: list[str | Path | Any],
) -> PromptContext:
    if not images:
        raise ValueError("build_qwen35_multimodal_prompt_context requires at least one image.")

    bundle = load_qwen35_processor_bundle(model_path)
    pil_images = [_load_image_source(image) for image in images]
    image_inputs = _to_mlx(bundle.image_processor(images=pil_images))
    image_grid_thw = image_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        raise ValueError("Qwen3.5 image processor did not return image_grid_thw.")

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    prompt = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    merge_size = int(getattr(bundle.image_processor, "merge_size", 2) ** 2)
    expanded_prompt = prompt
    for grid in image_grid_thw.tolist():
        num_image_tokens = int(np.prod(grid) // merge_size)
        expanded_prompt = expanded_prompt.replace(
            bundle.image_token,
            "<|placeholder|>" * num_image_tokens,
            1,
        )
    expanded_prompt = expanded_prompt.replace("<|placeholder|>", bundle.image_token)

    text_inputs = bundle.tokenizer([expanded_prompt], add_special_tokens=False)
    text_inputs = _to_mlx(text_inputs)

    return PromptContext(
        input_ids=text_inputs["input_ids"][0].astype(mx.uint32),
        prompt_text=prompt_text,
        pixel_values=image_inputs.get("pixel_values"),
        image_grid_thw=image_grid_thw,
        extras={
            "messages": messages,
            "images": list(images),
            "attention_mask": text_inputs.get("attention_mask"),
        },
    )
