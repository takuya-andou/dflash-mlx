from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class PromptContext:
    input_ids: mx.array
    prompt_text: str | None = None
    pixel_values: mx.array | None = None
    image_grid_thw: mx.array | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_input_tokens(self) -> int:
        return int(self.input_ids.shape[0])


def prompt_context_from_tokens(
    input_ids: mx.array,
    *,
    prompt_text: str | None = None,
    pixel_values: mx.array | None = None,
    image_grid_thw: mx.array | None = None,
    extras: dict[str, Any] | None = None,
) -> PromptContext:
    return PromptContext(
        input_ids=input_ids.astype(mx.uint32),
        prompt_text=prompt_text,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        extras={} if extras is None else dict(extras),
    )
