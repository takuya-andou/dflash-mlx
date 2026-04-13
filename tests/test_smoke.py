"""Smoke test: the package imports and public surface is wired correctly.

Intentionally does not load any model weights. A contributor can run this
without an Apple Silicon GPU and without a ~12 GB model download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_public_api_importable():
    import dflash_mlx

    expected = {
        "DFlashGenerator",
        "DFlashResult",
        "DFlashDraftModel",
        "LoadedTargetModel",
        "adapter_for_model_type",
        "dflash_generate",
        "load_draft_model",
        "load_target_model",
        "longest_prefix_match",
        "sample_tokens",
    }
    assert expected.issubset(set(dflash_mlx.__all__))
    for name in expected:
        assert hasattr(dflash_mlx, name), f"missing export: {name}"


def test_generator_is_callable_class():
    from dflash_mlx import DFlashGenerator

    assert callable(DFlashGenerator)


def test_prompt_context_exported():
    from dflash_mlx import PromptContext

    assert PromptContext.__name__ == "PromptContext"


def test_cli_entrypoints_importable():
    from dflash_mlx import benchmark_cli, chat_cli, cli, inspect_cli, model_prep

    for module in (cli, benchmark_cli, chat_cli, inspect_cli, model_prep):
        assert callable(module.main)


def test_cli_verify_mode_choices_exclude_unsafe_modes():
    from dflash_mlx import chat_cli, cli

    saved_argv = sys.argv
    try:
        for module in (cli, chat_cli):
            sys.argv = [module.__name__]
            args = module.parse_args()
            assert args.verify_mode == "parallel-replay"
    finally:
        sys.argv = saved_argv


def test_cli_image_flags_parse():
    from dflash_mlx import chat_cli, cli

    saved_argv = sys.argv
    try:
        sys.argv = ["dflash_mlx.cli", "--image", "a.png", "--image", "https://example.com/b.png"]
        args = cli.parse_args()
        assert args.image == ["a.png", "https://example.com/b.png"]

        sys.argv = ["dflash_mlx.chat_cli", "--image", "a.png"]
        args = chat_cli.parse_args()
        assert args.image == ["a.png"]
    finally:
        sys.argv = saved_argv


def test_runtime_accepts_prompt_context_for_prefill_only_generation():
    import mlx.core as mx

    from dflash_mlx.prompting import prompt_context_from_tokens
    from dflash_mlx.runtime import dflash_generate

    class FakeTarget:
        def __init__(self):
            self.seen_prompt_context = None

        def make_cache(self):
            return ["target-cache"]

        def prefill_with_hidden_states(self, prompt_context, cache, layer_ids):
            self.seen_prompt_context = prompt_context
            logits = mx.array([[[0.0, 1.0]]])
            target_hidden = mx.zeros((1, prompt_context.num_input_tokens, 1))
            return logits, target_hidden

        def cache_summary(self, cache):
            return "ok"

    class FakeDraft:
        block_size = 1
        mask_token_id = 0

        def make_cache(self):
            return []

    target = FakeTarget()
    draft = FakeDraft()
    prompt_context = prompt_context_from_tokens(mx.array([10, 20, 30], dtype=mx.uint32))

    output_tokens, metrics = dflash_generate(
        target=target,
        draft=draft,
        prompt_tokens=prompt_context,
        max_new_tokens=0,
        temperature=0.0,
        stop_token_ids={999},
        layer_ids=[],
        speculative_tokens=None,
        verify_mode="parallel-replay",
        verify_chunk_size=4,
    )

    assert target.seen_prompt_context is prompt_context
    assert output_tokens == [10, 20, 30]
    assert metrics["num_input_tokens"] == 3
    assert metrics["num_output_tokens"] == 0


def test_qwen35_adapter_uses_multimodal_builder_when_images_present(monkeypatch):
    import mlx.core as mx

    import dflash_mlx.adapters as adapters
    from dflash_mlx.prompting import PromptContext

    captured = {}

    def fake_builder(model_path, prompt_text, images):
        captured["model_path"] = model_path
        captured["prompt_text"] = prompt_text
        captured["images"] = images
        return PromptContext(input_ids=mx.array([1, 2, 3], dtype=mx.uint32))

    monkeypatch.setattr(adapters, "build_qwen35_multimodal_prompt_context", fake_builder)

    adapter = adapters.Qwen35TargetAdapter()
    context = adapter.build_prompt_context(
        tokenizer=object(),
        prompt_text="describe",
        images=["example.png"],
        model_path=Path("/tmp/qwen35"),
    )

    assert captured == {
        "model_path": Path("/tmp/qwen35"),
        "prompt_text": "describe",
        "images": ["example.png"],
    }
    assert context.input_ids.tolist() == [1, 2, 3]


def test_qwen35_multimodal_prefill_uses_custom_model_forward_dflash():
    import mlx.core as mx

    from dflash_mlx.adapters import Qwen35TargetAdapter
    from dflash_mlx.prompting import PromptContext

    captured = {}

    class FakeModel:
        def forward_dflash(self, **kwargs):
            captured.update(kwargs)
            logits = mx.array([[[0.0, 1.0]]])
            target_hidden = mx.zeros((1, 2, 1))
            return logits, target_hidden

    adapter = Qwen35TargetAdapter()
    prompt_context = PromptContext(
        input_ids=mx.array([1, 2], dtype=mx.uint32),
        pixel_values=mx.zeros((1, 3, 4, 4)),
        image_grid_thw=mx.array([[1, 2, 2]], dtype=mx.int32),
        extras={
            "attention_mask": mx.array([[1, 1]], dtype=mx.int32),
            "position_ids": mx.arange(6, dtype=mx.int32).reshape(3, 1, 2),
        },
    )

    logits, target_hidden = adapter.prefill_with_hidden_states(
        model=FakeModel(),
        prompt_context=prompt_context,
        cache=["cache"],
        layer_ids=[3, 7],
    )

    assert logits.shape == (1, 1, 2)
    assert target_hidden.shape == (1, 2, 1)
    assert captured["inputs"].tolist() == [[1, 2]]
    assert captured["cache"] == ["cache"]
    assert captured["layer_ids"] == [3, 7]
    assert captured["attention_mask"].tolist() == [[1, 1]]
    assert captured["pixel_values"].shape == (1, 3, 4, 4)
    assert captured["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert captured["position_ids"].shape == (3, 1, 2)
    assert captured["return_rollback_records"] is False


def test_qwen35_multimodal_verifier_uses_resolved_position_ids():
    import mlx.core as mx

    from dflash_mlx.adapters import Qwen35TargetAdapter

    captured = {}

    class FakeInnerModel:
        class FakeEmbedTokens:
            @staticmethod
            def as_linear(hidden_states):
                return hidden_states

        def __init__(self):
            self.embed_tokens = self.FakeEmbedTokens()

        def forward_dflash(self, **kwargs):
            captured.update(kwargs)
            norm_hidden_states = mx.zeros((1, 2, 4))
            target_hidden = mx.zeros((1, 2, 3))
            rollback_records = {}
            if kwargs["return_rollback_records"]:
                return norm_hidden_states, target_hidden, rollback_records
            return norm_hidden_states, target_hidden

    class FakeLanguageModel:
        def __init__(self):
            self.args = type("Args", (), {"tie_word_embeddings": True})()
            self.model = FakeInnerModel()

        def resolve_position_ids(self, inputs, cache):
            captured["resolved_inputs"] = inputs
            captured["resolved_cache"] = cache
            return mx.arange(6, dtype=mx.int32).reshape(3, 1, 2)

    class FakeModel:
        def __init__(self):
            self.language_model = FakeLanguageModel()

    adapter = Qwen35TargetAdapter()
    model = FakeModel()
    inputs = mx.array([[4, 5]], dtype=mx.uint32)
    cache = ["cache"]

    norm_hidden_states, target_hidden, rollback_records = adapter.forward_verifier_states(
        model=model,
        inputs=inputs,
        cache=cache,
        layer_ids=[1],
    )

    assert norm_hidden_states.shape == (1, 2, 4)
    assert target_hidden.shape == (1, 2, 3)
    assert rollback_records == {}
    assert captured["resolved_inputs"] is inputs
    assert captured["resolved_cache"] == cache
    assert captured["position_ids"].shape == (3, 1, 2)
    assert captured["return_rollback_records"] is True

    logits, accept_hidden = adapter.forward_accept_all_block(
        model=model,
        inputs=inputs,
        cache=cache,
        layer_ids=[1],
    )
    assert logits.shape == (1, 1, 4)
    assert accept_hidden.shape == (1, 2, 3)
    assert captured["return_rollback_records"] is False
