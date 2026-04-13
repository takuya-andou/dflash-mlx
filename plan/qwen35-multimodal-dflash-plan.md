# Qwen3.5-4B Image Support Plan for `dflash-mlx`

## Goal

Add image input support for `Qwen3.5-4B` without weakening the core value of DFlash:

- exact output parity with plain target decoding
- speculative decoding only on text generation
- single-pass verification and exact cache rollback

The design constraint is simple: treat images as part of the **prefill-only multimodal prefix**, then run the existing DFlash text decode loop on top of the prefilled cache.

## Non-Goals

- No video support in phase 1
- No generic multimodal support for every target family
- No inexact verifier shortcuts for image prompts

## Why the Current Code Cannot Handle Images

Current Qwen3.5 support in this repo is text-only:

- `dflash_mlx/adapters.py` builds prompts from a plain `prompt_text: str`
- `dflash_mlx/api.py` only accepts string prompts and token arrays
- `load_target_model()` loads `model, tokenizer` via `mlx_lm.load(...)`, not a multimodal processor
- `dflash_mlx/custom_qwen35_model.py` explicitly drops visual weights

That means the repo has no path for:

- chat-template image placeholders
- image preprocessing to `pixel_values`
- `image_grid_thw`
- vision encoder execution
- scattering vision embeddings into text placeholders before decode

## Core Design

The correct integration point is **before speculative decode begins**.

1. Build a multimodal prompt for Qwen3.5 using its image-aware chat template.
2. Run the processor to produce:
   - `input_ids`
   - `pixel_values`
   - `image_grid_thw`
3. Run one exact target prefill that:
   - computes image embeddings
   - inserts them into placeholder token positions
   - fills the target KV / linear-attention cache
   - returns the hidden states needed by the DFlash drafter
4. Start the normal DFlash loop from that cache state.

This preserves DFlash where it matters: the speculative path still applies only to generated text tokens.

## Architecture Changes

### 1. Introduce a Prompt Context Abstraction

Replace the implicit `prompt_text -> tokens` assumption with a richer prompt object.

Suggested shape:

```python
@dataclass
class PromptContext:
    input_ids: mx.array
    prompt_text: str | None = None
    pixel_values: mx.array | None = None
    image_grid_thw: mx.array | None = None
    num_input_tokens: int | None = None
```

This should become the unit passed from API and adapter layers into the target prefill path.

Files:

- `dflash_mlx/api.py`
- `dflash_mlx/adapters.py`
- `dflash_mlx/runtime.py`

### 2. Split Prefill From Decode

Today the runtime effectively assumes a text token prefix. Refactor it into:

- `prefill_target(prompt_context, ...) -> PrefillState`
- `dflash_decode_from_prefill(prefill_state, ...)`

Suggested `PrefillState`:

```python
@dataclass
class PrefillState:
    cache: list[Any]
    prompt_tokens: mx.array
    last_hidden: mx.array
    target_hidden: mx.array
    num_input_tokens: int
```

The decode path should remain unchanged as much as possible once prefill is done.

### 3. Add a Qwen3.5 Multimodal Target Path

Keep Qwen3 text-only support working, but add a new Qwen3.5-specific multimodal path:

- image-aware prompt building
- processor-backed image preparation
- multimodal prefill

This should live behind the existing adapter boundary so the rest of the runtime stays model-agnostic.

Files:

- `dflash_mlx/adapters.py`
- possibly new helper: `dflash_mlx/qwen35_multimodal.py`

### 4. Upgrade `custom_qwen35_model.py` From Text-Only to Multimodal

This is the main engineering task.

Required changes:

- stop dropping `vision_tower` / `model.visual` weights
- add a visual module to the custom model
- implement the equivalent of:
  - `get_image_features(...)`
  - placeholder masking
  - inserting image embeddings into `inputs_embeds`
- keep DFlash-specific methods:
  - hidden-state extraction
  - exact rollback for linear attention caches
  - verifier-specific forward paths

Important constraint:

The multimodal prefill must produce the same text hidden states and cache state as plain Qwen3.5 multimodal decoding would produce.

### 5. Loader / Processor Strategy

Do not bolt `mlx-vlm` on top as a separate runtime. That would split the code path and make exactness harder to reason about.

Better approach:

- reuse `mlx-vlm` conversion and processor conventions
- integrate the necessary processor + multimodal prefill logic into the Qwen3.5 target path used by `dflash-mlx`

This repo should still own the final prefill + decode flow.

## Phase Plan

### Phase 0: Research Spike

Goal: confirm the minimal path needed for exact multimodal prefill.

Tasks:

- inspect `mlx-community/Qwen3.5-4B-MLX-bf16` processor inputs and config
- inspect `mlx-vlm` Qwen3.5 generation path
- map which tensors are required before text decode starts
- confirm draft model can still condition on target hidden states after image prefill

Exit criteria:

- written tensor-flow note
- exact list of fields needed in `PromptContext`

### Phase 1: Text-Only Refactor With No Behavior Change

Goal: make the runtime capable of handling richer prompt inputs without changing behavior.

Tasks:

- add `PromptContext`
- split prefill from decode
- route existing text prompts through the new prefill abstraction
- keep all current tests passing

Exit criteria:

- existing text behavior unchanged
- smoke tests unchanged

### Phase 2: Qwen3.5 Image Prefill

Goal: implement image-aware prefill for Qwen3.5 only.

Tasks:

- add image-aware prompt building
- add processor invocation
- extend custom Qwen3.5 model with visual weights and image embedding injection
- return exact prefilled cache and hidden states

Exit criteria:

- one image + one text prompt reaches the normal DFlash decode loop
- no speculative logic depends on re-running the vision encoder

### Phase 3: Exactness Validation

Goal: prove that DFlash remains exact on multimodal prompts.

Tests:

- greedy decode parity against plain Qwen3.5 multimodal decode
- parity with different image sizes / aspect ratios
- parity when verifier rejections happen
- parity across short and longer generations

Exit criteria:

- deterministic match on curated fixtures
- no cache corruption after rollback

### Phase 4: Public API / CLI

Goal: expose image input safely.

Suggested additions:

- Python API:
  - `generate_messages(messages, images=...)`
  - or `generate(prompt_text, image=...)` for the simple case
- CLI:
  - `--image path.jpg`
  - later `--image-url ...` if needed

Keep image support explicitly limited to Qwen3.5 in phase 1.

## Tests

Add tests in layers.

### Unit Tests

- prompt construction inserts image placeholders correctly
- placeholder count matches produced image embeddings
- prefill returns valid cache shapes

### Integration Tests

- text-only regression tests still pass
- Qwen3.5 image prompt smoke test
- exact output parity test against plain target decode

### Performance Checks

- verify the vision encoder runs only during prefill
- compare generation TPS before and after the change on text-only inputs
- record multimodal prefill cost separately from generation TPS

## Risks

### Risk 1: Processor / MLX Runtime Mismatch

The current repo is centered on `mlx_lm`, while the converted multimodal model references `mlx-vlm`.

Mitigation:

- isolate all multimodal logic behind Qwen3.5 adapter boundaries
- treat `mlx-vlm` as a reference implementation, not the final runtime owner

### Risk 2: Draft Acceptance Degradation

Even if exactness holds, acceptance rate may drop for image-heavy prompts because the target hidden-state distribution shifts.

Mitigation:

- validate on a small image prompt suite early
- ship image support first, optimize acceptance second

### Risk 3: Cache Bugs After Multimodal Prefill

The hardest failure mode is subtle divergence after one or more rollback events.

Mitigation:

- add parity tests that force rejection paths
- compare against plain target decode token-by-token

## Recommended Implementation Order

1. Refactor text path into `PromptContext` + `PrefillState`
2. Add multimodal prompt building for Qwen3.5
3. Extend custom Qwen3.5 model to keep visual weights
4. Implement image prefill and cache initialization
5. Reuse existing DFlash decode from prefilled state
6. Add exactness tests
7. Expose Python API
8. Expose CLI

## Definition of Done

The feature is done when all of the following are true:

- Qwen3.5 image prompts run in `dflash-mlx`
- generated text exactly matches plain Qwen3.5 multimodal decoding under deterministic settings
- rollback remains exact after partial verifier rejection
- the vision encoder runs only during prefill
- text-only Qwen3 and Qwen3.5 behavior does not regress

