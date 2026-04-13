# Local fork of MLX Qwen3.5 for DFlash verifier experiments.

from itertools import accumulate
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx.utils import tree_map

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import (
    compute_g,
    gated_delta_kernel,
    gated_delta_ops,
    gated_delta_update,
)
from mlx_lm.models.qwen3_next import Qwen3NextMLP as MLP
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated as RMSNormGated
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock


def make_gated_delta_state_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto dv_idx = thread_position_in_grid.y;
        constexpr int n_per_t = Dk / 32;

        auto k_ = k + (b_idx * T * Hv + hv_idx) * Dk;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          auto g_t = g_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
            state[i] = state[i] * g_t;
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
          }

          k_ += Hv * Dk;
          v_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_state_update",
        input_names=["k", "v", "g", "beta", "state_in", "T"],
        output_names=["state_out"],
        source=source,
    )


GATED_DELTA_STATE_KERNEL = make_gated_delta_state_kernel()
FULL_ATTENTION_VERIFY_COMPILED_FNS: dict[int, Any] = {}
LINEAR_VERIFY_COMPILED_FNS: dict[int, Any] = {}
VERIFY_WITH_ROLLBACK_COMPILED_FNS: dict[tuple[int, tuple[int, ...], int], Any] = {}
ENABLE_EXPLICIT_CACHE_COMPILED_VERIFY = False
ENABLE_LINEAR_LAYER_COMPILED_VERIFY = True


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
) -> mx.array:
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)
    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened
    return mx.reshape(final_embedding_flattened, final_embedding_shape)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


class Qwen3_5RotaryEmbedding:
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        mrope_section: list[int] | None = None,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mrope_section = mrope_section or [11, 11, 10]
        self.inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim)
        )

    def apply_interleaved_mrope(
        self,
        freqs: mx.array,
        mrope_section: list[int],
    ) -> mx.array:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 2:
            position_ids = mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )

        inv_freq_expanded = mx.broadcast_to(
            self.inv_freq[None, None, :, None].astype(mx.float32),
            (3, position_ids.shape[1], self.inv_freq.shape[0], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)
        freqs = inv_freq_expanded @ position_ids_expanded
        freqs = mx.swapaxes(freqs, 2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


def apply_multimodal_rotary_pos_emb(
    queries: mx.array,
    keys: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot = queries[..., :rotary_dim]
    q_pass = queries[..., rotary_dim:]
    k_rot = keys[..., :rotary_dim]
    k_pass = keys[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = mx.concatenate([q_embed, q_pass], axis=-1)
    k_embed = mx.concatenate([k_embed, k_pass], axis=-1)
    return q_embed, k_embed


def build_position_ids(
    batch_size: int,
    seq_len: int,
    *,
    offset: int = 0,
    position_ids: mx.array | None = None,
) -> mx.array:
    if position_ids is not None:
        if position_ids.ndim == 2:
            return mx.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1]),
            )
        return position_ids

    base = mx.arange(offset, offset + seq_len)
    base = mx.broadcast_to(base[None, :], (batch_size, seq_len))
    return mx.broadcast_to(base[None, ...], (3, batch_size, seq_len))


class Attention(nn.Module):
    def __init__(self, args: "TextModelArgs"):
        super().__init__()
        self.num_key_value_heads = args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            int(self.head_dim * args.partial_rotary_factor),
            max_position_embeddings=args.max_position_embeddings,
            base=args.rope_theta,
            mrope_section=args.rope_parameters["mrope_section"],
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(batch_size, seq_len, self.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(batch_size, seq_len, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            -1,
        ).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        position_ids = build_position_ids(
            batch_size,
            seq_len,
            offset=offset,
            position_ids=position_ids,
        )
        cos, sin = self.rotary_emb(values, position_ids)
        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output * mx.sigmoid(gate))


def advance_gated_delta_states(
    initial_states: mx.array,
    keys: mx.array,
    values: mx.array,
    g: mx.array,
    beta: mx.array,
) -> mx.array:
    if (
        GATED_DELTA_STATE_KERNEL is not None
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    ):
        batch_size, _, _, head_dim = keys.shape
        num_v_heads = values.shape[2]
        value_dim = values.shape[-1]
        output = GATED_DELTA_STATE_KERNEL(
            inputs=[keys, values, g, beta, initial_states, keys.shape[1]],
            template=[
                ("InT", initial_states.dtype),
                ("Dk", head_dim),
                ("Dv", value_dim),
                ("Hv", num_v_heads),
            ],
            grid=(32, value_dim, batch_size * num_v_heads),
            threadgroup=(32, 4, 1),
            output_shapes=[initial_states.shape],
            output_dtypes=[initial_states.dtype],
        )
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    state = initial_states.astype(mx.float32)
    keys_f = keys.astype(mx.float32)
    values_f = values.astype(mx.float32)
    g_f = g.astype(mx.float32)
    beta_f = beta.astype(mx.float32)
    for token_idx in range(keys.shape[1]):
        state = state * g_f[:, token_idx, :, None, None]
        kv_mem = mx.sum(state * keys_f[:, token_idx, :, None, :], axis=-1)
        delta = (values_f[:, token_idx] - kv_mem) * beta_f[:, token_idx, :, None]
        state = state + delta[..., None] * keys_f[:, token_idx, :, None, :]
    return state.astype(initial_states.dtype)


def get_compiled_full_attention_verify_fn(layer):
    key = id(layer)
    compiled = FULL_ATTENTION_VERIFY_COMPILED_FNS.get(key)
    if compiled is not None:
        return compiled

    attn = layer.self_attn

    @mx.compile
    def compiled_full_attention_verify(
        hidden_states: mx.array,
        old_keys: mx.array,
        old_values: mx.array,
        offset: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        residual = hidden_states
        inputs = layer.input_layernorm(hidden_states)
        B, L, _ = inputs.shape

        q_proj_output = attn.q_proj(inputs)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, attn.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(B, L, -1)

        new_keys = attn.k_proj(inputs)
        new_values = attn.v_proj(inputs)

        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        new_keys = attn.k_norm(
            new_keys.reshape(B, L, attn.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        new_values = new_values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        position_ids = build_position_ids(B, L, offset=offset)
        cos, sin = attn.rotary_emb(new_values, position_ids)
        queries, new_keys = apply_multimodal_rotary_pos_emb(
            queries,
            new_keys,
            cos,
            sin,
        )

        keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
        values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=attn.scale,
            mask="causal",
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = attn.o_proj(output * mx.sigmoid(gate))

        hidden_states = residual + output
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)
        return hidden_states, new_keys, new_values

    FULL_ATTENTION_VERIFY_COMPILED_FNS[key] = compiled_full_attention_verify
    return compiled_full_attention_verify


def forward_full_attention_layer_dflash(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: KVCache | None,
    position_ids: mx.array | None = None,
) -> mx.array:
    if (
        cache is not None
        and cache.keys is not None
        and hidden_states.shape[1] > 1
        and mask == "causal"
        and position_ids is None
    ):
        compiled = get_compiled_full_attention_verify_fn(layer)
        hidden_states, new_keys, new_values = compiled(
            hidden_states,
            cache.keys,
            cache.values,
            cache.offset,
        )
        cache.update_and_fetch(new_keys, new_values)
        return hidden_states
    return layer(hidden_states, mask=mask, cache=cache, position_ids=position_ids)


def get_compiled_linear_verify_fn(layer):
    key = id(layer)
    compiled = LINEAR_VERIFY_COMPILED_FNS.get(key)
    if compiled is not None:
        return compiled

    # Verification repeatedly evaluates fixed-size draft blocks with warm SSM
    # caches, so compiling the linear-attention layer body avoids rebuilding the
    # same MLX graph while keeping the rollback tensors exact.
    @mx.compile
    def compiled_linear_verify(
        hidden_states: mx.array,
        initial_conv_state: mx.array,
        initial_state: mx.array,
    ) -> tuple[
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
        mx.array,
    ]:
        return forward_linear_layer_explicit_with_record(
            layer,
            hidden_states,
            None,
            initial_conv_state,
            initial_state,
        )

    LINEAR_VERIFY_COMPILED_FNS[key] = compiled_linear_verify
    return compiled_linear_verify


def forward_linear_layer_with_rollback_record(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    cache: ArraysCache | None,
) -> tuple[mx.array, dict[str, mx.array]]:
    if (
        ENABLE_LINEAR_LAYER_COMPILED_VERIFY
        and cache is not None
        and cache[0] is not None
        and cache[1] is not None
        and hidden_states.shape[1] > 1
        and mask is None
    ):
        initial_conv_state = cache[0]
        initial_state = cache[1]
        compiled = get_compiled_linear_verify_fn(layer)
        (
            hidden_states,
            new_conv_state,
            new_state,
            qkv,
            keys,
            values,
            g,
            beta,
        ) = compiled(hidden_states, initial_conv_state, initial_state)
        cache[0] = new_conv_state
        cache[1] = new_state
        linear = layer.linear_attn
        rollback_record = {
            "initial_conv_state": initial_conv_state,
            "initial_state": initial_state,
            "qkv": qkv,
            "k": keys,
            "v": values,
            "g": g,
            "beta": beta,
            "repeat_factor": linear.num_v_heads // linear.num_k_heads,
        }
        return hidden_states, rollback_record

    linear = layer.linear_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    batch_size, seq_len, _ = inputs.shape

    qkv = linear.in_proj_qkv(inputs)
    z = linear.in_proj_z(inputs).reshape(
        batch_size,
        seq_len,
        linear.num_v_heads,
        linear.head_v_dim,
    )
    b = linear.in_proj_b(inputs)
    a = linear.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        initial_conv_state = cache[0]
    else:
        initial_conv_state = mx.zeros(
            (batch_size, linear.conv_kernel_size - 1, linear.conv_dim),
            dtype=inputs.dtype,
        )

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
    if cache is not None:
        cache[0] = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))

    queries, keys, values = [
        tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        for tensor, num_heads, head_dim in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

    state = cache[1] if cache is not None else None
    if state is not None:
        initial_state = state
    else:
        initial_state = mx.zeros(
            (batch_size, linear.num_v_heads, linear.head_v_dim, linear.head_k_dim),
            dtype=inputs.dtype,
        )
    inv_scale = keys.shape[-1] ** -0.5
    queries = (inv_scale**2) * mx.fast.rms_norm(queries, None, 1e-6)
    keys = inv_scale * mx.fast.rms_norm(keys, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(linear.A_log, a, linear.dt_bias)

    use_kernel = (
        not linear.training
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )
    if use_kernel:
        out, state = gated_delta_kernel(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=state,
            mask=mask,
        )
    else:
        out, state = gated_delta_ops(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=state,
            mask=mask,
        )

    if cache is not None:
        cache[1] = state

    out = linear.norm(out, z)
    out = linear.out_proj(out.reshape(batch_size, seq_len, -1))
    hidden_states = residual + out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)

    rollback_record = {
        "initial_conv_state": initial_conv_state,
        "initial_state": initial_state,
        "qkv": qkv,
        "k": keys,
        "v": values,
        "g": g,
        "beta": beta,
        "repeat_factor": linear.num_v_heads // linear.num_k_heads,
    }
    return hidden_states, rollback_record


def forward_linear_layer_explicit_with_record(
    layer,
    hidden_states: mx.array,
    mask: mx.array | None,
    initial_conv_state: mx.array,
    initial_state: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    linear = layer.linear_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    batch_size, seq_len, _ = inputs.shape

    qkv = linear.in_proj_qkv(inputs)
    z = linear.in_proj_z(inputs).reshape(
        batch_size,
        seq_len,
        linear.num_v_heads,
        linear.head_v_dim,
    )
    b = linear.in_proj_b(inputs)
    a = linear.in_proj_a(inputs)

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
    new_conv_state = conv_input[:, -(linear.conv_kernel_size - 1) :]
    conv_out = nn.silu(linear.conv1d(conv_input))

    queries, keys, values = [
        tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        for tensor, num_heads, head_dim in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

    inv_scale = keys.shape[-1] ** -0.5
    queries = (inv_scale**2) * mx.fast.rms_norm(queries, None, 1e-6)
    keys = inv_scale * mx.fast.rms_norm(keys, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(linear.A_log, a, linear.dt_bias)

    use_kernel = (
        not linear.training
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )
    if use_kernel:
        out, new_state = gated_delta_kernel(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=initial_state,
            mask=mask,
        )
    else:
        out, new_state = gated_delta_ops(
            q=queries,
            k=keys,
            v=values,
            g=g,
            beta=beta,
            state=initial_state,
            mask=mask,
        )

    out = linear.norm(out, z)
    out = linear.out_proj(out.reshape(batch_size, seq_len, -1))
    hidden_states = residual + out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states, new_conv_state, new_state, qkv, keys, values, g, beta


def forward_full_attention_layer_explicit(
    layer,
    hidden_states: mx.array,
    old_keys: mx.array,
    old_values: mx.array,
    offset: int,
    position_ids: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    attn = layer.self_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    batch_size, seq_len, _ = inputs.shape

    q_proj_output = attn.q_proj(inputs)
    queries, gate = mx.split(
        q_proj_output.reshape(batch_size, seq_len, attn.num_attention_heads, -1),
        2,
        axis=-1,
    )
    gate = gate.reshape(batch_size, seq_len, -1)

    new_keys = attn.k_proj(inputs)
    new_values = attn.v_proj(inputs)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    new_keys = attn.k_norm(
        new_keys.reshape(batch_size, seq_len, attn.num_key_value_heads, -1)
    ).transpose(0, 2, 1, 3)
    new_values = new_values.reshape(
        batch_size,
        seq_len,
        attn.num_key_value_heads,
        -1,
    ).transpose(0, 2, 1, 3)

    position_ids = build_position_ids(
        batch_size,
        seq_len,
        offset=offset,
        position_ids=position_ids,
    )
    cos, sin = attn.rotary_emb(new_values, position_ids)
    queries, new_keys = apply_multimodal_rotary_pos_emb(
        queries,
        new_keys,
        cos,
        sin,
    )

    keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
    values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn.scale,
        mask="causal",
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = attn.o_proj(output * mx.sigmoid(gate))

    hidden_states = residual + output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)
    return hidden_states, new_keys, new_values


def get_compiled_verify_with_rollback_fn(
    text_model,
    layer_ids: tuple[int, ...],
    seq_len: int,
):
    key = (id(text_model), layer_ids, seq_len)
    compiled = VERIFY_WITH_ROLLBACK_COMPILED_FNS.get(key)
    if compiled is not None:
        return compiled

    target_layer_ids = set(layer_ids)

    @mx.compile
    def compiled_verify(
        inputs: mx.array,
        fa_offset: int,
        full_keys: list[mx.array],
        full_values: list[mx.array],
        linear_conv_states: list[mx.array],
        linear_states: list[mx.array],
    ):
        hidden_states = text_model.embed_tokens(inputs)
        selected_hidden_states: list[mx.array] = []
        new_full_keys: list[mx.array] = []
        new_full_values: list[mx.array] = []
        new_linear_conv_states: list[mx.array] = []
        new_linear_states: list[mx.array] = []
        record_qkv: list[mx.array] = []
        record_k: list[mx.array] = []
        record_v: list[mx.array] = []
        record_g: list[mx.array] = []
        record_beta: list[mx.array] = []

        full_idx = 0
        linear_idx = 0
        for layer_idx, layer in enumerate(text_model.layers):
            if layer.is_linear:
                (
                    hidden_states,
                    new_conv_state,
                    new_state,
                    qkv,
                    keys,
                    values,
                    g,
                    beta,
                ) = forward_linear_layer_explicit_with_record(
                    layer,
                    hidden_states,
                    None,
                    linear_conv_states[linear_idx],
                    linear_states[linear_idx],
                )
                new_linear_conv_states.append(new_conv_state)
                new_linear_states.append(new_state)
                record_qkv.append(qkv)
                record_k.append(keys)
                record_v.append(values)
                record_g.append(g)
                record_beta.append(beta)
                linear_idx += 1
            else:
                hidden_states, new_keys, new_values = forward_full_attention_layer_explicit(
                    layer,
                    hidden_states,
                    full_keys[full_idx],
                    full_values[full_idx],
                    fa_offset,
                )
                new_full_keys.append(new_keys)
                new_full_values.append(new_values)
                full_idx += 1

            if layer_idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        return (
            text_model.norm(hidden_states),
            mx.concatenate(selected_hidden_states, axis=-1),
            new_full_keys,
            new_full_values,
            new_linear_conv_states,
            new_linear_states,
            mx.concatenate(record_qkv, axis=0),
            mx.concatenate(record_k, axis=0),
            mx.concatenate(record_v, axis=0),
            mx.concatenate(record_g, axis=0),
            mx.concatenate(record_beta, axis=0),
        )

    VERIFY_WITH_ROLLBACK_COMPILED_FNS[key] = compiled_verify
    return compiled_verify


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = ""
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    linear_num_value_heads: int = 64
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 192
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    head_dim: Optional[int] = None
    full_attention_interval: int = 4

    # MoE fields (optional, for Qwen3_5MoeForConditionalGeneration)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    decoder_sparse_step: int = 1
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True

    # Rope parameters
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 100000,
            "partial_rotary_factor": 0.25,
        }
    )

    # Derived from rope_parameters (set in __post_init__)
    partial_rotary_factor: float = 0.25
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")

            self.partial_rotary_factor = self.rope_parameters.get(
                "partial_rotary_factor", 0.25
            )
            self.rope_theta = self.rope_parameters.get("rope_theta", 100000.0)
            self.rope_scaling = self.rope_parameters


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by num_k_heads ({self.num_k_heads})"
            )

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = mx.ones(self.num_v_heads)

        A = mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,))
        self.A_log = mx.log(A)

        self.norm = RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.sharding_group = None

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape

        if self.sharding_group is not None:
            inputs = sum_gradients(self.sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state

        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))

        if self.sharding_group is not None:
            out = mx.distributed.all_sum(out, group=self.sharding_group)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = (layer_idx + 1) % args.full_attention_interval != 0
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if args.num_experts > 0:
            self.mlp = SparseMoeBlock(args)
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen3_5TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args=args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ssm_idx = 0
        self.fa_idx = args.full_attention_interval - 1

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states,
                mask=mask,
                cache=c,
                position_ids=position_ids,
            )

        return self.norm(hidden_states)

    def forward_dflash(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        input_embeddings: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        if (
            ENABLE_EXPLICIT_CACHE_COMPILED_VERIFY
            and (
            return_rollback_records
            and input_embeddings is None
            and position_ids is None
            and inputs.shape[0] == 1
            and inputs.shape[1] > 1
            )
        ):
            full_layer_indices = [idx for idx, layer in enumerate(self.layers) if not layer.is_linear]
            linear_layer_indices = [idx for idx, layer in enumerate(self.layers) if layer.is_linear]
            if all(cache[idx].keys is not None and cache[idx].values is not None for idx in full_layer_indices) and all(
                cache[idx][0] is not None and cache[idx][1] is not None for idx in linear_layer_indices
            ):
                compiled = get_compiled_verify_with_rollback_fn(
                    self,
                    tuple(layer_ids),
                    int(inputs.shape[1]),
                )
                full_keys = [cache[idx].keys for idx in full_layer_indices]
                full_values = [cache[idx].values for idx in full_layer_indices]
                linear_initial_conv_states = [cache[idx][0] for idx in linear_layer_indices]
                linear_initial_states = [cache[idx][1] for idx in linear_layer_indices]
                (
                    norm_hidden_states,
                    target_hidden,
                    new_full_keys,
                    new_full_values,
                    new_linear_conv_states,
                    new_linear_states,
                    record_qkv,
                    record_k,
                    record_v,
                    record_g,
                    record_beta,
                ) = compiled(
                    inputs,
                    cache[self.fa_idx].offset,
                    full_keys,
                    full_values,
                    linear_initial_conv_states,
                    linear_initial_states,
                )
                for idx, new_keys, new_values in zip(
                    full_layer_indices,
                    new_full_keys,
                    new_full_values,
                ):
                    cache[idx].update_and_fetch(new_keys, new_values)
                for idx, new_conv_state, new_state in zip(
                    linear_layer_indices,
                    new_linear_conv_states,
                    new_linear_states,
                ):
                    cache[idx][0] = new_conv_state
                    cache[idx][1] = new_state

                rollback_records: dict[int, dict[str, mx.array]] = {}
                for bundle_idx, layer_idx in enumerate(linear_layer_indices):
                    linear = self.layers[layer_idx].linear_attn
                    rollback_records[layer_idx] = {
                        "initial_conv_state": linear_initial_conv_states[bundle_idx],
                        "initial_state": linear_initial_states[bundle_idx],
                        "qkv": record_qkv[bundle_idx : bundle_idx + 1],
                        "k": record_k[bundle_idx : bundle_idx + 1],
                        "v": record_v[bundle_idx : bundle_idx + 1],
                        "g": record_g[bundle_idx : bundle_idx + 1],
                        "beta": record_beta[bundle_idx : bundle_idx + 1],
                        "repeat_factor": linear.num_v_heads // linear.num_k_heads,
                    }
                return norm_hidden_states, target_hidden, rollback_records

        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        selected_hidden_states: list[mx.array] = []
        target_layer_ids = set(layer_ids)
        rollback_records: dict[int, dict[str, mx.array]] = {}

        for idx, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if return_rollback_records and layer.is_linear:
                hidden_states, rollback_record = forward_linear_layer_with_rollback_record(
                    layer,
                    hidden_states,
                    mask,
                    layer_cache,
                )
                rollback_records[idx] = rollback_record
            else:
                if layer.is_linear:
                    hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
                else:
                    hidden_states = forward_full_attention_layer_dflash(
                        layer,
                        hidden_states,
                        mask,
                        layer_cache,
                        position_ids=position_ids,
                    )
            if idx in target_layer_ids:
                selected_hidden_states.append(hidden_states)

        norm_hidden_states = self.norm(hidden_states)
        target_hidden = mx.concatenate(selected_hidden_states, axis=-1)
        if return_rollback_records:
            return norm_hidden_states, target_hidden, rollback_records
        return norm_hidden_states, target_hidden

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        snapshots: dict[int, list[mx.array | None]] = {}
        for idx, layer_cache in enumerate(cache):
            if isinstance(layer_cache, ArraysCache):
                snapshots[idx] = [
                    None if value is None else mx.array(value) for value in layer_cache.cache
                ]
        return snapshots

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        for idx, values in snapshots.items():
            layer_cache = cache[idx]
            layer_cache.cache = [
                None if value is None else mx.array(value) for value in values
            ]
            layer_cache.left_padding = None
            layer_cache.lengths = None

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        layer_indices: list[int] = []
        initial_states: list[mx.array] = []
        keys: list[mx.array] = []
        values: list[mx.array] = []
        gs: list[mx.array] = []
        betas: list[mx.array] = []

        for idx, record in rollback_records.items():
            layer_cache = cache[idx]
            initial_conv_state = record["initial_conv_state"]
            qkv = record["qkv"]
            n_keep = initial_conv_state.shape[1]
            conv_prefix = mx.concatenate(
                [initial_conv_state, qkv[:, :accepted_inputs, :]],
                axis=1,
            )
            layer_cache[0] = conv_prefix[:, -n_keep:, :]
            layer_indices.append(idx)
            initial_states.append(record["initial_state"])
            record_keys = record["k"][:, :accepted_inputs]
            repeat_factor = int(record["repeat_factor"])
            if repeat_factor > 1:
                record_keys = mx.repeat(record_keys, repeat_factor, axis=2)
            keys.append(record_keys)
            values.append(record["v"][:, :accepted_inputs])
            gs.append(record["g"][:, :accepted_inputs])
            betas.append(record["beta"][:, :accepted_inputs])

        if not layer_indices:
            return

        rebuilt_states = advance_gated_delta_states(
            initial_states=mx.concatenate(initial_states, axis=0),
            keys=mx.concatenate(keys, axis=0),
            values=mx.concatenate(values, axis=0),
            g=mx.concatenate(gs, axis=0),
            beta=mx.concatenate(betas, axis=0),
        )
        for offset, idx in enumerate(layer_indices):
            cache[idx][1] = rebuilt_states[offset : offset + 1]


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs, config: Any | None = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen3_5TextModel(args)
        self._position_ids: mx.array | None = None
        self._rope_deltas: mx.array | None = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))

        vision_config = getattr(self.config, "vision_config", {}) or {}
        if isinstance(vision_config, dict):
            spatial_merge_size = vision_config.get("spatial_merge_size", 2)
        else:
            spatial_merge_size = vision_config.spatial_merge_size

        image_token_id = getattr(self.config, "image_token_id", 248056)
        video_token_id = getattr(self.config, "video_token_id", 248057)
        vision_start_token_id = getattr(self.config, "vision_start_token_id", 248053)
        mrope_position_deltas: list[mx.array] = []

        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones(
                (3, input_ids.shape[0], input_ids.shape[1]),
                dtype=input_ids.dtype,
            )
            image_index = 0
            video_index = 0
            for batch_idx, batch_input_ids in enumerate(total_input_ids):
                masked_input_ids = mx.where(
                    attention_mask[batch_idx] == 1,
                    batch_input_ids,
                    mx.zeros_like(batch_input_ids),
                )
                vision_start_indices = mx.where(
                    masked_input_ids == vision_start_token_id,
                    mx.arange(masked_input_ids.shape[0]),
                    mx.zeros_like(masked_input_ids),
                )
                vision_start_indices = vision_start_indices[
                    masked_input_ids == vision_start_token_id
                ]
                vision_tokens = (
                    masked_input_ids[vision_start_indices + 1]
                    if vision_start_indices.size > 0
                    else mx.array([], dtype=masked_input_ids.dtype)
                )
                image_nums = int((vision_tokens == image_token_id).sum().item())
                video_nums = int((vision_tokens == video_token_id).sum().item())

                input_tokens = masked_input_ids.tolist()
                llm_pos_ids_list: list[mx.array] = []
                start = 0
                remain_images = image_nums
                remain_videos = video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        image_pos = input_tokens.index(image_token_id, start)
                    else:
                        image_pos = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        video_pos = input_tokens.index(video_token_id, start)
                    else:
                        video_pos = len(input_tokens) + 1

                    if image_pos < video_pos:
                        t, h, w = image_grid_thw[image_index].tolist()
                        image_index += 1
                        remain_images -= 1
                        end = image_pos
                    else:
                        t, h, w = video_grid_thw[video_index].tolist()
                        video_index += 1
                        remain_videos -= 1
                        end = video_pos

                    llm_grid_t = int(t)
                    llm_grid_h = int(h) // spatial_merge_size
                    llm_grid_w = int(w) // spatial_merge_size
                    text_len = end - start
                    start_idx = (
                        int(llm_pos_ids_list[-1].max().item()) + 1
                        if llm_pos_ids_list
                        else 0
                    )
                    text_index = mx.arange(text_len).reshape(1, text_len)
                    text_index = mx.broadcast_to(text_index, (3, text_len)) + start_idx
                    llm_pos_ids_list.append(text_index)

                    t_index = mx.arange(llm_grid_t).reshape(llm_grid_t, 1)
                    t_index = mx.broadcast_to(
                        t_index,
                        (llm_grid_t, llm_grid_h * llm_grid_w),
                    ).flatten()
                    h_index = mx.arange(llm_grid_h).reshape(1, llm_grid_h, 1)
                    h_index = mx.broadcast_to(
                        h_index,
                        (llm_grid_t, llm_grid_h, llm_grid_w),
                    ).flatten()
                    w_index = mx.arange(llm_grid_w).reshape(1, 1, llm_grid_w)
                    w_index = mx.broadcast_to(
                        w_index,
                        (llm_grid_t, llm_grid_h, llm_grid_w),
                    ).flatten()
                    llm_pos_ids_list.append(
                        mx.stack([t_index, h_index, w_index]) + text_len + start_idx
                    )
                    start = end + llm_grid_t * llm_grid_h * llm_grid_w

                if start < len(input_tokens):
                    start_idx = (
                        int(llm_pos_ids_list[-1].max().item()) + 1
                        if llm_pos_ids_list
                        else 0
                    )
                    text_len = len(input_tokens) - start
                    text_index = mx.arange(text_len).reshape(1, text_len)
                    text_index = mx.broadcast_to(text_index, (3, text_len)) + start_idx
                    llm_pos_ids_list.append(text_index)

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                mask = mx.array(attention_mask[batch_idx] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(
                    expanded_mask,
                    (3, 1, mask.shape[0]),
                )
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask,
                    expanded_positions,
                    position_ids[:, batch_idx : batch_idx + 1, :],
                )
                position_ids = mx.concatenate(
                    [
                        position_ids[:, :batch_idx, :],
                        new_positions,
                        position_ids[:, batch_idx + 1 :, :],
                    ],
                    axis=1,
                )
                mrope_position_deltas.append(
                    mx.array(int(llm_positions.max().item()) + 1 - len(total_input_ids[batch_idx]))
                )

            return position_ids, mx.array(mrope_position_deltas)

        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(
                attention_mask == 0,
                mx.ones_like(position_ids),
                position_ids,
            )
            position_ids = mx.expand_dims(position_ids[0], axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))
            max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(
                position_ids,
                (3, input_ids.shape[0], input_ids.shape[1]),
            )
            mrope_position_deltas = mx.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas

    def resolve_position_ids(
        self,
        inputs: mx.array,
        cache: Optional[Any],
        *,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        if position_ids is not None:
            return position_ids

        if pixel_values is not None:
            self._rope_deltas = None
            self._position_ids = None

        cache_offset = 0
        if cache and cache[self.model.fa_idx] is not None:
            offset = cache[self.model.fa_idx].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        rope_mask = attention_mask
        if attention_mask is not None and attention_mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if rope_mask is None or rope_mask.ndim == 2:
            if (
                ((cache is not None and cache[self.model.fa_idx] is not None and cache_offset == 0))
                or self._rope_deltas is None
                or cache is None
            ):
                if self._position_ids is not None:
                    seq_length = inputs.shape[1]
                    return self._position_ids[
                        :,
                        :,
                        cache_offset : cache_offset + seq_length,
                    ]

                resolved_position_ids, rope_deltas = self.get_rope_index(
                    inputs,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=rope_mask,
                )
                self._rope_deltas = rope_deltas
                self._position_ids = resolved_position_ids
                return resolved_position_ids

            batch_size, seq_length = inputs.shape
            delta = mx.array(cache_offset + self._rope_deltas if cache is not None else 0)
            position_ids = mx.arange(seq_length).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))
            if cache_offset is not None:
                if delta.ndim == 0:
                    delta = mx.expand_dims(delta, axis=0)
                if delta.shape[0] < batch_size:
                    delta = mx.tile(delta, (batch_size, 1))
                else:
                    delta = delta[:batch_size]

            position_ids = mx.add(position_ids, delta)[None, ...]
            return mx.broadcast_to(position_ids, (3, batch_size, seq_length))

        return build_position_ids(inputs.shape[0], inputs.shape[1])

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        resolved_position_ids = self.resolve_position_ids(
            inputs,
            cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        out = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            position_ids=resolved_position_ids,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def forward_dflash(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        resolved_position_ids = self.resolve_position_ids(
            inputs,
            cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        outputs = self.model.forward_dflash(
            inputs=inputs,
            cache=cache,
            layer_ids=layer_ids,
            input_embeddings=input_embeddings,
            position_ids=resolved_position_ids,
            return_rollback_records=return_rollback_records,
        )
        if return_rollback_records:
            norm_hidden_states, target_hidden, rollback_records = outputs
        else:
            norm_hidden_states, target_hidden = outputs

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(norm_hidden_states)
        else:
            logits = self.lm_head(norm_hidden_states)

        if return_rollback_records:
            return logits, target_hidden, rollback_records
        return logits, target_hidden

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return self.model.snapshot_linear_caches(cache)

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        self.model.restore_linear_caches(cache, snapshots)

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        self.model.rollback_linear_caches(cache, rollback_records, accepted_inputs)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]

    def sanitize(self, weights):
        has_mtp_weights = any("mtp." in k for k in weights)
        has_unsanitized_conv1d = any(
            "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
        )
        should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
        weights = {k: v for k, v in weights.items() if "mtp." not in k}

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in weights.items():
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
                if v.ndim == 1:
                    weights[k] = v + 1.0
        return weights

    @property
    def quant_predicate(self):
        if self.args.num_experts <= 0:
            return None

        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(path: str):
            if path.endswith("A_log"):
                return False
            return True

        return predicate


@dataclass
class VisionModelArgs(BaseModelArgs):
    model_type: str = "qwen3_5"
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] = field(default_factory=list)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        return mx.outer(seq, inv_freq)


def apply_rotary_pos_emb_vision(tensor: mx.array, freqs: mx.array) -> mx.array:
    orig_dtype = tensor.dtype
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    cos = mx.expand_dims(cos, axis=1)
    cos = mx.tile(cos, (1, 1, 2))
    cos = mx.expand_dims(cos, axis=0)
    sin = mx.expand_dims(sin, axis=1)
    sin = mx.tile(sin, (1, 1, 2))
    sin = mx.expand_dims(sin, axis=0)
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.astype(orig_dtype)


class VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.hidden_size)


class VisionPatchMerger(nn.Module):
    def __init__(self, config: VisionModelArgs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x).reshape(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array = None,
    ) -> mx.array:
        seq_length = x.shape[0]
        qkv = self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3)
        q = apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
        k = apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        splits = [mx.split(tensor, cu_seqlens[1:-1].tolist(), axis=2) for tensor in (q, k, v)]
        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(*splits):
            attn_outputs.append(
                mx.fast.scaled_dot_product_attention(
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    scale=self.scale,
                )
            )
        output = mx.concatenate(attn_outputs, axis=2)
        output = output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        return self.proj(output)


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear_fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.act_fn = nn.GELU(approx="tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionModelArgs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(dim=config.hidden_size, num_heads=config.num_heads)
        self.mlp = VisionMLP(dim=config.hidden_size, hidden_dim=config.intermediate_size)

    def __call__(self, hidden_states: mx.array, cu_seqlens: mx.array, rotary_pos_emb: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings,
            config.hidden_size,
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = VisionPatchMerger(config=config)

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        merge_size = self.spatial_merge_size
        max_hw = int(mx.max(grid_thw[:, 1:]).item())
        freq_table = self.rotary_pos_emb(max_hw)
        pos_ids = []
        for num_frames, height, width in grid_thw.tolist():
            num_frames = int(num_frames)
            height = int(height)
            width = int(width)
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = mx.arange(merged_h)
            block_cols = mx.arange(merged_w)
            intra_row = mx.arange(merge_size)
            intra_col = mx.arange(merge_size)
            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )
            row_idx = mx.broadcast_to(
                row_idx,
                (merged_h, merged_w, merge_size, merge_size),
            ).reshape(-1)
            col_idx = mx.broadcast_to(
                col_idx,
                (merged_h, merged_w, merge_size, merge_size),
            ).reshape(-1)
            coords = mx.stack([row_idx, col_idx], axis=-1)
            if num_frames > 1:
                coords = mx.tile(coords, (num_frames, 1))
            pos_ids.append(coords)
        pos_ids = mx.concatenate(pos_ids, axis=0)
        h_embeddings = freq_table[pos_ids[:, 0]]
        w_embeddings = freq_table[pos_ids[:, 1]]
        return mx.concatenate([h_embeddings, w_embeddings], axis=-1)

    def fast_pos_embed_interpolate(self, grid_thw: mx.array) -> mx.array:
        grid_thw_list = grid_thw.tolist()
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid_thw_list:
            h = int(h)
            w = int(w)
            t = int(t)
            h_idxs = mx.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = mx.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.astype(mx.int32)
            w_idxs_floor = w_idxs.astype(mx.int32)
            h_idxs_ceil = mx.minimum(h_idxs_floor + 1, self.num_grid_per_side - 1)
            w_idxs_ceil = mx.minimum(w_idxs_floor + 1, self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor.astype(mx.float32)
            dw = w_idxs - w_idxs_floor.astype(mx.float32)
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
            for idx in range(4):
                idx_list[idx].extend(indices[idx].tolist())
                weight_list[idx].extend(weights[idx].tolist())

        idx_tensor = mx.array(idx_list, dtype=mx.int32)
        weight_tensor = mx.array(weight_list, dtype=self.pos_embed.weight.dtype)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        split_sizes = [int(h * w) for t, h, w in grid_thw_list]
        if len(split_sizes) > 1:
            split_indices = list(accumulate(split_sizes[:-1]))
            patch_pos_embeds_split = mx.split(patch_pos_embeds, split_indices, axis=0)
        else:
            patch_pos_embeds_split = [patch_pos_embeds]

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds_split, grid_thw_list):
            t = int(t)
            h = int(h)
            w = int(w)
            feature_dim = pos_embed.shape[-1]
            pos_embed = mx.tile(pos_embed, (t, 1))
            pos_embed = pos_embed.reshape(t, h, w, feature_dim)
            pos_embed = (
                pos_embed.reshape(
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    feature_dim,
                )
                .transpose(0, 1, 3, 2, 4, 5)
                .reshape(-1, feature_dim)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return mx.concatenate(patch_pos_embeds_permute)

    def __call__(self, hidden_states: mx.array, grid_thw: mx.array) -> tuple[mx.array, list[mx.array]]:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        batch_size = grid_thw.shape[0]
        cu_seqlens = []
        for idx in range(batch_size):
            seq_len = grid_thw[idx, 1] * grid_thw[idx, 2]
            cu_seqlens.append(mx.repeat(seq_len, grid_thw[idx, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        hidden_states = self.merger(hidden_states)
        return hidden_states, []


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict
    vision_config: dict | None = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(
                model_type=params["model_type"],
                text_config=params,
                vision_config=params.get("vision_config"),
                image_token_id=params.get("image_token_id", 248056),
                video_token_id=params.get("video_token_id", 248057),
                vision_start_token_id=params.get("vision_start_token_id", 248053),
                vision_end_token_id=params.get("vision_end_token_id", 248054),
            )
        return super().from_dict(params)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.vision_tower = (
            VisionModel(VisionModelArgs.from_dict(args.vision_config))
            if args.vision_config is not None
            else None
        )
        self.language_model = TextModel(
            TextModelArgs.from_dict(args.text_config),
            config=args,
        )

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features: mx.array,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        image_token_id: int,
        video_token_id: int,
    ) -> tuple[mx.array, mx.array]:
        special_image_mask = input_ids == image_token_id
        special_video_mask = input_ids == video_token_id
        special_image_mask = special_image_mask | special_video_mask
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(
            inputs_embeds,
            special_image_mask,
            image_features,
        )
        return inputs_embeds, special_image_mask

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        cached_image_features: Optional[mx.array] = None,
    ) -> mx.array:
        if pixel_values is None:
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            return self.language_model.model.embed_tokens(input_ids)

        if self.vision_tower is None:
            raise ValueError("This model instance does not have a vision tower.")

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        hidden_states = (
            cached_image_features
            if cached_image_features is not None
            else self.vision_tower(pixel_values, image_grid_thw)[0]
        )
        inputs_embeds, _ = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.args.image_token_id,
            self.args.video_token_id,
        )
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas
        return inputs_embeds

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
    ):
        if input_embeddings is None and pixel_values is not None:
            input_embeddings = self.get_input_embeddings(
                input_ids=inputs,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
        )

    def forward_dflash(
        self,
        inputs: mx.array,
        cache: list[Any],
        layer_ids: list[int],
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        return_rollback_records: bool = False,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, dict[int, dict[str, mx.array]]]:
        if input_embeddings is None and pixel_values is not None:
            input_embeddings = self.get_input_embeddings(
                input_ids=inputs,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        return self.language_model.forward_dflash(
            inputs=inputs,
            cache=cache,
            layer_ids=layer_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            return_rollback_records=return_rollback_records,
        )

    def snapshot_linear_caches(
        self,
        cache: list[Any],
    ) -> dict[int, list[mx.array | None]]:
        return self.language_model.snapshot_linear_caches(cache)

    def restore_linear_caches(
        self,
        cache: list[Any],
        snapshots: dict[int, list[mx.array | None]],
    ) -> None:
        self.language_model.restore_linear_caches(cache, snapshots)

    def rollback_linear_caches(
        self,
        cache: list[Any],
        rollback_records: dict[int, dict[str, mx.array]],
        accepted_inputs: int,
    ) -> None:
        self.language_model.rollback_linear_caches(cache, rollback_records, accepted_inputs)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif key.startswith("model.language_model"):
                key = key.replace("model.language_model", "language_model.model")
            elif key.startswith("language_model."):
                pass
            elif key.startswith("vision_tower."):
                pass
            else:
                key = "language_model." + key
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    def shard(self, group=None):
        group = group or mx.distributed.init()
        N = group.size()
        rank = group.rank()

        # A sharding factory for the convolution in gated delta net
        def conv_sharding(key_dim):
            return lambda p, w: (0, [key_dim, 2 * key_dim])

        def repeat_kv_layer_inplace(layer, h):
            # No repeat needed cause we have more heads than nodes
            if N <= h:
                return

            # Repeat function to apply to the layer weights
            def _repeat(p):
                s = p.shape
                p = p.reshape(h, s[0] // h, *s[1:])
                p = mx.repeat(p, N // h, axis=0)
                p = p.reshape(-1, *s[1:])
                return p

            layer.update(tree_map(_repeat, layer.parameters()))

        for layer in self.layers:
            # Linear attention
            if layer.is_linear:
                kd = layer.linear_attn.key_dim
                layer.linear_attn.sharding_group = group
                shard_inplace(layer.linear_attn.conv1d, conv_sharding(kd), group=group)
                layer.linear_attn.conv1d.groups //= N
                shard_inplace(
                    layer.linear_attn.in_proj_qkv,
                    "all-to-sharded",
                    segments=[kd, 2 * kd],
                    group=group,
                )
                shard_inplace(
                    layer.linear_attn.in_proj_z, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_b, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.linear_attn.in_proj_a, "all-to-sharded", group=group
                )
                layer.linear_attn.dt_bias = mx.contiguous(
                    mx.split(layer.linear_attn.dt_bias, N)[rank]
                )
                layer.linear_attn.A_log = mx.contiguous(
                    mx.split(layer.linear_attn.A_log, N)[rank]
                )
                shard_inplace(layer.linear_attn.out_proj, "sharded-to-all", group=group)
                layer.linear_attn.num_k_heads //= N
                layer.linear_attn.num_v_heads //= N
                layer.linear_attn.key_dim //= N
                layer.linear_attn.value_dim //= N
                layer.linear_attn.conv_dim //= N

            # Softmax attention
            else:
                layer.self_attn.o_proj = shard_linear(
                    layer.self_attn.o_proj, "sharded-to-all", group=group
                )
                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.k_proj, layer.self_attn.num_key_value_heads
                )
                repeat_kv_layer_inplace(
                    layer.self_attn.v_proj, layer.self_attn.num_key_value_heads
                )
                layer.self_attn.k_proj = shard_linear(
                    layer.self_attn.k_proj, "all-to-sharded", group=group
                )
                layer.self_attn.v_proj = shard_linear(
                    layer.self_attn.v_proj, "all-to-sharded", group=group
                )
                layer.self_attn.num_attention_heads //= N
                layer.self_attn.num_key_value_heads = max(
                    1, layer.self_attn.num_key_value_heads // N
                )

            # MLP
            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj, "all-to-sharded", group=group
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj, "sharded-to-all", group=group
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj, "all-to-sharded", group=group
                )

            # MoE
            else:
                layer.mlp.sharding_group = group
                shard_inplace(
                    layer.mlp.shared_expert.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.shared_expert.up_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.gate_proj, "all-to-sharded", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.down_proj, "sharded-to-all", group=group
                )
                shard_inplace(
                    layer.mlp.switch_mlp.up_proj, "all-to-sharded", group=group
                )

    @property
    def layers(self):
        return self.language_model.model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
