"""Native Transformers implementation of the Limite causal language model.

BF16 SDPA is the portable attention path. The implementation also supports
Transformers' cache protocol so the same model can be used by ``generate``
without a separate decoding graph.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from .configuration_limite import LimiteConfig
from .registration import register_weight_converters

register_weight_converters()


def _uses_external_flash_attention(attn_implementation: str | None) -> bool:
    """Identify native and Hub-provided FlashAttention implementations."""
    normalized = str(attn_implementation or "").lower().replace("-", "_")
    return "flash_attention" in normalized or "flash_attn" in normalized


def _reject_external_flash_static_cache(attn_implementation: str | None) -> None:
    if _uses_external_flash_attention(attn_implementation):
        raise ValueError(
            "Limite does not support FlashAttention with StaticCache because "
            "that combination can produce incorrect logits. Use "
            "attn_implementation='sdpa' with StaticCache, or use "
            "DynamicCache with FlashAttention."
        )


def _validate_cache_attention_pair(
    *,
    attn_implementation: str | None,
    past_key_values: Cache | None,
) -> None:
    if isinstance(past_key_values, StaticCache):
        _reject_external_flash_static_cache(attn_implementation)


def rms_norm(x: Tensor) -> Tensor:
    """Gain-free RMS norm with PyTorch's dtype-dependent default epsilon."""
    return F.rms_norm(x, (x.size(-1),))


class LimiteRMSNorm(nn.Module):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return rms_norm(hidden_states)


class LimiteRotaryEmbedding(nn.Module):
    """Checkpoint-exact rotary factors with the static frequencies cached."""

    def __init__(self, config: LimiteConfig) -> None:
        super().__init__()
        self.rope_base_local = float(config.rope_base_local)
        self.rope_n_pairs = int(config.rope_n_pairs)
        self.head_dim = int(config.head_dim)
        self.register_buffer(
            "frequency",
            self._build_frequency(),
            persistent=False,
        )

    def _build_frequency(self, device: torch.device | None = None) -> Tensor:
        frequency = (1.0 / self.rope_base_local) ** torch.linspace(
            0,
            1,
            steps=self.rope_n_pairs,
            dtype=torch.float32,
            device="cpu",
        )
        frequency = frequency.repeat_interleave(2)
        frequency = torch.cat(
            [frequency, frequency.new_zeros(self.head_dim - frequency.numel())]
        )
        return frequency if device is None else frequency.to(device=device)

    def _apply(self, fn: Any, recurse: bool = True) -> "LimiteRotaryEmbedding":
        super()._apply(fn, recurse=recurse)
        # Transformers applies ``dtype=...`` to buffers too. RoPE frequencies
        # are part of Limite's FP32 numerical contract, so reconstruct the
        # derived buffer from the CPU-FP32 formula on the destination device.
        self.frequency = self._build_frequency(device=self.frequency.device)
        return self

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        theta = position_ids.to(torch.float32).unsqueeze(-1) * self.frequency
        cosine = theta.cos().to(torch.bfloat16).unsqueeze(-2)
        sine = theta.sin().to(torch.bfloat16)
        sine[..., 1::2] *= -1
        return cosine, sine.unsqueeze(-2)


def apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    paired = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
    return cosine * x + sine * paired


def repeat_kv(hidden_states: Tensor, num_groups: int) -> Tensor:
    """Expand key/value heads for the eager attention oracle."""
    if num_groups == 1:
        return hidden_states
    batch_size, num_kv_heads, sequence_length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_kv_heads,
        num_groups,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(
        batch_size,
        num_kv_heads * num_groups,
        sequence_length,
        head_dim,
    )


def eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """Reference attention used for backend parity checks."""
    del kwargs
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    probabilities = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = F.dropout(
        probabilities,
        p=dropout,
        training=module.training,
    )
    output = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
    return output, probabilities


def _fp32_parameter(*shape: int, initial: float = 0.0) -> nn.Parameter:
    return nn.Parameter(torch.full(shape, initial, dtype=torch.float32))


class LimiteAttention(nn.Module):
    def __init__(self, config: LimiteConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = int(config.head_dim)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.num_key_value_groups = self.num_kv_groups
        self.scaling = float(config.attention_softmax_scale)
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = True
        self.is_global = config.is_global_layer(layer_idx)
        self.window_span = None if self.is_global else int(config.sliding_window)
        self.applies_rope = not (self.is_global and bool(config.global_nope))
        self.has_ve = layer_idx in set(config.ve_layers)
        self.has_xsa = bool(config.xsa) and layer_idx in set(config.xsa_layers)
        self.attn_gate_channels = int(config.attn_gate_channels)

        hidden_size = int(config.hidden_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        self.qkv_scale = _fp32_parameter(initial=1.0)
        self.o_scale = _fp32_parameter(initial=1.0)
        self.register_buffer("_inference_qkv_weight", None, persistent=False)
        self.register_buffer("_inference_o_weight", None, persistent=False)
        self.register_buffer("_inference_xsa_alpha", None, persistent=False)
        self.register_buffer("_inference_ve_gate", None, persistent=False)
        self.register_buffer("_inference_attn_gate", None, persistent=False)
        if self.has_xsa:
            self.xsa_alpha = _fp32_parameter(self.num_heads)
        if self.has_ve:
            self.ve_gate = _fp32_parameter(
                int(config.ve_stored_heads), int(config.ve_gate_channels)
            )
        if self.attn_gate_channels:
            self.attn_gate = _fp32_parameter(self.num_heads, self.attn_gate_channels)

    @staticmethod
    def _scaled(weight: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
        return (scale.to(torch.float32).view(()) * weight).to(dtype)

    def train(self, mode: bool = True) -> "LimiteAttention":
        super().train(mode)
        if mode:
            self._inference_qkv_weight = None
            self._inference_o_weight = None
            self._inference_xsa_alpha = None
            self._inference_ve_gate = None
            self._inference_attn_gate = None
        else:
            dtype = self.qkv_proj.weight.dtype
            self._inference_qkv_weight = self._scaled(
                self.qkv_proj.weight,
                self.qkv_scale,
                dtype,
            ).detach()
            self._inference_o_weight = self._scaled(
                self.o_proj.weight, self.o_scale, self.o_proj.weight.dtype
            ).detach()
            if self.has_xsa:
                self._inference_xsa_alpha = torch.tanh(self.xsa_alpha.float()).detach()
            if self.has_ve:
                self._inference_ve_gate = (
                    self.ve_gate[: self.num_kv_heads].to(dtype).detach()
                )
            if self.attn_gate_channels:
                self._inference_attn_gate = self.attn_gate.to(dtype).detach()
        return self

    def _project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        sizes = (self.q_size, self.kv_size, self.kv_size)
        if not self.training and self._inference_qkv_weight is not None:
            return F.linear(hidden_states, self._inference_qkv_weight).split(
                sizes, dim=-1
            )
        dtype = hidden_states.dtype
        return F.linear(
            hidden_states,
            self._scaled(self.qkv_proj.weight, self.qkv_scale, dtype),
        ).split(
            sizes,
            dim=-1,
        )

    def _apply_value_embeddings(
        self, hidden_states: Tensor, value_embeds: Tensor, value_states: Tensor
    ) -> Tensor:
        gate_weight = (
            self._inference_ve_gate
            if not self.training and self._inference_ve_gate is not None
            else self.ve_gate[: self.num_kv_heads].to(hidden_states.dtype)
        )
        gate = float(self.config.ve_gate_scale) * torch.sigmoid(
            F.linear(hidden_states[..., : gate_weight.size(-1)], gate_weight)
        )
        return value_states + gate.unsqueeze(-1) * value_embeds.to(value_states.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        value_embeds: Tensor | None,
        cosine: Tensor,
        sine: Tensor,
        attention_mask: Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, query_length, _ = hidden_states.shape
        query_states, key_states, value_states = self._project_qkv(hidden_states)
        query_states = query_states.view(
            batch_size, query_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, query_length, self.num_kv_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, query_length, self.num_kv_heads, self.head_dim
        )

        if self.has_ve and value_embeds is not None:
            value_states = self._apply_value_embeddings(
                hidden_states, value_embeds, value_states
            )
        current_values = value_states
        query_states, key_states = rms_norm(query_states), rms_norm(key_states)
        if self.applies_rope:
            query_states = apply_rotary(query_states, cosine, sine)
            key_states = apply_rotary(key_states, cosine, sine)

        if use_cache:
            if past_key_values is None:
                raise ValueError("use_cache=True requires a cache instance")
            cached_keys, cached_values = past_key_values.update(
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                self.layer_idx,
            )
            key_states = cached_keys.transpose(1, 2)
            value_states = cached_values.transpose(1, 2)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is None
            and query_length == 1
        ):
            flash_decode = (
                query_states.is_cuda
                and query_states.dtype in (torch.float16, torch.bfloat16)
                and torch.cuda.get_device_capability(query_states.device)[0] >= 8
            )
            backend_context = (
                sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                if flash_decode
                else nullcontext()
            )
            with backend_context:
                attention_output = (
                    F.scaled_dot_product_attention(
                        query_states.transpose(1, 2),
                        key_states.transpose(1, 2),
                        value_states.transpose(1, 2),
                        dropout_p=(
                            0.0 if not self.training else self.attention_dropout
                        ),
                        scale=self.scaling,
                        is_causal=False,
                        enable_gqa=True,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            probabilities = None
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
            attention_output, probabilities = attention_interface(
                self,
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.window_span,
                output_attentions=output_attentions,
            )
        if self.has_xsa:
            alpha_values = (
                self._inference_xsa_alpha
                if not self.training and self._inference_xsa_alpha is not None
                else torch.tanh(self.xsa_alpha.float())
            )
            if not self.training:
                grouped_output = attention_output.view(
                    batch_size,
                    query_length,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    self.head_dim,
                )
                value_direction = F.normalize(
                    current_values.float(),
                    dim=-1,
                    eps=float(self.config.xsa_normalize_eps),
                ).unsqueeze(3)
                projection = (grouped_output.float() * value_direction).sum(
                    -1, keepdim=True
                )
                alpha = alpha_values.view(
                    1, 1, self.num_kv_heads, self.num_kv_groups, 1
                )
                attention_output = (
                    grouped_output
                    - (alpha * projection * value_direction).to(grouped_output.dtype)
                ).reshape(batch_size, query_length, self.num_heads, self.head_dim)
            else:
                value_direction = F.normalize(
                    current_values.repeat_interleave(self.num_kv_groups, dim=2).float(),
                    dim=-1,
                    eps=float(self.config.xsa_normalize_eps),
                )
                projection = (attention_output.float() * value_direction).sum(
                    -1, keepdim=True
                )
                alpha = alpha_values.view(1, 1, self.num_heads, 1)
                attention_output = attention_output - (
                    alpha * projection * value_direction
                ).to(attention_output.dtype)
        if self.attn_gate_channels:
            gate_weight = (
                self._inference_attn_gate
                if not self.training and self._inference_attn_gate is not None
                else self.attn_gate.to(hidden_states.dtype)
            )
            gate = float(self.config.attn_gate_scale) * torch.sigmoid(
                F.linear(
                    hidden_states[..., : self.attn_gate_channels],
                    gate_weight,
                )
            )
            attention_output = attention_output * gate.to(
                attention_output.dtype
            ).unsqueeze(-1)

        attention_output = attention_output.reshape(batch_size, query_length, -1)
        if not self.training and self._inference_o_weight is not None:
            attention_output = F.linear(attention_output, self._inference_o_weight)
        else:
            attention_output = F.linear(
                attention_output,
                self._scaled(
                    self.o_proj.weight,
                    self.o_scale,
                    attention_output.dtype,
                ),
            )
        return attention_output, probabilities if output_attentions else None


class LimiteMLP(nn.Module):
    def __init__(self, config: LimiteConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.intermediate_size = intermediate_size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate, up = F.linear(
            hidden_states,
            self.gate_up_proj.weight,
        ).split(self.intermediate_size, dim=-1)
        activated = F.silu(gate) * up
        return self.down_proj(activated)


class LimiteDecoderLayer(nn.Module):
    def __init__(self, config: LimiteConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LimiteAttention(config, layer_idx)
        self.mlp = LimiteMLP(config)
        self.resid_lambda_attn = _fp32_parameter(initial=1.0)
        self.post_lambda_attn = _fp32_parameter(initial=1.0)
        self.resid_lambda_mlp = _fp32_parameter(initial=1.0)
        self.post_lambda_mlp = _fp32_parameter(initial=1.0)
        self.register_buffer("_inference_residual_scales", None, persistent=False)

    def train(self, mode: bool = True) -> "LimiteDecoderLayer":
        super().train(mode)
        if mode:
            self._inference_residual_scales = None
        else:
            dtype = self.self_attn.qkv_proj.weight.dtype
            self._inference_residual_scales = (
                torch.stack(
                    (
                        self.resid_lambda_attn,
                        self.post_lambda_attn,
                        self.resid_lambda_mlp,
                        self.post_lambda_mlp,
                    )
                )
                .to(dtype)
                .detach()
            )
        return self

    def forward(
        self,
        hidden_states: Tensor,
        attention_input: Tensor,
        value_embeds: Tensor | None,
        cosine: Tensor,
        sine: Tensor,
        attention_mask: Tensor | None,
        past_key_values: Cache | None,
        use_cache: bool,
        output_attentions: bool,
        attention_residual: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, probabilities = self.self_attn(
            attention_input,
            value_embeds,
            cosine,
            sine,
            attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
        )
        residual_base = (
            hidden_states if attention_residual is None else attention_residual
        )
        use_constants = (
            not self.training and self._inference_residual_scales is not None
        )
        if use_constants:
            residual_scales = self._inference_residual_scales
            mixed = (
                residual_scales[0] * residual_base
                + residual_scales[1] * attention_output
            )
        else:
            mixed = (
                self.resid_lambda_attn.to(hidden_states.dtype) * residual_base
                + self.post_lambda_attn.to(attention_output.dtype) * attention_output
            )
        mlp_output = self.mlp(rms_norm(mixed))
        if use_constants:
            output = residual_scales[2] * mixed + residual_scales[3] * mlp_output
        else:
            output = (
                self.resid_lambda_mlp.to(mixed.dtype) * mixed
                + self.post_lambda_mlp.to(mlp_output.dtype) * mlp_output
            )
        return output, probabilities


class LimiteMudd(nn.Module):
    def __init__(self, config: LimiteConfig) -> None:
        super().__init__()
        self.dense1 = _fp32_parameter(int(config.mudd_inter), int(config.hidden_size))
        self.dense2 = _fp32_parameter(
            int(config.num_hidden_layers),
            int(config.mudd_taps),
            int(config.mudd_inter),
        )
        self.bias = _fp32_parameter(
            int(config.num_hidden_layers), int(config.mudd_taps)
        )
        self.register_buffer("_inference_dense1", None, persistent=False)
        self.register_buffer("_inference_dense2", None, persistent=False)
        self.register_buffer("_inference_bias", None, persistent=False)
        self.register_buffer("_inference_dense2_mlp", None, persistent=False)
        self.register_buffer("_inference_bias_mlp", None, persistent=False)
        self.uses_r_way = bool(config.mudd_mlp)
        if self.uses_r_way:
            self.dense2_mlp = _fp32_parameter(
                int(config.num_hidden_layers),
                int(config.mudd_taps),
                int(config.mudd_inter),
            )
            self.bias_mlp = _fp32_parameter(
                int(config.num_hidden_layers), int(config.mudd_taps)
            )

    def train(self, mode: bool = True) -> "LimiteMudd":
        super().train(mode)
        if mode:
            self._inference_dense1 = None
            self._inference_dense2 = None
            self._inference_bias = None
            self._inference_dense2_mlp = None
            self._inference_bias_mlp = None
        else:
            self._inference_dense1 = self.dense1.to(torch.bfloat16).detach()
            self._inference_dense2 = self.dense2.to(torch.bfloat16).detach()
            self._inference_bias = self.bias.to(torch.bfloat16).detach()
            if self.uses_r_way:
                self._inference_dense2_mlp = self.dense2_mlp.to(torch.bfloat16).detach()
                self._inference_bias_mlp = self.bias_mlp.to(torch.bfloat16).detach()
        return self

    def _inner(self, current: Tensor) -> Tensor:
        use_constants = not self.training and self._inference_dense1 is not None
        dense1 = (
            self._inference_dense1 if use_constants else self.dense1.to(current.dtype)
        )
        return F.gelu(F.linear(rms_norm(current), dense1))

    def _mix(
        self,
        taps: list[Tensor],
        inner: Tensor,
        layer_idx: int,
        *,
        r_way: bool,
    ) -> Tensor:
        count = len(taps)
        use_constants = not self.training and self._inference_dense1 is not None
        if r_way:
            if not self.uses_r_way:
                raise ValueError("R-way mixing requested without R-way weights")
            if use_constants:
                dense2, bias = (
                    self._inference_dense2_mlp,
                    self._inference_bias_mlp,
                )
            else:
                dense2, bias = self.dense2_mlp, self.bias_mlp
        elif use_constants:
            dense2, bias = self._inference_dense2, self._inference_bias
        else:
            dense2, bias = self.dense2, self.bias
        weights = torch.einsum(
            "btk,mk->btm",
            inner,
            dense2[layer_idx, :count].to(inner.dtype),
        )
        weights = weights + bias[layer_idx, :count].to(weights.dtype)
        output = weights[..., 0:1].type_as(taps[0]) * taps[0]
        for index in range(1, count):
            output = (
                output
                + weights[..., index : index + 1].type_as(taps[index]) * taps[index]
            )
        return output

    def forward(
        self,
        taps: list[Tensor],
        current: Tensor,
        layer_idx: int,
        *,
        r_way: bool = False,
    ) -> Tensor:
        return self._mix(
            taps,
            self._inner(current),
            layer_idx,
            r_way=r_way,
        )

    def forward_pair(
        self,
        taps: list[Tensor],
        current: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        if not self.uses_r_way:
            raise ValueError("paired MUDD mixing requires R-way weights")
        inner = self._inner(current)
        return (
            self._mix(taps, inner, layer_idx, r_way=False),
            self._mix(taps, inner, layer_idx, r_way=True),
        )


class LimitePreTrainedModel(PreTrainedModel):
    config_class = LimiteConfig
    base_model_prefix = "model"
    _no_split_modules = ["LimiteDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, LimiteRotaryEmbedding):
            # Transformers materializes non-persistent buffers with
            # ``empty_like`` during low-memory/device-map loading. Restore this
            # derived FP32 buffer before the loaded model is returned.
            module.frequency.copy_(
                module._build_frequency(device=module.frequency.device)
            )


class LimiteModel(LimitePreTrainedModel):
    def __init__(self, config: LimiteConfig) -> None:
        super().__init__(config)
        self.vocab_size = int(config.vocab_size)
        self.embed_tokens = nn.Embedding(
            int(config.vocab_size), int(config.hidden_size)
        )
        self.value_embeds = nn.Embedding(
            int(config.vocab_size), int(config.ve_stored_heads) * int(config.ve_dim)
        )
        self.layers = nn.ModuleList(
            [
                LimiteDecoderLayer(config, layer_idx)
                for layer_idx in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = LimiteRMSNorm()
        self.rotary_emb = LimiteRotaryEmbedding(config)
        self.mudd = LimiteMudd(config) if config.mudd else None
        self.retained_taps = sorted(
            {
                tap
                for layer, taps in config.mudd_tap_idx.items()
                for tap in taps
                if tap != int(layer)
            }
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def _value_embeddings(self, input_ids: Tensor) -> Tensor | None:
        if not self.config.ve_layers:
            return None
        config = self.config
        embeddings = self.value_embeds(input_ids).view(
            *input_ids.shape, int(config.ve_stored_heads), int(config.ve_dim)
        )
        if config.ve_dim < config.head_dim:
            embeddings = F.pad(
                embeddings,
                (0, int(config.head_dim) - int(config.ve_dim)),
            )
        return embeddings[..., : int(config.num_key_value_heads), :].contiguous()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        inputs_embeds: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[Tensor, ...]:
        del kwargs
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "Limite requires input_ids because value embeddings are a "
                "second token lookup"
            )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = bool(output_attentions)
        output_hidden_states = bool(output_hidden_states)
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if output_attentions:
            raise ValueError(
                "Limite does not materialize attention weights; "
                "output_attentions=True is unsupported"
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if use_cache:
            _validate_cache_attention_pair(
                attn_implementation=self.config._attn_implementation,
                past_key_values=past_key_values,
            )
        if (
            use_cache
            and isinstance(past_key_values, DynamicCache)
            and not hasattr(past_key_values, "_limite_unpadded")
        ):
            if attention_mask is None:
                past_key_values._limite_unpadded = True
            elif isinstance(attention_mask, Tensor):
                past_key_values._limite_unpadded = not bool(
                    (attention_mask == 0).any().item()
                )
            else:
                past_key_values._limite_unpadded = False
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                position_ids = position_ids[:, -input_ids.shape[1] :]
            else:
                past_length = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                position_ids = (
                    torch.arange(
                        past_length,
                        past_length + input_ids.shape[1],
                        device=input_ids.device,
                    )
                    .unsqueeze(0)
                    .expand(input_ids.shape[0], -1)
                )

        cosine, sine = self.rotary_emb(position_ids)
        value_embeds = self._value_embeddings(input_ids)
        hidden_states = rms_norm(self.embed_tokens(input_ids))
        maskless_sdpa_decode = (
            self.config._attn_implementation == "sdpa"
            and isinstance(past_key_values, DynamicCache)
            and input_ids.shape[1] == 1
            and bool(getattr(past_key_values, "_limite_unpadded", False))
        )
        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        elif maskless_sdpa_decode:
            causal_mask_mapping = {
                "full_attention": None,
                "sliding_attention": None,
            }
        else:
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        history: dict[int, Tensor] = (
            {0: hidden_states} if 0 in self.retained_taps else {}
        )
        all_hidden_states: tuple[Tensor, ...] = ()
        all_attentions: tuple[Tensor, ...] = ()

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            taps = self.config.tap_indices(layer_idx)
            if taps is not None:
                tap_values = [
                    history[tap] if tap != layer_idx else hidden_states for tap in taps
                ]
                if not self.training and self.mudd.uses_r_way:
                    attention_mix, residual_base = self.mudd.forward_pair(
                        tap_values, hidden_states, layer_idx
                    )
                    attention_input = rms_norm(attention_mix)
                else:
                    attention_input = rms_norm(
                        self.mudd(tap_values, hidden_states, layer_idx)
                    )
                    residual_base = (
                        self.mudd(
                            tap_values,
                            hidden_states,
                            layer_idx,
                            r_way=True,
                        )
                        if self.mudd.uses_r_way
                        else hidden_states
                    )
            else:
                attention_input = rms_norm(hidden_states)
                residual_base = hidden_states
            hidden_states, probabilities = decoder_layer(
                hidden_states,
                attention_input,
                value_embeds,
                cosine,
                sine,
                causal_mask_mapping[self.config.layer_types[layer_idx]],
                past_key_values,
                use_cache,
                output_attentions,
                residual_base,
            )
            if output_attentions:
                all_attentions += (probabilities,)
            if layer_idx + 1 in self.retained_taps:
                history[layer_idx + 1] = hidden_states

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            values: tuple[Tensor | Cache | tuple[Tensor, ...], ...] = (hidden_states,)
            if use_cache:
                values += (past_key_values,)
            if output_hidden_states:
                values += (all_hidden_states,)
            if output_attentions:
                values += (all_attentions,)
            return values
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )


class LimiteForCausalLM(LimitePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: LimiteConfig) -> None:
        super().__init__(config)
        self.model = LimiteModel(config)
        self.vocab_size = int(config.vocab_size)
        self.lm_head = nn.Linear(
            int(config.hidden_size), int(config.vocab_size), bias=False
        )
        softcap = dict(config.softcap_logits)
        self.softcap_a = float(softcap["a"])
        self.softcap_b = float(softcap["b"])
        self.softcap_c = float(softcap["c"])
        self.head_precision_mode = str(config.lm_head_precision_mode)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.lm_head = value

    def get_decoder(self) -> LimiteModel:
        return self.model

    def set_decoder(self, decoder: LimiteModel) -> None:
        self.model = decoder

    def _prepare_static_cache(
        self,
        cache_implementation: str,
        batch_size: int,
        max_cache_len: int,
        model_kwargs: dict[str, Any],
    ) -> Cache:
        # GenerationMixin allocates its persistent static cache before the
        # first model forward. Reject the unsupported backend/cache pair here
        # so users receive the Limite contract error instead of failing inside
        # Transformers' cache preparation. SDPA remains entirely native.
        _reject_external_flash_static_cache(self.config._attn_implementation)
        return super()._prepare_static_cache(
            cache_implementation,
            batch_size,
            max_cache_len,
            model_kwargs,
        )

    def _softcapped_logits(self, hidden_states: Tensor) -> Tensor:
        if self.head_precision_mode == "oracle_exact":
            logits = F.linear(hidden_states, self.lm_head.weight).float()
        elif hidden_states.is_cuda and hidden_states.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            logits = torch.mm(
                hidden_states.reshape(-1, hidden_states.shape[-1]),
                self.lm_head.weight.t(),
                out_dtype=torch.float32,
            ).reshape(*hidden_states.shape[:-1], self.lm_head.weight.shape[0])
        else:
            logits = F.linear(hidden_states.float(), self.lm_head.weight.float())
        return self.softcap_a * torch.sigmoid(
            (logits + self.softcap_b) / self.softcap_c
        )

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | Tensor = 0,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Tensor, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int) and logits_to_keep > 0
            else logits_to_keep
            if isinstance(logits_to_keep, Tensor)
            else slice(None)
        )
        logits = self._softcapped_logits(hidden_states[:, indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            result = (logits, outputs.past_key_values)
            return ((loss,) + result) if loss is not None else result
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LimiteDecoderLayer",
    "LimiteForCausalLM",
    "LimiteModel",
    "LimitePreTrainedModel",
]
