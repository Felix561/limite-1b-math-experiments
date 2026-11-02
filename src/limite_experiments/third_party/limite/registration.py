"""Checkpoint-layout registration for the Hub-hosted Limite implementation."""

import torch
from transformers import Chunk, Concatenate, WeightConverter
from transformers.conversion_mapping import register_checkpoint_conversion_mapping

from .configuration_limite import LimiteConfig


class _SplitGQAQKV(Chunk):
    """Reverse a fused GQA projection without assuming equal Q/K/V sizes."""

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        *,
        config: LimiteConfig,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del source_patterns, kwargs
        value = next(iter(input_dict.values()))
        tensor = value[0] if isinstance(value, list) else value
        q_size = int(config.num_attention_heads) * int(config.head_dim)
        kv_size = int(config.num_key_value_heads) * int(config.head_dim)
        expected = q_size + 2 * kv_size
        if tensor.shape[self.dim] != expected:
            raise ValueError(
                "Invalid fused QKV size: "
                f"expected {expected}, found {tensor.shape[self.dim]}."
            )
        chunks = tensor.split((q_size, kv_size, kv_size), dim=self.dim)
        return dict(zip(target_patterns, chunks, strict=True))

    @property
    def reverse_op(self) -> "_ConcatenateGQAQKV":
        return _ConcatenateGQAQKV(self.dim)


class _ConcatenateGQAQKV(Concatenate):
    """Fuse Q/K/V while retaining a GQA-aware reverse save transform."""

    @property
    def reverse_op(self) -> _SplitGQAQKV:
        return _SplitGQAQKV(self.dim)


def register_weight_converters() -> None:
    """Register split-checkpoint to fused-runtime transformations."""
    register_checkpoint_conversion_mapping(
        LimiteConfig.model_type,
        [
            WeightConverter(
                source_patterns=[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                target_patterns="self_attn.qkv_proj.weight",
                operations=[_ConcatenateGQAQKV(dim=0)],
            ),
            WeightConverter(
                source_patterns=[
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                ],
                target_patterns="mlp.gate_up_proj.weight",
                operations=[Concatenate(dim=0)],
            ),
        ],
        overwrite=True,
    )


__all__ = ["register_weight_converters"]
