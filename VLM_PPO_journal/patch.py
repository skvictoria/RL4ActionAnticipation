"""
Directly copied the code from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py and made some adjustments
"""

import inspect
import logging
import math
from typing import Optional, Tuple

import torch
import transformers.models.llama.modeling_llama
from torch import nn

try:
    import xformers.ops
except ImportError:
    logging.error("xformers not found! Please install it before trying to use it.")


def replace_llama_attn_with_xformers_attn():
    attention_cls = transformers.models.llama.modeling_llama.LlamaAttention
    params = list(inspect.signature(attention_cls.forward).parameters.keys())
    if "position_embeddings" in params:
        logging.warning(
            "Detected Transformers LlamaAttention signature incompatible with legacy xformers patch, "
            "skipping attention override."
        )
        return False
    attention_cls.forward = xformers_forward
    return True


def xformers_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # pylint: disable=duplicate-code
    bsz, q_len, _ = hidden_states.size()

    if past_key_value is None and past_key_values is not None:
        past_key_value = past_key_values

    num_heads = getattr(self, "num_heads", self.config.num_attention_heads)
    num_kv_heads = getattr(self, "num_key_value_heads", self.config.num_key_value_heads)
    num_groups = getattr(self, "num_key_value_groups", num_heads // num_kv_heads)
    head_dim = getattr(self, "head_dim", self.config.hidden_size // num_heads)
    hidden_size = getattr(self, "hidden_size", self.config.hidden_size)

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, num_heads, head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, num_kv_heads, head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, num_kv_heads, head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    rotary_positions = cache_position if cache_position is not None else position_ids
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len, position_ids=rotary_positions)
    (
        query_states,
        key_states,
    ) = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        cached_k, cached_v = past_key_value
        if isinstance(cached_k, tuple):
            cached_k = cached_k[-1]
        if isinstance(cached_v, tuple):
            cached_v = cached_v[-1]
        key_states = torch.cat([cached_k, key_states], dim=2)
        value_states = torch.cat([cached_v, value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # We only apply xformers optimizations if we don't need to output the whole attention matrix
    if not output_attentions:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if num_kv_heads != num_heads:
            key_states = repeat_kv(key_states, num_groups)
            value_states = repeat_kv(value_states, num_groups)

        # This is a nasty hack. We know attention_mask in transformers is either LowerTriangular or all Zeros.
        # We therefore check if one element in the upper triangular portion is zero. If it is, then the mask is all zeros.
        if attention_mask is None or attention_mask[0, 0, 0, 1] == 0:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states, key_states, value_states, attn_bias=None
            )
        else:
            # input and output should be of form (bsz, q_len, num_heads, head_dim)
            attn_output = xformers.ops.memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                attn_bias=xformers.ops.LowerTriangularMask(),
            )
        attn_weights = None
    else:
        if num_kv_heads != num_heads:
            key_states = repeat_kv(key_states, num_groups)
            value_states = repeat_kv(value_states, num_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value
