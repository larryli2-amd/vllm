# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.verify_extension import (
    FullAttentionVerifyBackend,
    FullAttentionVerifyExtension,
    FullAttentionVerifyKVCache,
    FullAttentionVerifyMetadata,
    FullAttentionVerifyRows,
    build_full_attention_verify_metadata,
    dispatch_full_attention_verify,
    should_dispatch_full_attention_verify,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)
_ROCM_ATTN_PAGED_KV_LAYOUT = "ROCM_ATTN_PAGED_KV_V1"


@dataclass
class RocmAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None

    # DFlash drafting sets this to False via CommonAttentionMetadata.
    causal: bool = True

    full_attn_verify_extension: FullAttentionVerifyExtension | None = None
    full_attn_verify_metadata: FullAttentionVerifyMetadata | None = None


class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()
        self.full_attn_verify_extension: FullAttentionVerifyExtension | None = None
        self.full_attn_verify_backend: FullAttentionVerifyBackend | None = None

    def set_full_attention_verify_extension(
        self,
        extension: FullAttentionVerifyExtension,
        backend: FullAttentionVerifyBackend,
    ) -> None:
        expected_backend = FullAttentionVerifyBackend.from_backend(RocmAttentionBackend)
        if backend != expected_backend:
            raise ValueError(
                "ROCM_ATTN can only attach an extension for its exact backend identity."
            )
        if getattr(self.kv_cache_spec, "is_draft", False):
            raise ValueError("ROCM_ATTN full-attention verify dispatch is target-only.")
        if getattr(self.kv_cache_spec, "non_causal", False):
            raise ValueError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "non-causal KV cache specs."
            )
        if self.vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "decode context parallelism."
            )
        if is_quantized_kv_cache(self.vllm_config.cache_config.cache_dtype):
            raise ValueError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "quantized KV cache."
            )
        self.full_attn_verify_extension = extension
        self.full_attn_verify_backend = backend

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        if self.full_attn_verify_extension is not None:
            raise RuntimeError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "CUDA graph capture."
            )
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)

        # Zero device query start locations to avoid invalid memory access in
        # the prefix prefill kernel during graph capture (#25985).
        common_attn_metadata.query_start_loc.zero_()

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        full_attn_verify_metadata = None
        if self.full_attn_verify_extension is not None:
            assert self.full_attn_verify_backend is not None
            full_attn_verify_metadata = build_full_attention_verify_metadata(
                self.full_attn_verify_backend, common_attn_metadata
            )

        attn_metadata = RocmAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            full_attn_verify_extension=self.full_attn_verify_extension,
            full_attn_verify_metadata=full_attn_verify_metadata,
        )
        return attn_metadata


class RocmAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # ROCM paged attention native C++ kernel only supports block sizes 16 and 32
        # due to shared memory (LDS) constraints on AMD GPUs.
        # The (head_size, block_size) pairs that have a native kernel are enumerated
        # by csrc/rocm/attention_paged_attn_<head>_<dtype>_<kv>.cu, each of which
        # instantiates ATTN_LAUNCH from csrc/rocm/attention_kernels.cuh; the runtime
        # gate is use_rocm_custom_paged_attention() in vllm/platforms/rocm.py.
        # However, vLLM allows support for any multiple of 16 via the Triton path.
        # As addressed in PR: https://github.com/vllm-project/vllm/pull/31380,
        # non-standard models (like qwen3-next with block_size 544, or qwen3_5
        # with 784 and 1056) are dynamically routed to our optimized Triton kernel
        # in `do_kv_cache_update`.
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Head sizes with a native ROCm kernel are only 64 and 128 (see
        # csrc/rocm/attention_paged_attn_h{64,128}_*.cu); the remaining entries are
        # served by the Triton fallback selected in use_rocm_custom_paged_attention().
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Not implemented
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        # ROCM custom attention kernel does not support sinks.
        # Callink this backend with sinks will cause it to fall back to the Triton
        # kernel, which is less efficient than the proper triton backends.
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is
        # incompatible with KV connectors that require blocks-first layout.
        return False

    @classmethod
    def supports_full_attention_verify_extension(cls) -> bool:
        return True

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["RocmAttentionImpl"]:
        return RocmAttentionImpl

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ENCODER_DECODER is not supported because
        chunked_prefill_paged_decode's prefill kernel (context_attention_fwd)
        assumes self-attention semantics: it treats passed K/V as new tokens
        to mix with cached K/V. For cross-attention layers the encoder K/V
        are already fully cached, so mixing them again produces incorrect
        results when max_query_len > 1 (e.g. beam search).
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """K and V as two head groups so the native HIP kernels address each side
        as one contiguous region (x-packed interior applied in split_kv_cache)."""
        if spec.state_content_bytes is not None:
            return spec
        assert spec.head_size == spec.head_size_v, (
            "Separate K/V head groups require symmetric K/V head sizes."
        )
        return replace(
            spec,
            num_head_slots=2,
            state_content_bytes=spec.num_kv_heads
            * spec.head_size
            * get_dtype_size(spec.dtype),
        )

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The native HIP kernels hardcode group-contiguous block addressing, so
        # they need the K and V groups to span all blocks (H outermost within the layer)
        # The Triton fallbacks are stride-aware, so block-interior LBHNC is also
        # supported, just without the native kernels.
        return (KVCacheLayout.LHBNC, KVCacheLayout.LBHNC)

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder


class RocmAttentionImpl(AttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

        self._use_interleaved_v_cache = False

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer

        """
        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
            sinks=self.sinks,
        )
        return output

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact K/V views used by the ROCM_ATTN forward path."""
        return PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )

    def _full_attention_verify_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> FullAttentionVerifyKVCache:
        """Adapt ROCM_ATTN's native cache views for a verify extension."""
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        return FullAttentionVerifyKVCache(
            layout=_ROCM_ATTN_PAGED_KV_LAYOUT,
            native=kv_cache,
            key=key_cache,
            value=value_cache,
        )

    def _validate_full_attention_verify_dispatch(
        self,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        attn_metadata: RocmAttentionMetadata,
        verify_metadata: FullAttentionVerifyMetadata,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> None:
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch only supports decoder "
                "attention."
            )
        if key is None or value is None:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "shared or missing K/V."
            )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "quantized KV cache."
            )
        if self.sliding_window != (-1, -1):
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "sliding-window attention."
            )
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "ALiBi slopes or attention sinks."
            )
        if self.logits_soft_cap != 0:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "logits soft cap."
            )
        if self._use_interleaved_v_cache:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "interleaved V cache."
            )
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "fused output quantization."
            )
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "decode context parallelism."
            )
        if attn_metadata.causal is not True:
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch only supports "
                "causal metadata."
            )
        is_prefilling = verify_metadata.is_prefilling
        if is_prefilling is None or is_prefilling.device.type != "cpu":
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch requires CPU "
                "is_prefilling metadata."
            )
        if bool(is_prefilling.any()):
            raise NotImplementedError(
                "ROCM_ATTN full-attention verify dispatch does not support "
                "prefill or mixed prefill/verify batches."
            )

    def _maybe_dispatch_full_attention_verify(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        extension = attn_metadata.full_attn_verify_extension
        verify_metadata = attn_metadata.full_attn_verify_metadata
        if extension is None and verify_metadata is None:
            return False
        if extension is None or verify_metadata is None:
            raise RuntimeError(
                "ROCM_ATTN full-attention verify metadata and extension must "
                "either both be set or both be absent."
            )
        if not should_dispatch_full_attention_verify(extension, verify_metadata):
            return False

        self._validate_full_attention_verify_dispatch(
            key,
            value,
            attn_metadata,
            verify_metadata,
            output_scale,
            output_block_scale,
        )
        assert key is not None and value is not None
        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str):
            raise RuntimeError(
                "ROCM_ATTN full-attention verify dispatch requires an attention "
                "layer name."
            )
        num_actual_tokens = attn_metadata.num_actual_tokens
        rows = FullAttentionVerifyRows(
            layer_name=layer_name,
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens],
            value=value[:num_actual_tokens],
            kv_cache=self._full_attention_verify_kv_cache(kv_cache),
            output=output[:num_actual_tokens],
            metadata=verify_metadata,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            scale=self.scale,
            kv_cache_dtype=self.kv_cache_dtype,
            attn_type=self.attn_type,
            sliding_window=None,
            logits_soft_cap=None,
            alibi_slopes=None,
            sinks=None,
            q_scale=layer._q_scale,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
        )
        dispatch_full_attention_verify(extension, rows)
        return True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: logical [num_blocks, 2, block_size, num_kv_heads *
                head_size] under LHBNC (physically K/V-group-first)
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if self._maybe_dispatch_full_attention_verify(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        ):
            return output

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # The bound view is logical [B, 2, N, H*hs]; split_kv_cache expects
        # the K/V groups first.
        key_cache, value_cache = self._split_kv_cache(kv_cache)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            # q_scale only applies to an fp8 query; this path keeps the query
            # in full precision, so a non-1.0 q_scale is not applicable here.
            if query.dtype == self.fp8_dtype and layer._q_scale_float != 1.0:
                raise NotImplementedError(
                    "A non 1.0 q_scale with an fp8 query is not currently "
                    "supported by RocmAttentionImpl."
                )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # Compute attention and update output up to `num_actual_tokens`.
        chunked_prefill_paged_decode(
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens] if key is not None else None,
            value=value[:num_actual_tokens] if value is not None else None,
            output=output[:num_actual_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            query_start_loc=cu_seqlens_q,
            seq_lens=seqused_k,
            max_seq_len=max_seqlen_k,
            max_query_len=max_seqlen_q,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            alibi_slopes=self.alibi_slopes,
            # self.sliding_window[0] is the FlashAttention-style left span (W - 1).
            # chunked_prefill_paged_decode expects the full window length W.
            sliding_window=1 + self.sliding_window[0],
            sm_scale=self.scale,
            output_scale=output_scale,
            sinks=self.sinks,
            use_interleaved_v_cache=self._use_interleaved_v_cache,
            causal=attn_metadata.causal,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1), self.num_kv_heads, self.head_size
        )

        # Reshape the input keys and values and store them in the cache.
        # Get the actual block_size from value_cache
        # value_cache shape: [num_blocks, num_heads, head_size, block_size]
        block_size = value_cache.shape[3]
        has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)

        # Match the fused writer's interleaved V-cache when fusion is active.
        interleaved_v = self._use_interleaved_v_cache
        interleaved_v_pack_factor = (
            (16 // value_cache.element_size()) if interleaved_v else 0
        )

        if block_size in (16, 32) and has_native_layout:
            # Normal 16, 32 with contiguous blocks: use vLLM native HIP C++ logic.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
                interleaved_v,
            )
        else:
            # Non-standard blocks and hybrid attention/Mamba layouts need the
            # stride-aware Triton writer. The native reshape_and_cache kernel
            # assumes contiguous block storage and writes to the wrong hybrid
            # cache blocks.
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
                interleaved_v_pack_factor,
            )

    def fused_qk_norm_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def set_interleaved_v_cache(self):
        # Decode/prefill kernels must read the interleaved V-cache the AITER
        # fused write emits when use_shuffle_layout=True.
        self._use_interleaved_v_cache = True

    def do_qk_norm_rope_kvcache_update(
        self,
        layer: AttentionLayer,
        qkv: torch.Tensor,
        q_out: torch.Tensor,
        k_out: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        rms_norm_eps: float,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        # kv_cache is logical [num_blocks, 2, block_size, num_kv_heads * head_size]
        # (see forward); the AITER shuffle writer takes one (B, N, H, hs) per side.
        key_cache, value_cache = kv_cache.unflatten(
            -1, (self.num_kv_heads, self.head_size)
        ).unbind(1)
        use_shuffle_layout = self._use_interleaved_v_cache
        rocm_aiter_ops.do_qk_norm_rope_kvcache_update(
            qkv=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            head_dim=self.head_size,
            is_neox=is_neox,
            rms_norm_eps=rms_norm_eps,
            q_out=q_out,
            k_out=k_out,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=layer_slot_mapping,
            k_scale=layer._k_scale_cpu,
            v_scale=layer._v_scale_cpu,
            kv_cache_dtype=self.kv_cache_dtype,
            use_shuffle_layout=use_shuffle_layout,
            return_kv=True,
        )

    def fused_rope_kvcache_supported(self):
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        if self._use_interleaved_v_cache:
            # The Triton RoPE + cache writer only stores the regular V layout.
            # With layer_name hoisted, one RoPE + KV-cache pattern serves every
            # layer, so a layer switched to the interleaved V cache is handled
            # here at run time rather than left out of the fusion.
            ops.rotary_embedding(
                positions,
                query,
                key,
                layer.head_size,  # type: ignore[attr-defined]
                cos_sin_cache,
                is_neox,
            )
            self.do_kv_cache_update(layer, key, value, kv_cache, layer_slot_mapping)
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache.transpose(0, 1),
            layer.num_kv_heads,  # type: ignore[attr-defined]
            layer.head_size,  # type: ignore[attr-defined]
        )
        flash_layout = False

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
