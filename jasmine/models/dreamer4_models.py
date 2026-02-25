import math
import itertools
from typing import Tuple, Callable, Dict, Any, Optional
from collections import OrderedDict
from enum import IntEnum
import numpy as np

from flax import nnx
import flax
import jax
import jax.numpy as jnp
import einops

import optax
import orbax.checkpoint as ocp

from jasmine.utils.dreamer4_utils import patchify, unpatchify, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck



class Modality(IntEnum):
    LATENT   = -1
    IMAGE    = 0
    ACTION   = 1
    PROPRIO  = 2
    REGISTER = 3
    SPATIAL = 4
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@flax.struct.dataclass  # immutable, PyTree-friendly
class TokenLayout:
    """
    Ordered token layout for a single timestep: latents first (if any),
    then a sequence of (modality, count) segments.
    """
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]  # e.g., ((Modality.IMAGE, n_patches), (Modality.ACTION, n_act), ...)

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> np.ndarray:
        parts = [np.full((self.n_latents,), Modality.LATENT, dtype=np.int32)] if self.n_latents > 0 else []
        for m, n in self.segments:
            if n > 0:
                parts.append(np.full((n,), int(m), dtype=np.int32))
        return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.int32)  # (S,)

    def slices(self) -> dict:
        """Convenience: start/stop indices per modality (first occurrence if repeated)."""
        idx = 0
        out = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents); idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


def _get_spatiotemporal_positional_encoding(d_model: int, max_len: int = 5000):
    """
    Creates a function that applies separate sinusoidal positional encodings to the temporal and spatial dimensions.
    """
    pe = jnp.zeros((max_len, d_model))
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    def _encode(x: jax.Array) -> jax.Array:
        """
        Args:
            x: The input tensor of shape (Batch, Time, Space, Dimension).

        Returns:
            The input tensor with positional encodings added.
        """
        assert x.ndim == 4, f"Input must be 4-dimensional, but got shape {x.shape}"

        num_timesteps = x.shape[1]
        num_spatial_patches = x.shape[2]

        # Temporal positional encoding: (1, T, 1, D)
        temporal_pe = pe[jnp.newaxis, :num_timesteps, jnp.newaxis, :]
        x = x + temporal_pe

        # Spatial positional encoding: (1, 1, S, D)
        spatial_pe = pe[jnp.newaxis, jnp.newaxis, :num_spatial_patches, :]
        x = x + spatial_pe

        return x

    return _encode


def _get_rotary_positional_encoding(
    head_dim: int, 
    max_len: int = 5000, 
    base: float = 10000.0,
    dtype: jnp.dtype = jnp.bfloat16,
):
    """
    Creates Rotary Position Embedding (RoPE) for attention.
    
    RoPE applies rotation to query and key vectors based on their position,
    encoding relative position information directly into the attention mechanism.
    
    Args:
        head_dim: Dimension per attention head (must be even)
        max_len: Maximum sequence length
        base: Base for the sinusoidal frequencies (default: 10000.0)
        dtype: Data type for computations (default: jnp.bfloat16)
    
    Returns:
        A function that applies RoPE to query and key tensors.
    
    Reference: https://arxiv.org/abs/2104.09864 (RoFormer)
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    
    # Precompute in float32 for numerical precision, then cast to target dtype.
    # Exponentiation and trigonometric functions are sensitive to bfloat16 precision loss.
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim))
    
    # Precompute position * frequency table: (max_len, half_dim)
    positions = jnp.arange(max_len, dtype=jnp.float32)
    freqs = jnp.outer(positions, inv_freq)  # (max_len, half_dim)
    
    # Precompute and cache cos/sin in float32 for maximum precision.
    # Cast to target dtype only at the point of use (multiplication with q/k).
    cos_cached = jnp.cos(freqs)  # (max_len, half_dim), float32
    sin_cached = jnp.sin(freqs)  # (max_len, half_dim), float32
    
    def _rotate_half(x: jax.Array) -> jax.Array:
        """Rotate half the hidden dims of x: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]"""
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return jnp.concatenate([-x2, x1], axis=-1)
    
    def _apply_rope(
        q: jax.Array, 
        k: jax.Array, 
        positions: Optional[jax.Array] = None
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            q: Query tensor of shape (..., seq_len, num_heads, head_dim)
            k: Key tensor of shape (..., seq_len, num_heads, head_dim)
            positions: Optional position indices of shape (..., seq_len).
                       If None, uses sequential positions [0, 1, 2, ...].
        
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs.
        """
        seq_len = q.shape[-3]
        
        if positions is None:
            # Use sequential positions
            cos = cos_cached[:seq_len]  # (seq_len, half_dim)
            sin = sin_cached[:seq_len]  # (seq_len, half_dim)
        else:
            # Gather cos/sin based on provided positions
            cos = cos_cached[positions]  # (..., seq_len, half_dim)
            sin = sin_cached[positions]  # (..., seq_len, half_dim)
        
        # Expand dims to match q/k shape: (..., seq_len, 1, half_dim)
        # Then tile to full head_dim: (..., seq_len, 1, head_dim)
        cos = jnp.concatenate([cos, cos], axis=-1)
        sin = jnp.concatenate([sin, sin], axis=-1)
        
        # Add head dimension
        if cos.ndim == 2:
            cos = cos[:, None, :]  # (seq_len, 1, head_dim)
            sin = sin[:, None, :]
        else:
            cos = cos[..., None, :]  # (..., seq_len, 1, head_dim)
            sin = sin[..., None, :]
        
        # Cast to target dtype right before multiplication with q/k
        cos = cos.astype(dtype)
        sin = sin.astype(dtype)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        q_rotated = q * cos + _rotate_half(q) * sin
        k_rotated = k * cos + _rotate_half(k) * sin
        
        return q_rotated, k_rotated
    
    return _apply_rope


def _create_flash_attention_fn(
    use_flash_attention: bool, 
    is_causal: bool, 
    mask: np.ndarray = None,
    rope_fn: Optional[Callable] = None,
) -> Callable:
    """
    Create an attention function that uses flash attention if enabled.

    flax.nnx.MultiHeadAttention provides tensors with shape (batch..., length, num_heads, head_dim),
    but jax.nn.dot_product_attention expects (batch, length, num_heads, head_dim). We reshape to
    ensure compatibility. cuDNN's flash attention additionally requires a sequence length that
    is a multiple of 4. We pad the sequence length to the nearest multiple of 4 and mask
    accordingly. Note that cuDNN requires the mask to be broadcast before calling the attention
    function due to strict shape checking.

    Args:
        use_flash_attention: Whether to use cuDNN flash attention.
        is_causal: Whether to apply causal masking.
        mask: np.ndarray with shape (S, S) for custom attention masking.
        rope_fn: Optional RoPE function to apply to query and key tensors.
                 Should have signature: (q, k, positions) -> (q_rotated, k_rotated)
    """
    if mask is not None:
        mask = jnp.array(mask)

    def attention_fn(
        query_BTHD, key_BSHD, value_BSHD, bias=None, mask_B111=None, **kwargs
    ):
        implementation = "cudnn" if use_flash_attention else None

        def _merge_batch_dims(x):
            return einops.rearrange(x, "... l h k -> (...) l h k")

        def _pad(x, pad_size):
            return jnp.pad(x, ((0, 0), (0, pad_size), (0, 0), (0, 0)))

        original_shape = query_BTHD.shape
        T = query_BTHD.shape[-3]
        S = key_BSHD.shape[-3]

        # Apply RoPE before merging batch dims if provided
        if rope_fn is not None:
            query_BTHD, key_BSHD = rope_fn(query_BTHD, key_BSHD, positions=None)

        # Pad to nearest multiple of 4
        Q = ((T + 3) // 4) * 4
        pad_size_Q = Q - T
        K = ((S + 3) // 4) * 4
        pad_size_K = K - S

        query_BQHD = _pad(_merge_batch_dims(query_BTHD), pad_size_Q)
        key_BKHD = _pad(_merge_batch_dims(key_BSHD), pad_size_K)
        value_BKHD = _pad(_merge_batch_dims(value_BSHD), pad_size_K)

        if mask is not None:
            attention_mask = jnp.pad(mask, ((0, pad_size_Q), (0, pad_size_K))).astype(jnp.bool_)
        else:
            attention_mask = jnp.ones((Q, K), dtype=jnp.bool_)
        attention_mask = attention_mask.at[T:, :].set(False)
        attention_mask = attention_mask.at[:, S:].set(False)

        mask_11TS = attention_mask[jnp.newaxis, jnp.newaxis, :, :]

        bias_4d = (
            jnp.pad(
                _merge_batch_dims(bias),
                ((0, 0), (0, 0), (0, pad_size_Q), (0, pad_size_K)),
            )
            if bias is not None
            else None
        )

        # NOTE: jax.nn.dot_product_attention does not support dropout
        output_4d = jax.nn.dot_product_attention(
            query=query_BQHD,
            key=key_BKHD,
            value=value_BKHD,
            bias=bias_4d,
            mask=mask_11TS,
            implementation=implementation,
            is_causal=is_causal,
        )
        return output_4d[..., :T, :, :].reshape(original_shape)

    return attention_fn


class MLP(nnx.Module):
    """
    2-layer MLP with swiglu activation
    """

    def __init__(
        self,
        d_model: int,
        mlp_ratio: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.mlp_ratio = mlp_ratio
        self.param_dtype = param_dtype
        self.dtype = dtype
        mult = self.mlp_ratio * (2.0 / 3.0)
        hidden = int(self.d_model * mult)  # param parity with GELU MLP
        self.pre = nnx.Linear(
            in_features=self.d_model,
            out_features=hidden * 2,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.linear = nnx.Linear(
            in_features=hidden,
            out_features=self.d_model,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pre(x)
        u, v = jnp.split(x, 2, axis=-1)     # (..., H), (..., H)
        h = u * jax.nn.silu(v)                # (..., H)
        y = self.linear(h)
        return y


class ModalityAxialBlock(nnx.Module):
    """Spatial transformer block"""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        mode: str,
        modality_ids: jnp.ndarray,
        sow_weights: bool,
        sow_activations: bool,
        decode: bool,
        spatial_causal: bool,
        temporal_causal: bool,
        layer_index: int,
        time_every: int,
        pos_emb_type: str = "sinusoidal",  # "sinusoidal", "rope", or "none"
        max_len: int = 5000,
    ):
        """
        modality_ids: jnp.ndarray with shape (S,), per-token modality id for the S tokens.
        pos_emb_type: "sinusoidal" (additive PE in transformer), "rope" (rotary in attention), or "none"
        """
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.mode = mode
        self.modality_ids = modality_ids
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations
        self.decode = decode
        self.spatial_causal = spatial_causal
        self.temporal_causal = temporal_causal
        self.layer_index = layer_index
        self.time_every = time_every
        self.pos_emb_type = pos_emb_type
        
        # Create RoPE functions if needed
        head_dim = dim // num_heads
        self.spatial_rope_fn = None
        self.temporal_rope_fn = None
        if pos_emb_type == "rope":
            self.spatial_rope_fn = _get_rotary_positional_encoding(head_dim, max_len=max_len, dtype=dtype)
            self.temporal_rope_fn = _get_rotary_positional_encoding(head_dim, max_len=max_len, dtype=dtype)
        # generate mask
        S = int(self.modality_ids.shape[0])
        q_idx = np.arange(S)[:, None]   # (S,1)
        k_idx = np.arange(S)[None, :]   # (1,S)
        q_mod = self.modality_ids[q_idx] # (S,1)
        k_mod = self.modality_ids[k_idx] # (1,S)
        same_mod = (q_mod == k_mod)           # (S,S)
        is_q_lat = q_mod == Modality.LATENT     # (S,1) bool
        is_k_lat = k_mod == Modality.LATENT     # (1,S) bool

        if self.mode == "encoder":
            # latents -> all; non-latents -> same modality only (no access to latents)
            allow_lat_q = np.ones((S, S), dtype=bool)             # lat q attends to everything
            allow_nonlat_q = same_mod                              # non-lat q attends within itself only
            self.mask = np.where(is_q_lat, allow_lat_q, allow_nonlat_q)
        elif self.mode == "decoder":
            # latents -> latents only; non-latents -> same modality OR latents
            allow_lat_q = is_k_lat                                  # lat q -> lat k only
            allow_nonlat_q = np.logical_or(same_mod, is_k_lat)     # non-lat q -> same mod + latents
            self.mask = np.where(is_q_lat, allow_lat_q, allow_nonlat_q)
        elif self.mode in ["wm_agent", "wm_agent_isolated"]:
            S = int(self.modality_ids.shape[0])
            q_idx = np.arange(S)[:, None]   # (S,1)
            k_idx = np.arange(S)[None, :]   # (1,S)
            q_mod = self.modality_ids[q_idx] # (S,1)
            k_mod = self.modality_ids[k_idx] # (1,S)

            is_agent_q = (q_mod == Modality.AGENT)
            is_agent_k = (k_mod == Modality.AGENT)
            is_action_q = (q_mod == Modality.ACTION)
            is_action_k = (k_mod == Modality.ACTION)

            # Observation bucket = spatial ∪ register ∪ shortcut tokens
            is_obs_k = (
                (k_mod == Modality.SPATIAL) |
                (k_mod == Modality.REGISTER) |
                (k_mod == Modality.SHORTCUT_SIGNAL) |
                (k_mod == Modality.SHORTCUT_STEP)
            )
            is_obs_q = (
                (q_mod == Modality.SPATIAL) |
                (q_mod == Modality.REGISTER) |
                (q_mod == Modality.SHORTCUT_SIGNAL) |
                (q_mod == Modality.SHORTCUT_STEP)
            )

            # Agent queries:
            #  - wm_agent: agent reads all (obs ∪ action ∪ agent)
            #  - wm_agent_isolated: agent reads nobody
            allow_for_agent_q = np.where(
                self.mode == "wm_agent",
                np.ones((S, S), dtype=bool),
                np.zeros((S, S), dtype=bool)
            )

            # Non-agent queries (route by query modality)
            allow_for_action_q = is_action_k                                  # action -> action only  (1,S)
            allow_for_obs_q    = (is_obs_k | is_action_k)                     # obs -> obs ∪ action    (1,S)

            # Build per-query row permissions with broadcasting from (1,S) to (S,S)
            allow_nonagent = np.where(
                is_action_q, allow_for_action_q,
                np.where(is_obs_q, allow_for_obs_q, np.zeros((S, S), dtype=bool))
            )

            # Nobody can read agent keys except agent q
            allow_nonagent = np.where(is_agent_k, False, allow_nonagent)

            self.mask = np.where(is_agent_q, allow_for_agent_q, allow_nonagent)
        else:
            raise ValueError(f"Unknown mode {self.mode}")


        self.spatial_norm = nnx.RMSNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # rms norm in full precision
            rngs=rngs,
        )
        self.spatial_attention = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=self.dim,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            attention_fn=_create_flash_attention_fn(
                self.use_flash_attention, 
                is_causal=self.spatial_causal, 
                mask=self.mask,
                rope_fn=self.spatial_rope_fn,
            ),
            rngs=rngs,
            decode=self.decode,
        )

        if (self.layer_index + 1) % self.time_every == 0:
            self.temporal_norm = nnx.RMSNorm(
                num_features=self.dim,
                param_dtype=self.param_dtype,
                dtype=self.param_dtype,  # rms norm in full precision
                rngs=rngs,
            )
            self.temporal_attention = nnx.MultiHeadAttention(
                num_heads=self.num_heads,
                in_features=self.dim,
                qkv_features=self.dim,
                dropout_rate=self.dropout,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                attention_fn=_create_flash_attention_fn(
                    self.use_flash_attention, 
                    is_causal=self.temporal_causal,
                    rope_fn=self.temporal_rope_fn,
                ),
                rngs=rngs,
                decode=self.decode,
            )

        self.ffn_norm = nnx.RMSNorm(
            num_features=self.dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # rms norm in full precision
            rngs=rngs,
        )
        self.mlp = MLP(
            d_model=self.dim,
            mlp_ratio=self.mlp_ratio,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )


    @nnx.remat
    def __call__(self, x_BTNM: jax.Array) -> jax.Array:
        # --- Spatial attention ---
        z_BTNM = self.spatial_norm(x_BTNM)
        z_BTNM = self.spatial_attention(z_BTNM, sow_weights=self.sow_weights)
        x_BTNM = x_BTNM + z_BTNM

        if (self.layer_index + 1) % self.time_every == 0:
            # --- Temporal attention ---
            x_BNTM = x_BTNM.swapaxes(1, 2)
            z_BNTM = self.temporal_norm(x_BNTM)
            z_BNTM = self.temporal_attention(z_BNTM, sow_weights=self.sow_weights)
            x_BNTM = x_BNTM + z_BNTM
            x_BTNM = x_BNTM.swapaxes(1, 2)

        # --- Feedforward ---
        z_BTNM = self.ffn_norm(x_BTNM)
        z_BTNM = self.mlp(z_BTNM)
        x_BTNM = x_BTNM + z_BTNM
        if self.sow_activations:
            self.sow(nnx.Intermediate, "activations", x_BTNM)
        return x_BTNM


class ModalityAxialTransformer(nnx.Module):
    """
    Modality axial transformer

    Dimension keys:
        B: batch size
        T: number of frames
        N: number of patches per frame
        I: number of input features
        M: model dimension
        D: FFN dimension
        O: output dimension

    modality_ids: modality hint for attention masking
    """

    def __init__(
        self,
        model_dim: int,
        mlp_ratio: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        mode: str,
        modality_ids: jnp.ndarray,
        spatial_causal: bool,
        temporal_causal: bool,
        decode: bool = False,
        sow_weights: bool = False,
        sow_activations: bool = False,
        sow_logits: bool = False,
        max_len: int = 5000,
        time_every: int = 4,
        pos_emb_type: str = "sinusoidal",  # "sinusoidal", "rope", or "none"
    ):
        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.mode = mode
        self.modality_ids = modality_ids
        self.spatial_causal = spatial_causal
        self.temporal_causal = temporal_causal
        self.sow_logits = sow_logits
        self.sow_weights = sow_weights
        self.sow_activations = sow_activations
        self.decode = decode
        self.time_every = time_every
        self.pos_emb_type = pos_emb_type
        self.max_len = max_len
        self.input_norm = nnx.RMSNorm(
            num_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.param_dtype,  # rms norm in full precision
            rngs=rngs,
        )

        # Only create sinusoidal PE if using that type (RoPE is applied in attention)
        self.pos_enc = None
        if pos_emb_type == "sinusoidal":
            self.pos_enc = _get_spatiotemporal_positional_encoding(
                self.model_dim, max_len=max_len
            )

        self.blocks = []
        for layer_index in range(self.num_blocks):
            self.blocks.append(
                ModalityAxialBlock(
                    dim=self.model_dim,
                    mlp_ratio=self.mlp_ratio,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    param_dtype=self.param_dtype,
                    dtype=self.dtype,
                    use_flash_attention=self.use_flash_attention,
                    rngs=rngs,
                    mode=self.mode,
                    modality_ids=self.modality_ids,
                    spatial_causal=self.spatial_causal,
                    temporal_causal=self.temporal_causal,
                    sow_weights=self.sow_weights,
                    sow_activations=self.sow_activations,
                    decode=self.decode,
                    layer_index=layer_index,
                    time_every=self.time_every,
                    pos_emb_type=self.pos_emb_type,
                    max_len=self.max_len,
                )
            )


    def __call__(self, x_BTNM: jax.Array) -> jax.Array:
        x_BTNM = self.input_norm(x_BTNM)
        # Apply sinusoidal PE only if using that type (RoPE is applied inside attention)
        if self.pos_enc is not None:
            x_BTNM = self.pos_enc(x_BTNM)
        for block in self.blocks:
            x_BTNM = block(x_BTNM)

        if self.sow_logits:
            self.sow(nnx.Intermediate, "logits", x_BTNM)
        return x_BTNM


class TokenizerDreamer4(nnx.Module):
    """
    MAE in Dreamer 4.

    Dimension keys:
        B: batch size
        T: sequence length
        N: number of patches per frame
        L: latent dimension
        Nl: number of latent tokens
        M: model_dim
        H: height
        W: width
        C: number of channels
        P: patch token dimension (patch_size^2 * C)
        S: N + Nl
    """

    def __init__(
        self,
        in_dim: int,
        image_height: int,
        image_width: int,
        model_dim: int,
        mlp_ratio: int,
        latent_dim: int,
        num_latent_tokens: int,
        time_every: int,
        patch_size: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        max_mask_ratio: float,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        use_flash_attention: bool,
        rngs: nnx.Rngs,
        pos_emb_type: str = "rope",  # "sinusoidal", "rope", or "none"
    ):
        self.in_dim = in_dim
        self.image_height = image_height
        self.image_width = image_width
        self.model_dim = model_dim
        self.mlp_ratio = mlp_ratio
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
        self.time_every = time_every
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_mask_ratio = max_mask_ratio
        self.param_dtype = param_dtype
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.pos_emb_type = pos_emb_type
        dummy_patch_11NP = patchify(jnp.ones((1, 1, self.image_height, self.image_width, self.in_dim)), self.patch_size)
        N = dummy_patch_11NP.shape[2]
        segments = [
            (Modality.IMAGE, N),
        ]
        self.layout = TokenLayout(n_latents=self.num_latent_tokens, segments=tuple(segments))
        self.modality_ids = self.layout.modality_ids()

        self.encoder = ModalityAxialTransformer(
            self.model_dim,
            self.mlp_ratio,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            rngs=rngs,
            mode="encoder",
            modality_ids=self.modality_ids,
            time_every=self.time_every,
            pos_emb_type=self.pos_emb_type,
        )
        self.encoder_proj = nnx.Linear(
            in_features=self.model_dim,
            out_features=self.latent_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )

        self.decoder = ModalityAxialTransformer(
            self.model_dim,
            self.mlp_ratio,
            self.num_blocks,
            self.num_heads,
            self.dropout,
            self.param_dtype,
            self.dtype,
            use_flash_attention=self.use_flash_attention,
            spatial_causal=False,
            temporal_causal=True,
            rngs=rngs,
            mode="decoder",
            modality_ids=self.modality_ids,
            time_every=self.time_every,
            pos_emb_type=self.pos_emb_type,
        )
        self.out_dim = self.in_dim * self.patch_size**2
        self.decoder_proj = nnx.Linear(
            in_features=self.model_dim,
            out_features=self.out_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.patch_proj = nnx.Linear(
            in_features=self.in_dim * self.patch_size**2,
            out_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.latent_proj = nnx.Linear(
            in_features=self.latent_dim,
            out_features=self.model_dim,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
            rngs=rngs,
        )
        self.mask_patch = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(), (1, 1, 1, self.model_dim)
            )
        )
        self.latent_tokens = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(), (self.num_latent_tokens, self.model_dim)
            )
        )
        self.query_tokens = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(), (N, self.model_dim)
            )
        )

    def __call__(
        self, batch: Dict[str, jax.Array], training: bool = True
    ) -> Dict[str, jax.Array]:
        H, W = batch["videos"].shape[2:4]
        videos_BTHWC = batch["videos"]
        outputs = self.mask_and_encode(videos_BTHWC, batch["rng"], training)
        z_BTNlL = outputs["z"]
        recon_BTHWC = self.decode(z_BTNlL, (H, W))
        outputs["recon"] = recon_BTHWC
        return outputs

    def mask_and_encode(
        self, videos: jax.Array, rng: jax.Array, training: bool = True
    ) -> Dict[str, jax.Array]:
        # --- Preprocess videos ---
        B, T = videos.shape[:2]
        patch_BTNP = patchify(videos, self.patch_size)
        N = patch_BTNP.shape[2]
        patch_BTNM = self.patch_proj(patch_BTNP)

        # --- Randomly mask patches ---
        if training:
            _rng_prob, _rng_mask = jax.random.split(rng, 2)
            mask_prob = jax.random.uniform(
                _rng_prob, shape=(B * T,), minval=0, maxval=self.max_mask_ratio
            )
            mask = jax.vmap(
                lambda rng, prob: jax.random.bernoulli(rng, prob, (N,)),
                in_axes=(0, 0),
            )(jax.random.split(_rng_mask, B * T), mask_prob)
            mask_BTN = mask.reshape(B, T, N)
            patch_BTNM = jnp.where(
                mask_BTN[..., jnp.newaxis], self.mask_patch.value, patch_BTNM
            )

        # --- Encode ---
        latent_tokens_repeated = einops.repeat(self.latent_tokens.value, "Nl M -> B T Nl M", B=B, T=T)
        patch_BTSM = jnp.concatenate([latent_tokens_repeated, patch_BTNM], axis=2)
        z_BTSM = self.encoder(patch_BTSM)
        z_BTNlM = z_BTSM[:, :, :self.num_latent_tokens]
        z_BTNlL = self.encoder_proj(z_BTNlM)
        # squeeze latents through tanh as described in Dreamer 4 section 3.1
        z_BTNlL = nnx.tanh(z_BTNlL)
        outputs = dict(z=z_BTNlL)
        return outputs

    def decode(self, z_BTNlL: jax.Array, video_hw: Tuple[int, int]) -> jax.Array:
        B, T = z_BTNlL.shape[:2]
        z_BTNlM = self.latent_proj(z_BTNlL)
        query_tokens_repeated = einops.repeat(self.query_tokens.value, "N M -> B T N M", B=B, T=T)
        z_BTSM = jnp.concatenate([z_BTNlM, query_tokens_repeated], axis=2)
        recon_BTSM = self.decoder(z_BTSM)
        recon_BTNM = recon_BTSM[:, :, self.num_latent_tokens:]
        recon_BTNP = self.decoder_proj(recon_BTNM)
        recon_BTNP = recon_BTNP.astype(jnp.float32)
        recon_BTNP = nnx.sigmoid(recon_BTNP)
        recon_BTNP = recon_BTNP.astype(self.dtype)
        return unpatchify(recon_BTNP, self.patch_size, *video_hw)



# =============================================================================
# Action Mapping for Minecraft (Dreamer4 compatible)
# =============================================================================

class MinecraftButtons:
    """Minecraft button definitions matching VPT format."""
    ATTACK = "attack"
    BACK = "back"
    FORWARD = "forward"
    JUMP = "jump"
    LEFT = "left"
    RIGHT = "right"
    SNEAK = "sneak"
    SPRINT = "sprint"
    USE = "use"
    DROP = "drop"
    INVENTORY = "inventory"

    ALL = [
        ATTACK, BACK, FORWARD, JUMP, LEFT, RIGHT,
        SNEAK, SPRINT, USE, DROP, INVENTORY,
    ] + [f"hotbar.{i}" for i in range(1, 10)]
    
    # Mapping from raw keyboard key names to button names
    KEY_TO_BUTTON = {
        "key.keyboard.escape" :"ESC",
        "key.keyboard.s" :"back",
        "key.keyboard.q" :"drop",
        "key.keyboard.w" :"forward",
        "key.keyboard.1" :"hotbar.1",
        "key.keyboard.2" :"hotbar.2",
        "key.keyboard.3" :"hotbar.3",
        "key.keyboard.4" :"hotbar.4",
        "key.keyboard.5" :"hotbar.5",
        "key.keyboard.6" :"hotbar.6",
        "key.keyboard.7" :"hotbar.7",
        "key.keyboard.8" :"hotbar.8",
        "key.keyboard.9" :"hotbar.9",
        "key.keyboard.e" :"inventory",
        "key.keyboard.space" :"jump",
        "key.keyboard.a" :"left",
        "key.keyboard.d" :"right",
        "key.keyboard.left.shift" :"sneak",
        "key.keyboard.left.control" :"sprint",
        "key.keyboard.f" :"swapHands",
    }
    
    # Mouse button mapping: 0 = left click (attack), 1 = right click (use)
    # Note: pickItem (middle click, button 2) exists in MineRL but NOT in VPT model
    MOUSE_BUTTON_TO_BUTTON = {
        0: "attack",
        1: "use",
        # 2: "pickItem",  # Not in Buttons.ALL
    }


class CameraQuantizer:
    """Quantizes continuous camera values to discrete bins with optional mu-law encoding."""
    
    def __init__(
        self,
        camera_maxval: int = 10,
        camera_binsize: int = 2,
        use_mu_law: bool = False,
        mu: float = 5.0,
    ):
        self.camera_maxval = camera_maxval
        self.camera_binsize = camera_binsize
        self.use_mu_law = use_mu_law
        self.mu = mu
        self.n_bins = (2 * camera_maxval // camera_binsize) + 1
        self.null_bin = camera_maxval // camera_binsize
    
    def discretize(self, xy: np.ndarray) -> np.ndarray:
        """Continuous camera [-maxval, maxval] → discrete bins [0, n_bins)."""
        xy = np.clip(xy, -self.camera_maxval, self.camera_maxval)
        if self.use_mu_law:
            xy = xy / self.camera_maxval
            xy = np.sign(xy) * (np.log(1.0 + self.mu * np.abs(xy)) / np.log(1.0 + self.mu))
            xy = xy * self.camera_maxval
        return np.round((xy + self.camera_maxval) / self.camera_binsize).astype(np.int32)
    
    def undiscretize(self, xy: np.ndarray) -> np.ndarray:
        """Discrete bins [0, n_bins) → continuous camera values."""
        xy = xy.astype(np.float32) * self.camera_binsize - self.camera_maxval
        if self.use_mu_law:
            xy = xy / self.camera_maxval
            xy = np.sign(xy) * (1.0 / self.mu) * ((1.0 + self.mu) ** np.abs(xy) - 1.0)
            xy = xy * self.camera_maxval
        return xy


class CameraHierarchicalActionMapping:
    """
    Hierarchical action mapping for Minecraft, compatible with Dreamer4.
    
    Converts between:
    - Factored actions: {"buttons": (B, 20), "camera": (B, 2)} 
    - Discrete actions: single integer index
    
    Button groups are made mutually exclusive, and camera has a meta-action
    (on/off) integrated into the button space. When camera meta-action is "on",
    a separate camera index determines the actual camera movement.
    
    This creates a joint discrete action space suitable for categorical policies.
    """
    
    # Mutually exclusive button groups
    BUTTONS_GROUPS = OrderedDict(
        hotbar=["none"] + [f"hotbar.{i}" for i in range(1, 10)],
        fore_back=["none", "forward", "back"],
        left_right=["none", "left", "right"],
        sprint_sneak=["none", "sprint", "sneak"],
        use=["none", "use"],
        drop=["none", "drop"],
        attack=["none", "attack"],
        jump=["none", "jump"],
        camera=["none", "camera"],  # meta-action for camera on/off
    )
    
    def __init__(
        self,
        n_camera_bins: int = 11,
        camera_maxval: int = 10,
        camera_binsize: int = 2,
        use_mu_law: bool = False,
        camera_mu: float = 5.0,
    ):
        """
        Args:
            n_camera_bins: Number of discrete bins per camera axis (should be odd).
            camera_maxval: Maximum camera value for quantization.
            camera_binsize: Bin size for camera quantization.
            use_mu_law: Whether to use mu-law encoding for camera.
            camera_mu: Mu parameter for mu-law encoding.
        """
        assert n_camera_bins % 2 == 1, "n_camera_bins should be odd"
        self.n_camera_bins = n_camera_bins
        self.camera_null_bin = n_camera_bins // 2
        
        self.camera_quantizer = CameraQuantizer(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            use_mu_law=use_mu_law,
            mu=camera_mu,
        )
        
        # Build button combinations (all products of button groups + special "inventory")
        self.BUTTONS_COMBINATIONS = list(itertools.product(*self.BUTTONS_GROUPS.values())) + ["inventory"]
        self.BUTTONS_COMBINATION_TO_IDX = {comb: i for i, comb in enumerate(self.BUTTONS_COMBINATIONS)}
        self.BUTTONS_IDX_TO_COMBINATION = {i: comb for i, comb in enumerate(self.BUTTONS_COMBINATIONS)}
        
        # Build camera combinations (x, y joint)
        self.camera_groups = OrderedDict(
            camera_x=[f"camera_x{i}" for i in range(self.n_camera_bins)],
            camera_y=[f"camera_y{i}" for i in range(self.n_camera_bins)],
        )
        self.camera_combinations = list(itertools.product(*self.camera_groups.values()))
        self.camera_combination_to_idx = {comb: i for i, comb in enumerate(self.camera_combinations)}
        self.camera_idx_to_combination = {i: comb for i, comb in enumerate(self.camera_combinations)}
        self.camera_null_idx = self.camera_combination_to_idx[
            (f"camera_x{self.camera_null_bin}", f"camera_y{self.camera_null_bin}")
        ]
        
        # Precompute lookup tables for fast conversion
        self._precompute_lookup_tables()
        
    def _precompute_lookup_tables(self):
        """Precompute joint action → factored action matrices."""
        button_dim = len(MinecraftButtons.ALL)
        
        # Button index → factored buttons
        self.BUTTON_IDX_TO_FACTORED = np.zeros(
            (len(self.BUTTONS_IDX_TO_COMBINATION), button_dim), dtype=np.int32
        )
        self.BUTTON_IDX_TO_CAMERA_META_OFF = np.zeros(
            (len(self.BUTTONS_IDX_TO_COMBINATION),), dtype=bool
        )
        
        for jnt_ac, button_comb in self.BUTTONS_IDX_TO_COMBINATION.items():
            new_button_ac = np.zeros(button_dim, dtype=np.int32)
            if button_comb == "inventory":
                new_button_ac[MinecraftButtons.ALL.index("inventory")] = 1
            else:
                for group_choice in button_comb[:-1]:  # Last one is camera meta
                    if group_choice != "none":
                        new_button_ac[MinecraftButtons.ALL.index(group_choice)] = 1
                if button_comb[-1] != "camera":  # Camera meta action is off
                    self.BUTTON_IDX_TO_CAMERA_META_OFF[jnt_ac] = True
            self.BUTTON_IDX_TO_FACTORED[jnt_ac] = new_button_ac
        
        # Camera index → factored camera
        self.CAMERA_IDX_TO_FACTORED = np.zeros(
            (len(self.camera_idx_to_combination), 2), dtype=np.int32
        )
        for jnt_ac, camera_comb in self.camera_idx_to_combination.items():
            self.CAMERA_IDX_TO_FACTORED[jnt_ac, 0] = self.camera_groups["camera_x"].index(camera_comb[0])
            self.CAMERA_IDX_TO_FACTORED[jnt_ac, 1] = self.camera_groups["camera_y"].index(camera_comb[1])
    
    @property
    def n_buttons(self) -> int:
        """Number of discrete button actions (including camera meta-action)."""
        return len(self.BUTTONS_COMBINATIONS)
    
    @property
    def n_camera(self) -> int:
        """Number of discrete camera actions."""
        return len(self.camera_combinations)
    
    @property
    def n_actions(self) -> int:
        """Total number of discrete actions (buttons * camera, for flat action space)."""
        return self.n_buttons * self.n_camera
    
    def _factored_buttons_to_groups(self, ac_buttons: np.ndarray, button_group: list) -> list:
        """
        For a mutually exclusive group of buttons, find which option was chosen.
        
        Args:
            ac_buttons: Button actions (B, len(MinecraftButtons.ALL))
            button_group: List of buttons in a mutually exclusive group, starting with 'none'
        
        Returns:
            List of length B with the chosen option from button_group.
        """
        assert button_group[0] == "none"
        group_indices = [MinecraftButtons.ALL.index(b) for b in button_group if b != "none"]
        ac_choices = ac_buttons[:, group_indices]
        
        # Handle mutual press (forward+back or left+right) → neither
        if "forward" in button_group and "back" in button_group:
            ac_choices[np.all(ac_choices, axis=-1)] = 0
        if "left" in button_group and "right" in button_group:
            ac_choices[np.all(ac_choices, axis=-1)] = 0
        
        ac_non_zero = np.where(ac_choices)
        ac_choice = ["none"] * ac_buttons.shape[0]
        for index, action in zip(ac_non_zero[0], ac_non_zero[1]):
            ac_choice[index] = button_group[action + 1]
        return ac_choice
    
    def from_factored(self, ac: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert factored actions to hierarchical discrete space.
        
        Args:
            ac: {"buttons": (B, 20), "camera": (B, 2)} 
                - buttons: binary, one-hot per button
                - camera: discrete bin indices [0, n_camera_bins)
        
        Returns:
            {"buttons": (B, 1), "camera": (B, 1)} - discrete indices
        """
        assert ac["camera"].ndim == 2 and ac["buttons"].ndim == 2
        
        # Get button choices for each group (except camera meta)
        choices_by_group = OrderedDict(
            (k, self._factored_buttons_to_groups(ac["buttons"], v))
            for k, v in self.BUTTONS_GROUPS.items() if k != "camera"
        )
        
        # Set camera meta-action based on whether camera moved
        camera_is_null = np.all(ac["camera"] == self.camera_null_bin, axis=1)
        choices_by_group["camera"] = ["none" if is_null else "camera" for is_null in camera_is_null]
        
        new_button_ac = []
        new_camera_ac = []
        
        for i in range(ac["buttons"].shape[0]):
            # Buttons
            key = tuple([v[i] for v in choices_by_group.values()])
            if ac["buttons"][i, MinecraftButtons.ALL.index("inventory")] == 1:
                key = "inventory"
            new_button_ac.append(self.BUTTONS_COMBINATION_TO_IDX[key])
            
            # Camera (inventory is exclusive with camera)
            if key == "inventory":
                cam_key = (f"camera_x{self.camera_null_bin}", f"camera_y{self.camera_null_bin}")
            else:
                cam_key = (f"camera_x{ac['camera'][i, 0]}", f"camera_y{ac['camera'][i, 1]}")
            new_camera_ac.append(self.camera_combination_to_idx[cam_key])
        
        return {
            "buttons": np.array(new_button_ac, dtype=np.int32)[:, None],
            "camera": np.array(new_camera_ac, dtype=np.int32)[:, None],
        }
    
    def to_factored(self, ac: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert hierarchical discrete actions to factored space.
        
        Args:
            ac: {"buttons": (B, 1), "camera": (B, 1)} - discrete indices
        
        Returns:
            {"buttons": (B, 20), "camera": (B, 2)} - factored actions
        """
        assert ac["camera"].shape[-1] == 1 and ac["buttons"].shape[-1] == 1
        
        button_idx = np.squeeze(ac["buttons"], -1)
        camera_idx = np.squeeze(ac["camera"], -1)
        
        new_button_ac = self.BUTTON_IDX_TO_FACTORED[button_idx]
        camera_off = self.BUTTON_IDX_TO_CAMERA_META_OFF[button_idx]
        new_camera_ac = self.CAMERA_IDX_TO_FACTORED[camera_idx].copy()
        new_camera_ac[camera_off] = self.camera_null_bin
        
        return {"buttons": new_button_ac, "camera": new_camera_ac}
    
    def to_flat_index(self, ac: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert hierarchical discrete actions to single flat index.
        
        Args:
            ac: {"buttons": (B, 1), "camera": (B, 1)} - discrete indices
        
        Returns:
            (B,) flat action indices
        """
        button_idx = np.squeeze(ac["buttons"], -1)
        camera_idx = np.squeeze(ac["camera"], -1)
        return button_idx * self.n_camera + camera_idx
    
    def from_flat_index(self, flat_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert flat action index to hierarchical discrete actions.
        
        Args:
            flat_idx: (B,) flat action indices
        
        Returns:
            {"buttons": (B, 1), "camera": (B, 1)} - discrete indices
        """
        button_idx = flat_idx // self.n_camera
        camera_idx = flat_idx % self.n_camera
        return {
            "buttons": button_idx[:, None],
            "camera": camera_idx[:, None],
        }
    
    def factored_to_flat_index(self, ac: Dict[str, np.ndarray]) -> np.ndarray:
        """Convenience: factored → flat index in one call."""
        hierarchical = self.from_factored(ac)
        return self.to_flat_index(hierarchical)
    
    def flat_index_to_factored(self, flat_idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience: flat index → factored in one call."""
        hierarchical = self.from_flat_index(flat_idx)
        return self.to_factored(hierarchical)
    
    def get_null_action(self) -> Dict[str, np.ndarray]:
        """Return the null/no-op action."""
        null_button = self.BUTTONS_COMBINATION_TO_IDX[
            tuple("none" for _ in range(len(self.BUTTONS_GROUPS)))
        ]
        return {
            "buttons": np.array([[null_button]], dtype=np.int32),
            "camera": np.array([[self.camera_null_idx]], dtype=np.int32),
        }
    
    def get_null_flat_index(self) -> int:
        """Return the null/no-op action as flat index."""
        null_ac = self.get_null_action()
        return int(self.to_flat_index(null_ac)[0])
    
    # JAX-compatible versions for inference
    def to_factored_jax(self, ac: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """JAX-compatible version of to_factored for inference."""
        button_idx = jnp.squeeze(ac["buttons"], -1)
        camera_idx = jnp.squeeze(ac["camera"], -1)
        
        # Use precomputed lookup tables (converted to jax arrays)
        button_lut = jnp.array(self.BUTTON_IDX_TO_FACTORED)
        camera_lut = jnp.array(self.CAMERA_IDX_TO_FACTORED)
        camera_off_lut = jnp.array(self.BUTTON_IDX_TO_CAMERA_META_OFF)
        
        new_button_ac = button_lut[button_idx]
        camera_off = camera_off_lut[button_idx]
        new_camera_ac = camera_lut[camera_idx]
        new_camera_ac = jnp.where(
            camera_off[:, None],
            jnp.full_like(new_camera_ac, self.camera_null_bin),
            new_camera_ac
        )
        
        return {"buttons": new_button_ac, "camera": new_camera_ac}
    
    def flat_index_to_factored_jax(self, flat_idx: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """JAX-compatible: flat index → factored."""
        button_idx = flat_idx // self.n_camera
        camera_idx = flat_idx % self.n_camera
        hierarchical = {
            "buttons": button_idx[:, None],
            "camera": camera_idx[:, None],
        }
        return self.to_factored_jax(hierarchical)
    
    # =========================================================================
    # Raw JSONL data parsing methods
    # =========================================================================
    
    def parse_raw_action(self, raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Parse a single raw action from JSONL format to factored format.
        
        Args:
            raw: Raw action dict from JSONL, e.g.:
                {
                    "mouse": {"dx": 37.0, "dy": -49.0, "buttons": [0], ...},
                    "keyboard": {"keys": ["key.keyboard.w", "key.keyboard.space"], ...},
                    ...
                }
        
        Returns:
            Factored action dict: {"buttons": (20,), "camera": (2,)}
        """
        # Parse buttons (keyboard + mouse)
        buttons = np.zeros(len(MinecraftButtons.ALL), dtype=np.int32)
        
        # Keyboard keys
        for key in raw.get("keyboard", {}).get("keys", []):
            if key in MinecraftButtons.KEY_TO_BUTTON:
                button_name = MinecraftButtons.KEY_TO_BUTTON[key]
                if button_name in MinecraftButtons.ALL:
                    buttons[MinecraftButtons.ALL.index(button_name)] = 1
        
        # Mouse buttons
        for mb in raw.get("mouse", {}).get("buttons", []):
            if mb in MinecraftButtons.MOUSE_BUTTON_TO_BUTTON:
                button_name = MinecraftButtons.MOUSE_BUTTON_TO_BUTTON[mb]
                if button_name in MinecraftButtons.ALL:
                    buttons[MinecraftButtons.ALL.index(button_name)] = 1
        
        # Parse camera (mouse dx, dy)
        dx = raw.get("mouse", {}).get("dx", 0.0)
        dy = raw.get("mouse", {}).get("dy", 0.0)
        camera_continuous = np.array([dx, dy], dtype=np.float32)
        camera_discrete = self.camera_quantizer.discretize(camera_continuous)
        
        return {
            "buttons": buttons,
            "camera": camera_discrete,
        }
    
    def parse_raw_actions_batch(self, raw_list: list) -> Dict[str, np.ndarray]:
        """
        Parse a batch of raw actions from JSONL format.
        
        Args:
            raw_list: List of raw action dicts from JSONL
        
        Returns:
            Batched factored actions: {"buttons": (B, 20), "camera": (B, 2)}
        """
        parsed = [self.parse_raw_action(r) for r in raw_list]
        return {
            "buttons": np.stack([p["buttons"] for p in parsed], axis=0),
            "camera": np.stack([p["camera"] for p in parsed], axis=0),
        }
    
    def raw_to_discrete_index(self, raw: Dict[str, Any]) -> int:
        """
        Parse raw action and convert to single discrete index.
        
        Args:
            raw: Raw action dict from JSONL
        
        Returns:
            Single discrete action index
        """
        factored = self.parse_raw_action(raw)
        factored_batched = {
            "buttons": factored["buttons"][None, :],
            "camera": factored["camera"][None, :],
        }
        hierarchical = self.from_factored(factored_batched)
        return int(self.to_flat_index(hierarchical)[0])
    
    def raw_batch_to_discrete_indices(self, raw_list: list) -> np.ndarray:
        """
        Parse batch of raw actions and convert to discrete indices.
        
        Args:
            raw_list: List of raw action dicts from JSONL
        
        Returns:
            (B,) array of discrete action indices
        """
        factored = self.parse_raw_actions_batch(raw_list)
        hierarchical = self.from_factored(factored)
        return self.to_flat_index(hierarchical)
    
    def raw_batch_to_hierarchical(self, raw_list: list) -> Dict[str, np.ndarray]:
        """
        Parse batch of raw actions and convert to hierarchical discrete space.
        
        Combines parse_raw_actions_batch + from_factored in one call.
        
        Args:
            raw_list: List of raw action dicts from JSONL
        
        Returns:
            {"buttons": (B, 1), "camera": (B, 1)} - hierarchical discrete indices
        """
        factored = self.parse_raw_actions_batch(raw_list)
        return self.from_factored(factored)
    
    def _factored_to_raw_single(self, buttons: np.ndarray, camera: np.ndarray) -> Dict[str, Any]:
        """
        Convert a single factored action to raw format.
        
        Args:
            buttons: (20,) binary button array
            camera: (2,) discrete camera bins
        
        Returns:
            Raw action dict for environment interaction
        """
        # Reverse mappings
        button_to_key = {v: k for k, v in MinecraftButtons.KEY_TO_BUTTON.items()}
        button_to_mouse = {v: k for k, v in MinecraftButtons.MOUSE_BUTTON_TO_BUTTON.items()}
        
        keys = []
        mouse_buttons = []
        
        for i, pressed in enumerate(buttons):
            if pressed:
                button_name = MinecraftButtons.ALL[i]
                if button_name in button_to_key:
                    keys.append(button_to_key[button_name])
                elif button_name in button_to_mouse:
                    mouse_buttons.append(button_to_mouse[button_name])
        
        # Convert camera bins to continuous values
        camera_continuous = self.camera_quantizer.undiscretize(camera.astype(np.float32))
        
        return {
            "keyboard": {"keys": keys},
            "mouse": {
                "dx": float(camera_continuous[0]),
                "dy": float(camera_continuous[1]),
                "buttons": mouse_buttons,
            },
        }
    
    def hierarchical_to_raw(self, ac: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Convert a single hierarchical action to raw format.
        
        Args:
            ac: {"buttons": (1,), "camera": (1,)} - single hierarchical action
        
        Returns:
            Raw action dict for environment interaction
        """
        # Ensure batch dimension
        if ac["buttons"].ndim == 0:
            ac = {"buttons": ac["buttons"][None], "camera": ac["camera"][None]}
        
        factored = self.to_factored(ac)
        return self._factored_to_raw_single(factored["buttons"][0], factored["camera"][0])
    
    def hierarchical_batch_to_raw(self, ac: Dict[str, np.ndarray]) -> list[Dict[str, Any]]:
        """
        Convert batch of hierarchical actions to raw format.
        
        Inverse of raw_batch_to_hierarchical.
        
        Args:
            ac: {"buttons": (B, 1), "camera": (B, 1)} - hierarchical discrete indices
        
        Returns:
            List of B raw action dicts for environment interaction
        """
        factored = self.to_factored(ac)
        batch_size = factored["buttons"].shape[0]
        
        return [
            self._factored_to_raw_single(factored["buttons"][i], factored["camera"][i])
            for i in range(batch_size)
        ]
    
    def discrete_index_to_raw(self, idx: int) -> Dict[str, Any]:
        """
        Convert discrete index back to raw-like format (for environment interaction).
        
        Args:
            idx: Discrete action index
        
        Returns:
            Dict with "keyboard" keys and "mouse" info for environment
        """
        flat_idx = np.array([idx])
        factored = self.flat_index_to_factored(flat_idx)
        return self._factored_to_raw_single(factored["buttons"][0], factored["camera"][0])


class ActionEncoder(nnx.Module):
    def __init__(
        self,
        d_model: int,
        n_keyboard: int,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        shift_time_by_one: bool,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.n_keyboard = n_keyboard
        self.shift_time_by_one = shift_time_by_one
        # Base "action token" embedding (used always)
        self.base_action_emb = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (self.d_model,))
        )
        # Embedding for categorical actions
        self.emb_key = nnx.Embed(self.n_keyboard, self.d_model, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        actions: Optional[jnp.ndarray] = None,  # (B, T) int32 in [0, n_keyboard)
        batch_time_shape: Optional[Tuple[int, int]] = None,
        as_tokens: bool = True,
    ):
        if actions is None:
            # unlabeled videos: just broadcast base embedding
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = jnp.broadcast_to(self.base_action_emb.value, (B, T, self.d_model))
        else:
            # embed categorical actions
            emb_key = self.emb_key(actions)
            if self.shift_time_by_one:
                emb_key = emb_key[:, :-1, :]
                emb_key = jnp.pad(emb_key, ((0, 0), (1, 0), (0, 0)))
            out = emb_key + self.base_action_emb.value  # broadcast add

        if as_tokens:
            # expand a token axis (S_a = 1)
            out = out[:, :, None, :]

        return out


class HierarchicalActionEncoder(nnx.Module):
    """
    Hierarchical action encoder for Minecraft.

    Expects actions of shape (B, T, 2) where the last dimension is
    [button_idx, camera_idx].  Button and camera tokens are embedded with
    separate tables and summed together with a shared base embedding,
    matching the interface of ActionEncoder so that DynamicsDreamer4 can
    swap between the two transparently.
    """

    def __init__(
        self,
        d_model: int,
        n_buttons: int,
        n_camera: int,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        shift_time_by_one: bool,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.shift_time_by_one = shift_time_by_one
        self.base_action_emb = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (d_model,))
        )
        self.emb_buttons = nnx.Embed(n_buttons, d_model, param_dtype=param_dtype, dtype=dtype, rngs=rngs)
        self.emb_camera = nnx.Embed(n_camera, d_model, param_dtype=param_dtype, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        actions: Optional[jnp.ndarray] = None,  # (B, T, 2): [button_idx, camera_idx]
        batch_time_shape: Optional[Tuple[int, int]] = None,
        as_tokens: bool = True,
    ):
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = jnp.broadcast_to(self.base_action_emb.value, (B, T, self.d_model))
        else:
            button_idx = actions[:, :, 0]  # (B, T)
            camera_idx = actions[:, :, 1]  # (B, T)
            emb_button = self.emb_buttons(button_idx)  # (B, T, d_model)
            emb_cam = self.emb_camera(camera_idx)      # (B, T, d_model)
            if self.shift_time_by_one:
                emb_button = jnp.pad(emb_button[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
                emb_cam = jnp.pad(emb_cam[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
            out = emb_button + emb_cam + self.base_action_emb.value

        if as_tokens:
            out = out[:, :, None, :]
        return out


class DynamicsDreamer4(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        n_actions: int,
        depth: int,
        k_max: int,
        rngs: nnx.Rngs,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        time_every: int = 4,
        space_mode: str = "wm_agent_isolated",
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
        use_flash_attention: bool = True,
        shift_action_tokens_by_one: bool = False,
        pos_emb_type: str = "sinusoidal",  # "sinusoidal", "rope", or "none"
        n_camera: Optional[int] = None,    # If set, use HierarchicalActionEncoder (Minecraft)
    ):
        """
        Pretrain script: instantiate with space_mode="wm_agent_isolated" and pass agent_tokens=None (dummy).
        Fine-tune script: instantiate with space_mode="wm_agent" and pass real agent_tokens from task embedding.
        Args:
          packed_enc_tokens:      (B, T, n_spatial, d_spatial) packed encoder tokens
          actions:    (B, T, N_a, D_a) raw action tokens
          steps:      (B, T) bfloat16 — step sizes, 1/2^x
          signals:    (B, T) float32 - signal values, grid that is reachable by current step size

        Shapes produced:
          spatial_tokens: (B, T, n_spatial, d_model)
          action_tokens:  (B, T, 1, d_model)  # if your ActionEncoder emits one token
          signal_token:   (B, T, 1, d_model)
          step_token:     (B, T, 1, d_model)
        """
        self.d_model = d_model
        self.d_spatial = d_spatial
        self.n_register = n_register
        self.n_agent = n_agent
        self.k_max = k_max
        self.space_mode = space_mode
        self.n_spatial = n_spatial
        self.time_every = time_every
        self.pos_emb_type = pos_emb_type
        self.spatial_proj = nnx.Linear(d_spatial, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        if n_camera is not None:
            self.action_encoder = HierarchicalActionEncoder(
                d_model=d_model, n_buttons=n_actions, n_camera=n_camera,
                dtype=dtype, param_dtype=param_dtype,
                shift_time_by_one=shift_action_tokens_by_one, rngs=rngs,
            )
        else:
            self.action_encoder = ActionEncoder(
                d_model=d_model, n_keyboard=n_actions, dtype=dtype, param_dtype=param_dtype,
                rngs=rngs, shift_time_by_one=shift_action_tokens_by_one,
            )
        self.register_tokens = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (n_register, d_model))
        )

        segments = [
            (Modality.ACTION, 1),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1),
            (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, n_register),
        ]
        if n_agent > 0:
            segments.append((Modality.AGENT, n_agent))
        
        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        self.spatial_slice = self.layout.slices()[Modality.SPATIAL]
        self.agent_slice = self.layout.slices().get(Modality.AGENT, slice(0, 0))
        self.modality_ids = self.layout.modality_ids()

        self.transformer = ModalityAxialTransformer(
            model_dim=d_model,
            mlp_ratio=mlp_ratio,
            num_blocks=depth,
            num_heads=n_heads,
            dropout=dropout,
            dtype=dtype,
            param_dtype=param_dtype,
            use_flash_attention=use_flash_attention,
            rngs=rngs,
            mode=space_mode,
            modality_ids=self.modality_ids,
            spatial_causal=False,
            temporal_causal=True,
            time_every=time_every,
            pos_emb_type=self.pos_emb_type,
        )

        # -------- Discrete embeddings for shortcut conditioning --------
        # Step size d ∈ {1, 1/2, 1/4, ..., 1/256}
        # We index steps by: step_idx = log2(1/d) ∈ {0, 1, 2, ...,7, 8}
        self.num_step_bins = int(jnp.log2(k_max)) + 1
        self.step_embed = nnx.Embed(self.num_step_bins, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        # Signal level τ ∈ {0, 1/d, 2/d, ..., 1 - 1/d} (grid length = 1/d)
        # We use a *shared* table with  bins and only use the first (1/d) entries for a given d.
        self.signal_embed = nnx.Embed(k_max + 1, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.flow_x_head = nnx.Linear(
            d_model,
            d_spatial,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )


    def __call__(
        self,
        actions: jnp.ndarray,            # (B,T) - or whatever expected shape
        step_idxs: jnp.ndarray,          # (B,T) - time step d idxs
        signal_idxs: jnp.ndarray,        # (B,T) - step size τ idxs
        packed_enc_tokens: jnp.ndarray,  # (B,T,n_s,d_spatial)
        *,
        agent_tokens: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ):
        spatial_tokens = self.spatial_proj(packed_enc_tokens) # (B, T, n_spatial, d_model)

        action_tokens = self.action_encoder(actions)

        B, T = spatial_tokens.shape[:2]
        register_tokens = jnp.broadcast_to(
            self.register_tokens.value[None, None, ...], # (1,1,n_register,d_model)
            (B, T, self.n_register, self.d_model),
        )

        step_tok = self.step_embed(step_idxs)[:, :, None, :]       # (B, T, 1, d_model)
        signal_tok = self.signal_embed(signal_idxs)[:, :, None, :] # (B, T, 1, d_model)

        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = jnp.zeros((B, T, self.n_agent, self.d_model), dtype=spatial_tokens.dtype)
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens, agent_tokens]
        else:
            toks = [action_tokens, signal_tok, step_tok, spatial_tokens, register_tokens]
        
        tokens = jnp.concatenate(toks, axis=2) # (B,T,S,D)
        
        x = self.transformer(tokens)

        spatial_tokens_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_tokens_out)
        
        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]
            
        return x1_hat, h_t


class TaskEmbedder(nnx.Module):
    def __init__(
        self, d_model: int, 
        dtype: jnp.dtype, param_dtype: jnp.dtype, rngs: nnx.Rngs,
        n_agent: int = 1, use_ids: bool = True, n_tasks: int = 128, d_task: int = 64,
    ):
        self.d_model = d_model
        self.n_agent = n_agent
        self.use_ids = use_ids # True: task is int ids; False: task is vector
        self.n_tasks = n_tasks # only used if use_ids=True
        self.d_task = d_task   # only used if use_ids=False

        if self.use_ids:
            self.task_table = nnx.Embed(n_tasks, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:
            self.task_proj = nnx.Linear(d_task, d_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.base_agent_emb = nnx.Param(nnx.initializers.normal(0.02)(rngs.params(), (d_model,)))

    def __call__(self, task, B: int, T: int):
        """
        If use_ids=True:
            task: (B,) int32 ids in [0, n_tasks)
        Else:
            task: (B, d_task) float32 vector

        Returns agent tokens: (B, T, n_agent, d_model)
        """
        if self.use_ids:
            emb = self.task_table(task)  # (B, D)
        else:
            emb = self.task_proj(task)   # (B, D)

        # Learned base + optional small MLP to decouple from raw table
        x = emb + self.base_agent_emb[None, :]

        # Replicate across time and agent slots
        x = jnp.broadcast_to(x[:, None, None, :], (B, T, self.n_agent, self.d_model))
        return x

# === Phase B heads (use existing MLP) =========================================

class PolicyHeadMTP(nnx.Module):
    def __init__(
        self, d_model: int, 
        param_dtype: jnp.dtype, dtype: jnp.dtype, rngs: nnx.Rngs,
        action_dim: int, L: int = 8, kind: str = "categorical", mlp_ratio: float = 2.0, 
    ):
        self.d_model = d_model
        self.action_dim = action_dim
        self.L = L
        self.kind = kind
        self.mlp_ratio = mlp_ratio

        self.projector = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            self.d_model,
            self.L * self.action_dim,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, h_t: jnp.ndarray) -> jnp.ndarray:
        B, T = h_t.shape[:2]
        x = self.projector(h_t)  # (B, T, D)
        logits = self.out(x)     # (B, T, L * A)
        logits = logits.reshape(B, T, self.L, self.action_dim)   # (B, T, L, A)
        return logits  # softmax/sigmoid applied at loss-time based on `kind`


class RewardHeadMTP(nnx.Module):
    """Multi-Token reward prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, L, K), centers_log (K,)
    """

    def __init__(
        self,
        d_model: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        L: int = 8,
        num_bins: int = 101,
        mlp_ratio: float = 2.0,
        log_low: float = -8.0,
        log_high: float = 8.0,
    ):
        self.d_model = d_model
        self.L = L
        self.num_bins = num_bins
        self.mlp_ratio = mlp_ratio

        self.projector = MLP(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            d_model,
            L * num_bins,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        # Precompute bin centers as a constant (uniform in log-space)
        self.centers_log = jnp.linspace(log_low, log_high, num_bins)

    def __call__(self, h_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t)          # (B, T, D)
        logits = self.out(x)             # (B, T, L * K)
        B, T = h_t.shape[:2]
        logits = logits.reshape(B, T, self.L, self.num_bins)  # (B, T, L, K)
        return logits, self.centers_log


class ValueHead(nnx.Module):
    """Value prediction with symexp twohot bins.
    Input:  h_t (B, T, D)
    Output: logits (B, T, K), centers_log (K,)
    """

    def __init__(
        self,
        d_model: int,
        param_dtype: jnp.dtype,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        num_bins: int = 101,
        mlp_ratio: float = 2.0,
        log_low: float = -8.0,
        log_high: float = 8.0,
    ):
        self.d_model = d_model
        self.num_bins = num_bins
        self.mlp_ratio = mlp_ratio

        self.projector = MLP(
            d_model=d_model,
            mlp_ratio=mlp_ratio,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            d_model,
            num_bins,
            param_dtype=param_dtype,
            dtype=dtype,
            rngs=rngs,
        )
        # Precompute bin centers as a constant (uniform in log-space)
        self.centers_log = jnp.linspace(log_low, log_high, num_bins)

    def __call__(self, h_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.projector(h_t)   # (B, T, D)
        logits = self.out(x)      # (B, T, K)
        return logits, self.centers_log


def restore_dreamer4_tokenizer(
    sharding: jax.sharding.NamedSharding,
    rng: jax.Array,
    args,
) -> "TokenizerDreamer4":
    """Restore pre-trained Dreamer4 tokenizer from checkpoint.
    
    The tokenizer checkpoint was saved as a ModelAndOptimizer state (containing
    both model and optimizer states), so we need to restore in that format and
    extract just the model part.
    
    Args:
        sharding: Sharding specification for the restored parameters.
        rng: Random key for initializing the dummy tokenizer.
        args: Arguments containing tokenizer configuration and checkpoint path.
            Expected args attributes:
            - tokenizer_checkpoint: path to the tokenizer checkpoint
            - image_channels, image_height, image_width
            - tokenizer_d_model, mlp_ratio, d_latent, n_latent
            - time_every, patch_size, tokenizer_n_block, tokenizer_n_head
            - dropout, param_dtype, dtype, use_flash_attention
    
    Returns:
        Restored TokenizerDreamer4 model with pre-trained weights.
    """
    from jasmine.models.dreamer4_models import TokenizerDreamer4
    
    rngs = nnx.Rngs(rng)
    
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
    
    checkpoint_options = ocp.CheckpointManagerOptions(
        step_format_fixed_length=6,
    )
    tokenizer_checkpoint_manager = ocp.CheckpointManager(
        directory=args.tokenizer_checkpoint,
        options=checkpoint_options,
        handler_registry=handler_registry,
    )
    
    # Create dummy tokenizer to get the state structure
    dummy_tokenizer = TokenizerDreamer4(
        in_dim=args.image_channels,
        image_height=args.image_height,
        image_width=args.image_width,
        model_dim=args.tokenizer_d_model,
        mlp_ratio=args.mlp_ratio,
        latent_dim=args.d_latent,
        num_latent_tokens=args.n_latent,
        time_every=args.time_every,
        patch_size=args.patch_size,
        num_blocks=args.tokenizer_n_block,
        num_heads=args.tokenizer_n_head,
        dropout=args.dropout,
        max_mask_ratio=0.0,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        rngs=rngs,
        pos_emb_type=getattr(args, 'pos_emb_type', 'sinusoidal'),
    )

    # Wrap in ModelAndOptimizer to match checkpoint format
    # The checkpoint was saved as nnx.state(optimizer) where optimizer is ModelAndOptimizer
    tx = optax.adamw(learning_rate=1e-4)  # Dummy optimizer, we only need its structure
    dummy_optimizer = nnx.ModelAndOptimizer(dummy_tokenizer, tx)
    dummy_optimizer_state = nnx.state(dummy_optimizer)
    abstract_sharded_optimizer_state = _create_abstract_sharded_pytree(
        dummy_optimizer_state, sharding
    )
    
    # Build per-leaf restore args that explicitly specify the target sharding.
    # Using ocp.ArrayRestoreArgs (rather than relying on ShapeDtypeStruct.sharding)
    # is necessary when restoring a checkpoint saved on a different number of devices.
    restore_args_tree = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=sharding),
        dummy_optimizer_state,
    )

    # Restore the checkpoint
    restored_optimizer_state = tokenizer_checkpoint_manager.restore(
        step=tokenizer_checkpoint_manager.latest_step(),
        args=ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(  # type: ignore
                abstract_sharded_optimizer_state,
                partial_restore=True,
                restore_args=restore_args_tree,
            ),
        ),
    )["model_state"]
    
    # Update the dummy optimizer with restored weights, then extract the model
    nnx.update(dummy_optimizer, restored_optimizer_state)
    tokenizer = dummy_optimizer.model
    
    tokenizer_checkpoint_manager.close()
    print(f"Restored Dreamer4 tokenizer from {args.tokenizer_checkpoint}")
    
    return tokenizer

def _create_abstract_sharded_pytree(
    pytree_template: nnx.GraphState, sharding_spec: jax.sharding.NamedSharding
) -> jax.Array:
    """Replaces arrays in a pytree with ShapeDtypeStructs having the given sharding."""

    def map_fn(leaf_template):
        if hasattr(leaf_template, "shape") and hasattr(leaf_template, "dtype"):
            return jax.ShapeDtypeStruct(
                leaf_template.shape, leaf_template.dtype, sharding=sharding_spec
            )
        return leaf_template

    return jax.tree_util.tree_map(map_fn, pytree_template)



if __name__ == "__main__":
#     import treescope
#     x = jnp.ones((2, 10, 4+16, 64))
#     model = ModalityAxialTransformer(
#         model_dim=64,
#         mlp_ratio=4,
#         out_dim=7,
#         num_blocks=8,
#         num_heads=8,
#         dropout=0.0,
#         param_dtype=jnp.float32,
#         dtype=jnp.bfloat16,
#         use_flash_attention=True,
#         rngs=nnx.Rngs(0),
#         mode="decoder",
#         modality_ids=jnp.array([Modality.LATENT]*4 + [Modality.IMAGE]*16),
#         spatial_causal=False,
#         temporal_causal=True,
#         decode=False,
#         sow_weights=False,
#         sow_activations=False,
#         sow_logits=False,
#         max_len=5000,
#         time_every=4,
#     )
#     y = model(x)
#     print(y.shape)
#     html_content = treescope.render_to_html(model)
#     with open("model_view.html", "w", encoding="utf-8") as f:
#         f.write(html_content)
#     import pdb; pdb.set_trace()




# if __name__ == "__main__":
#     x = {'videos': jnp.ones((2, 10, 24, 36, 3)), 'rng': jax.random.PRNGKey(0)}
#     import treescope
#     model = TokenizerDreamer4(
#         in_dim=3,
#         image_height=24,
#         image_width=36,
#         model_dim=64,
#         mlp_ratio=4,
#         latent_dim=32,
#         num_latent_tokens=4,
#         patch_size=4,
#         num_blocks=8,
#         num_heads=8,
#         dropout=0.0,
#         max_mask_ratio=0.5,
#         param_dtype=jnp.float32,
#         dtype=jnp.bfloat16,
#         use_flash_attention=True,
#         rngs=nnx.Rngs(0),
#     )
#     y = model(x, training=True)
#     html_content = treescope.render_to_html(model)
#     with open("model_view.html", "w", encoding="utf-8") as f:
#         f.write(html_content)
#     import pdb; pdb.set_trace()


    # rngs = nnx.Rngs(0)
    # actions = jnp.zeros((2, 10), dtype=jnp.int32)
    # step_idxs = jnp.zeros((2, 10), dtype=jnp.int32)
    # signal_idxs = jnp.zeros((2, 10), dtype=jnp.int32)
    # packed_enc_tokens = jnp.zeros((2, 10, 8, 32))
    # model = DynamicsDreamer4(
    #     d_model=64,
    #     d_spatial=32,
    #     n_spatial=8,
    #     n_register=4,
    #     n_agent=0,
    #     n_heads=4,
    #     n_actions=15,
    #     depth=4,
    #     k_max=16,
    #     rngs=rngs,
    #     dropout=0.0,
    #     mlp_ratio=4,
    #     time_every=4,
    #     space_mode="wm_agent_isolated",
    #     dtype=jnp.bfloat16,
    #     param_dtype=jnp.float32,
    # )
    # x1_hat, h_t = model(actions, step_idxs, signal_idxs, packed_enc_tokens)
    # print(x1_hat.shape)
    # print(h_t.shape)


    import json
    import pdb
    # 초기화
    action_mapper = CameraHierarchicalActionMapping(
        n_camera_bins=11,
        camera_maxval=10,
        camera_binsize=2,
    )

    # JSONL 파일에서 액션 로드
    with open("/home/4bkang/rl/jasmine/data/open_ai_minecraft_actions_files/6.0_ilge-60b420bb3851-20210505-230344.jsonl") as f:
        raw_actions = [json.loads(line) for line in f]

    # 학습 데이터로 변환
    factored_actions = action_mapper.parse_raw_actions_batch(raw_actions)
    hierarchical_actions = action_mapper.from_factored(factored_actions)
    # → (B,) int32 array, ActionEncoder에 바로 사용 가능

    pdb.set_trace()
    # 단일 액션 변환
    raw = {
        "mouse": {"dx": 3.0, "dy": -4.0, "buttons": [0]},
        "keyboard": {"keys": ["key.keyboard.w", "key.keyboard.space"]},
    }
    idx = action_mapper.parse_raw_action(raw)  # → int


    # 환경 상호작용: discrete index → raw format
    raw_out = action_mapper.from_factored(idx)
    # → {"keyboard": {"keys": [...]}, "mouse": {"dx": ..., "dy": ..., "buttons": [...]}}
    print(raw_out)
    pdb.set_trace()

