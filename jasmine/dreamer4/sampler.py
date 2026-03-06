from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Literal, Tuple, Optional, Dict, Any, Callable
from einops import reduce

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from jasmine.models.dreamer4_models import TokenizerDreamer4, DynamicsDreamer4, TaskEmbedder
from jasmine.utils.dreamer4_utils import pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck


@partial(nnx.jit, static_argnames=('deterministic',))
def _jit_call_dynamics(
    dynamics: DynamicsDreamer4,
    actions, step_idxs, signal_idxs, packed_enc_tokens,
    agent_tokens=None,
    deterministic: bool = True,
):
    """JIT-compiled wrapper for DynamicsDreamer4.__call__.
    Compiles one forward pass into a single XLA program for efficient dispatch.
    """
    return dynamics(actions, step_idxs, signal_idxs, packed_enc_tokens,
                    agent_tokens=agent_tokens, deterministic=deterministic)

# ---------------------------
# Config & small utilities
# ---------------------------

StartMode   = Literal["pure", "fixed", "random"]
Schedule    = Literal["finest", "shortcut"]
RolloutMode = Literal["teacher_forced", "autoregressive"]

@dataclass
class SamplerConfig:
    dyna_k_max: int
    schedule: Schedule                      # "finest" or "shortcut"
    d: Optional[float] = None               # used iff schedule == "shortcut"
    start_mode: StartMode = "pure"          # in TF: {"pure","fixed","random"}; in AR: must be "pure"
    tau0_fixed: float = 0.5                 # used iff start_mode == "fixed"
    rollout: RolloutMode = "autoregressive" # "teacher_forced" or "autoregressive"
    horizon: int = 1
    ctx_length: int = 32
    ctx_noise_tau: Optional[float] = 0.9  # τ of context fed to dynamics; None/1.0=clean, 0.9=10% noise

    rng_key: Optional[jax.Array] = None
    mae_eval_key: Optional[jax.Array] = None
    # decoding sizes
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    patch_size: int = 16
    # tokenizer shapes
    dyna_n_spatial: int = 4
    dyna_packing_factor: int = 2
    # debugging (host-side only)
    debug: bool = False
    # Called with a dict. We call it twice: once with a high-level run "plan" (kind="plan"),
    # and once per step for the scalar fields (kind="step").
    debug_hook: Optional[Callable[[dict], None]] = None


def _assert_power_of_two(k: int):
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"k_max must be a positive power of two, got {k}")

def _is_power_of_two_fraction(x: float) -> bool:
    if x <= 0 or x > 1: return False
    inv = round(1.0 / x)
    return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

def _step_idx_from_d(d: float, k_max: int) -> int:
    K = round(1.0 / float(d))
    if abs(1.0 / K - d) > 1e-8:
        raise ValueError(f"d={d} is not an exact 1/(power of two)")
    e = int(round(np.log2(K)))
    emax = int(round(np.log2(k_max)))
    if e > emax:
        raise ValueError(f"step bin e={e} (d={d}) is coarser than allowed emax={emax} (k_max={k_max})")
    return e

def _choose_step_size(k_max: int, schedule: Schedule, d: Optional[float]) -> float:
    _assert_power_of_two(k_max)
    if schedule == "finest":
        return 1.0 / float(k_max)
    if d is None:
        raise ValueError("schedule='shortcut' requires config.d (e.g., 1/4)")
    if not _is_power_of_two_fraction(d):
        raise ValueError(f"d must be 1/(power of two); got d={d}")
    if d < 1.0 / float(k_max):
        raise ValueError(f"d={d} is finer than d_min=1/{k_max}")
    return float(d)

def _align_to_grid(tau0: float, d: float) -> float:
    # Snap tau0 to the {0, d, 2d, ...} grid. No-op if already aligned.
    return float(np.clip(np.floor(tau0 / d) * d, 0.0, 1.0))

def _signal_idx_from_tau(tau: jnp.ndarray, k_max: int) -> jnp.ndarray:
    idx = (tau * k_max).astype(jnp.int32)
    return jnp.clip(idx, 0, k_max - 1)

def _validate_modes(cfg: SamplerConfig):
    # AR ⇒ start must be "pure"
    if cfg.rollout == "autoregressive" and cfg.start_mode != "pure":
        raise ValueError("Autoregressive rollout supports only start_mode='pure'. "
                         "Use teacher_forced for fixed/random starts.")
    # Finest vs shortcut d usage
    if cfg.schedule == "finest" and cfg.d is not None:
        raise ValueError("Provide d only for schedule='shortcut'.")
    if cfg.schedule == "shortcut" and cfg.d is None:
        raise ValueError("schedule='shortcut' requires a valid d (e.g., 1/4).")

def _tau_grid_from(k_max: int, schedule: Schedule, d_opt: Optional[float], start_tau: float) -> Tuple[jnp.ndarray, int, float, int]:
    """
    if start_tau is 2/4 and d_opt = 1/4, 
    tau_seq = [2/4, 3/4, 4/4]
    S = 2 (len(tau_seq)-1)
    d = 1/4
    e = 2 (log2(1/d))
    """
    d = _choose_step_size(k_max, schedule, d_opt)
    # Use numpy for S computation to keep it concrete (not traced by JAX)
    start_aligned = float(np.clip(np.floor(start_tau / d) * d, 0.0, 1.0))
    S_float = (1.0 - start_aligned) / d
    S = int(round(S_float))  # Python int, not traced
    tau_seq = start_aligned + d * np.arange(S + 1, dtype=np.float32)
    tau_seq = np.clip(tau_seq, 0.0, 1.0)
    e = _step_idx_from_d(float(d), k_max) # e = log2(1/d)
    return tau_seq, S, float(d), e



# ---------- Plan builder (host-only) ----------

def _build_run_plan(cfg: SamplerConfig) -> dict:
    d_used = _choose_step_size(cfg.dyna_k_max, cfg.schedule, cfg.d)
    e = _step_idx_from_d(d_used, cfg.dyna_k_max)
    d_inv = int(round(1.0 / d_used))
    plan = {
        "kind": "plan",
        "k_max": cfg.dyna_k_max,
        "schedule": cfg.schedule,
        "d": d_used,
        "K": d_inv,
        "e": e,
        "rollout": cfg.rollout,
        "start_mode": cfg.start_mode,
        "ctx_length": cfg.ctx_length,
        "horizon": cfg.horizon,
    }
    if cfg.rollout == "autoregressive":
        plan["tau0_policy"] = "pure (0.0)"
        plan["S"] = int(1.0 / d_used)
        plan["S_range"] = (plan["S"], plan["S"])
        plan["tau0_grid"] = [0.0]
    else:
        if cfg.start_mode == "pure":
            plan["tau0_policy"] = "pure (0.0)"
            plan["S"] = int(1.0 / d_used)
            plan["S_range"] = (plan["S"], plan["S"])
            plan["tau0_grid"] = [0.0]
        elif cfg.start_mode == "fixed":
            tau0a = _align_to_grid(float(np.clip(cfg.tau0_fixed, 0.0, 1.0)), d_used)
            plan["tau0_policy"] = f"fixed(aligned={tau0a:.6f})"
            plan["S"] = int(round((1.0 - tau0a) / d_used))
            plan["S_range"] = (plan["S"], plan["S"])
            grid = [i * d_used for i in range(0, d_inv)]
            plan["tau0_grid"] = grid
        else:  # random
            plan["tau0_policy"] = f"random on grid {{0, d, ..., 1-d}}"
            plan["S_range"] = (1, int(1.0 / d_used))
            plan["tau0_grid"] = ["0", "d", "...", "1-d"] if d_inv > 16 else [i * d_used for i in range(0, d_inv)]
    return plan

def _emit_plan(plan: dict, hook: Optional[Callable[[dict], None]], enable_print: bool):
    if hook:
        hook(plan)
    if enable_print:
        keys = ["kind","k_max","schedule","d","K","e","rollout","start_mode",
                "ctx_length","horizon","tau0_policy","S","S_range","tau0_grid"]
        msg = {k: plan[k] for k in keys if k in plan}
        print(f"[sampler] {msg}")

# ---------------------------
# Core Single-Frame Sampler
# ---------------------------

def denoise_single_latent(
    *,
    dynamics: DynamicsDreamer4,
    actions_ctx: jnp.ndarray,     # (B, T_ctx) — original (unshifted) context actions
    z_ctx_clean: jnp.ndarray,     # (B, T_ctx, n_spatial, D_s) clean context
    z_t_init: jnp.ndarray,        # (B, 1, n_spatial, D_s) initial latent at tau0
    k_max: int,
    d: float,
    start_mode: StartMode,
    tau0_fixed: float,
    rng_key: jax.Array,
    clean_target_next: Optional[jnp.ndarray] = None,
    agent_tokens: jnp.ndarray = None,
    ctx_noise_tau: Optional[float] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Denoise a single future latent using a τ-ladder.

    actions_ctx holds the original (unshifted) context actions:
    actions_ctx[:, t] = a_t (action executed AT frame t, leading to frame t+1).
    The action for the future frame is actions_ctx[:, -1] = a_{T_ctx-1}.

    When dynamics.decode=True (inference model with KV cache):
      - The cache must already be filled with context frames by the caller.
      - All tau steps write to the same cache slot (no auto-increment).
        After the tau ladder completes, advance_cache() is called once.

    When dynamics.decode=False (training model):
      - Full-sequence forward pass each step.
      - Context is fed at ctx_noise_tau level (clean if None/1.0).
    """
    B, T_ctx, n_spatial, D_s = z_ctx_clean.shape
    _assert_power_of_two(k_max)
    assert actions_ctx.shape[:2] == (B, T_ctx), f"Expected actions_ctx leading dims ({B}, {T_ctx}), got {actions_ctx.shape}"

    # 1) choose tau0
    rng_key, r_tau, r_noise = jax.random.split(rng_key, 3)
    if start_mode == "pure":
        tau0 = 0.0
    elif start_mode == "fixed":
        tau0 = float(np.clip(tau0_fixed, 0.0, 1.0))
    else:
        tau0 = float(jax.random.uniform(r_tau, (), minval=0.0, maxval=1.0))
    tau0_aligned = _align_to_grid(tau0, d) if tau0 > 0.0 else 0.0

    # Pre-compute noisy context for full-sequence path when ctx_noise_tau < 1.0.
    # (When use_cache=True the cache path handles ctx_noise_tau separately below.)
    z_ctx_for_full = z_ctx_clean
    ctx_sig_for_full = k_max - 1  # clean context signal
    if ctx_noise_tau is not None and float(ctx_noise_tau) < 1.0:
        _tau_ctx = float(ctx_noise_tau)
        _noise_ctx = jax.random.normal(r_noise, z_ctx_clean.shape, dtype=z_ctx_clean.dtype)
        z_ctx_for_full = _tau_ctx * z_ctx_clean + (1.0 - _tau_ctx) * _noise_ctx
        ctx_sig_for_full = int(np.clip(int(_tau_ctx * k_max), 0, k_max - 1))

    # 2) init latent
    z_t = z_t_init

    # 3) tau ladder
    schedule = "shortcut" if d > (1.0 / k_max) else "finest"
    tau_seq, S, d_used, e = _tau_grid_from(k_max, schedule, d, tau0_aligned)
    tau_seq_host = list(np.asarray(tau_seq))

    h_t_final = None

    use_cache = dynamics.decode

    # 4) tau ladder steps
    for s in range(1, len(tau_seq_host)):
        tau_prev = float(tau_seq_host[s - 1])
        tau_curr = float(tau_seq_host[s])
        alpha = (tau_curr - tau_prev) / max(1.0 - tau_prev, 1e-8)

        signal_idx_future = jnp.full((B, 1), _signal_idx_from_tau(jnp.asarray(tau_prev), k_max), dtype=jnp.int32)

        if use_cache:
            # All tau steps write to the same cache slot (no auto-increment in forward).
            # The future frame's action token = a_{T_ctx-1} = actions_ctx[:, -1].
            step_idx_fut = jnp.full((B, 1), e, dtype=jnp.int32)
            agent_fut = agent_tokens[:, T_ctx:T_ctx+1] if agent_tokens is not None else None
            z_clean_pred, h_seq = _jit_call_dynamics(
                dynamics, actions_ctx[:, -1:], step_idx_fut, signal_idx_future, z_t,
                agent_tokens=agent_fut,
            )
        else:
            # Full-sequence path (training model / decode=False).
            # Shift: prepend sentinel -1 so frame 0 gets base_action_emb,
            # frame t gets a_{t-1}, future frame T_ctx gets a_{T_ctx-1}.
            # Context is fed at ctx_noise_tau level (precomputed as z_ctx_for_full).
            z_seq = jnp.concatenate([z_ctx_for_full, z_t], axis=1)
            _sent_val = float('nan') if jnp.issubdtype(actions_ctx.dtype, jnp.floating) else -1
            sentinel = jnp.full((B, 1) + actions_ctx.shape[2:], _sent_val, dtype=actions_ctx.dtype)
            actions_full = jnp.concatenate([sentinel, actions_ctx], axis=1)  # (B, T_ctx+1[, ...])
            step_idx = jnp.full((B, T_ctx + 1), e, dtype=jnp.int32)
            signal_idx_ctx = jnp.full((B, T_ctx), ctx_sig_for_full, dtype=jnp.int32)
            signal_idx = jnp.concatenate([signal_idx_ctx, signal_idx_future], axis=1)
            z_clean_pred_seq, h_seq = _jit_call_dynamics(
                dynamics, actions_full, step_idx, signal_idx, z_seq,
                agent_tokens=agent_tokens,
            )
            z_clean_pred = z_clean_pred_seq[:, -1:, :, :]

        if s == len(tau_seq_host) - 1 and h_seq is not None:
            h_t_final = h_seq[:, -1:, :, :]

        z_t = (1.0 - alpha) * z_t + alpha * z_clean_pred

    if use_cache:
        # The loop's last iteration wrote KV(z_t at τ_{S-1}, signal=k_max-2),
        # but z_t is now at τ=1.0 (fully denoised). Overwrite the cache slot
        # once more so the next frame attends to the correct representation.
        # Apply ctx_noise_tau (if set) for consistency with context pre-fill:
        # generated frames are stored with the same noise level as context frames.
        z_for_cache = z_t
        sig_val_final = k_max - 1
        if ctx_noise_tau is not None and float(ctx_noise_tau) < 1.0:
            _tau = float(ctx_noise_tau)
            _noise = jax.random.normal(rng_key, z_t.shape, dtype=z_t.dtype)
            z_for_cache = _tau * z_t + (1.0 - _tau) * _noise
            sig_val_final = int(np.clip(int(_tau * k_max), 0, k_max - 1))
        sig_idx_final = jnp.full((B, 1), sig_val_final, dtype=jnp.int32)
        agent_fut = agent_tokens[:, T_ctx:T_ctx+1] if agent_tokens is not None else None
        _, h_seq_final = _jit_call_dynamics(
            dynamics, actions_ctx[:, -1:],
            jnp.full((B, 1), e, dtype=jnp.int32),
            sig_idx_final, z_for_cache,
            agent_tokens=agent_fut,
        )
        if h_seq_final is not None:
            h_t_final = h_seq_final[:, -1:, :, :]
        # Advance the cache index so the next frame writes to the next slot.
        dynamics.advance_cache()

    return z_t, h_t_final

# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    tokenizer: TokenizerDreamer4,
    dynamics: DynamicsDreamer4,
    frames: jnp.ndarray,     # (B,T,H,W,C)
    actions: jnp.ndarray,    # (B,T)
    config: SamplerConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B, T, H, W, C = frames.shape
    assert config.ctx_length < T, "ctx_length must be < T"
    _validate_modes(config)

    # --- One-shot plan print before any work ---
    plan = _build_run_plan(config)
    _emit_plan(plan, config.debug_hook, config.debug)

    horizon = config.horizon
    rng = config.rng_key if config.rng_key is not None else jax.random.PRNGKey(0)
    mae_key = config.mae_eval_key if config.mae_eval_key is not None else jax.random.PRNGKey(777)

    # 1) encode once (deterministic key)
    z_btLd = tokenizer.mask_and_encode(frames, rng=None, training=False)['z']
    z_all = pack_bottleneck_to_spatial(z_btLd, n_spatial=config.dyna_n_spatial, k=config.dyna_packing_factor)  # (B,T,n_spatial,D_s)

    # 2) split context vs future
    z_ctx_clean = z_all[:, :config.ctx_length, :, :]
    actions_ctx = actions[:, :config.ctx_length]
    future_actions = actions[:, config.ctx_length: config.ctx_length + horizon]
    gt_future_latents = z_all[:, config.ctx_length: config.ctx_length + horizon, :, :]

    # 3) floor: decoder recon of (ctx + GT future)
    floor_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_ctx_clean, n_spatial=config.dyna_n_spatial, k=config.dyna_packing_factor),
        unpack_spatial_to_bottleneck(gt_future_latents, n_spatial=config.dyna_n_spatial, k=config.dyna_packing_factor)
    ], axis=1)
    floor_frames = tokenizer.decode(floor_btLd, (H, W))

    # 4) choose schedule/step size
    d = _choose_step_size(config.dyna_k_max, config.schedule, config.d)
    e = _step_idx_from_d(d, config.dyna_k_max)

    # 5) rollout
    preds: list[jnp.ndarray] = []
    n_spatial, D_s = int(z_all.shape[2]), int(z_all.shape[3])

    # Pre-fill inference cache with context frames (only when dynamics.decode=True)
    if dynamics.decode:
        z_ctx_for_encode = z_ctx_clean
        ctx_sig_val = config.dyna_k_max - 1
        if config.ctx_noise_tau is not None and float(config.ctx_noise_tau) < 1.0:
            rng, ctx_noise_key = jax.random.split(rng)
            _tau = float(config.ctx_noise_tau)
            _noise = jax.random.normal(ctx_noise_key, z_ctx_clean.shape, dtype=z_ctx_clean.dtype)
            z_ctx_for_encode = _tau * z_ctx_clean + (1.0 - _tau) * _noise
            ctx_sig_val = int(np.clip(int(_tau * config.dyna_k_max), 0, config.dyna_k_max - 1))

        # Allocate cache: context + full horizon
        dynamics.init_cache(B, config.ctx_length + horizon)
        step_idx_1 = jnp.full((B, 1), e, dtype=jnp.int32)
        sig_idx_1  = jnp.full((B, 1), ctx_sig_val, dtype=jnp.int32)

        # Encode context frame-by-frame.
        # Shift: frame t gets a_{t-1}; frame 0 gets sentinel -1.
        sentinel = jnp.full((B, 1) + actions_ctx.shape[2:], -1, dtype=actions_ctx.dtype)
        shifted_ctx = jnp.concatenate([sentinel, actions_ctx[:, :-1]], axis=1)  # (B, T_ctx)
        for t_ctx in range(config.ctx_length):
            _jit_call_dynamics(dynamics, shifted_ctx[:, t_ctx:t_ctx+1], step_idx_1, sig_idx_1,
                               z_ctx_for_encode[:, t_ctx:t_ctx+1])
            dynamics.advance_cache()

    for t in range(horizon):
        action_curr = future_actions[:, t:t+1]
        z1_ref = gt_future_latents[:, t:t+1, :, :] if config.rollout == "teacher_forced" else None

        rng, z0key = jax.random.split(rng)
        z0 = jax.random.normal(z0key, (B, 1, n_spatial, D_s), dtype=z_all.dtype)
        z_t_init = z0

        rng, step_key = jax.random.split(rng)
        z_clean_pred, _ = denoise_single_latent(
            dynamics=dynamics,
            actions_ctx=actions_ctx,
            z_ctx_clean=z_ctx_clean,
            z_t_init=z_t_init,
            k_max=config.dyna_k_max,
            d=d,
            start_mode=config.start_mode,
            tau0_fixed=config.tau0_fixed,
            rng_key=step_key,
            clean_target_next=z1_ref,
            ctx_noise_tau=config.ctx_noise_tau,
        )
        preds.append(z_clean_pred)

        # Advance context arrays (used by the full-sequence path and for bookkeeping)
        z_to_add = z_clean_pred if config.rollout == "autoregressive" else z1_ref
        z_ctx_clean = jnp.concatenate([z_ctx_clean, z_to_add], axis=1)[:, -config.ctx_length:, :, :]
        actions_ctx = jnp.concatenate([actions_ctx, action_curr], axis=1)[:, -config.ctx_length:]

    # 6) decode predictions (prepend context for viz)
    pred_latents = jnp.concatenate(preds, axis=1)
    pred_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_all[:, :config.ctx_length, :, :], n_spatial=config.dyna_n_spatial, k=config.dyna_packing_factor),
        unpack_spatial_to_bottleneck(pred_latents, n_spatial=config.dyna_n_spatial, k=config.dyna_packing_factor),
    ], axis=1)
    pred_frames = tokenizer.decode(pred_btLd, (H, W))

    gt_frames = frames[:, :config.ctx_length + horizon]
    return pred_frames, floor_frames, gt_frames

# ---------------------------
# Imagination rollouts for RL training
# ---------------------------

# @partial(
#     jax.jit,
#     static_argnames=("dynamics", "task_embedder", "policy_head", "k_max", "horizon", "context_length", 
#                     "n_spatial", "d", "start_mode"),
# )
def imagine_rollouts(
    *,
    dynamics: DynamicsDreamer4,
    task_embedder: TaskEmbedder,
    policy_head: PolicyHeadMTP,
    z_context: jnp.ndarray,  # (B, context_length, n_spatial, d_spatial)
    context_actions: jnp.ndarray,  # (B, context_length)
    task_ids: jnp.ndarray,  # (B,) task IDs for task embedder
    k_max: int,
    horizon: int,
    context_length: int,
    n_spatial: int,
    d: float,  # step size for denoising schedule
    start_mode: StartMode = "pure",
    tau0_fixed: float = 0.5,
    rng_key: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate imagined rollouts using dynamics and policy head.
    
    Similar to sample_video but queries the policy head to get actions instead of
    using pre-specified actions.
    
    Args:
        dynamics: DynamicsDreamer4 model (NNX module)
        task_embedder: TaskEmbedder for agent tokens (NNX module)
        policy_head: PolicyHeadMTP for action prediction (NNX module)
        z_context: (B, context_length, n_spatial, d_spatial) encoded context latents
        context_actions: (B, context_length) context actions
        task_ids: (B,) task IDs for task embedder
        k_max: Maximum k for denoising schedule
        horizon: Number of steps to imagine
        context_length: Length of context
        n_spatial: Number of spatial tokens
        d: Step size for denoising (e.g., 1/k_max for finest)
        start_mode: Start mode for denoising ("pure", "fixed", "random")
        tau0_fixed: Fixed tau0 if start_mode == "fixed"
        rng_key: PRNG key
        
    Returns:
        imagined_latents: (B, horizon + 1, n_spatial, d_spatial)
            - index 0 is the last context state (s_ctx_last)
            - indices 1..horizon are imagined future states
        imagined_actions: (B, horizon)
            - actions[t] takes you from imagined_latents[:, t] → imagined_latents[:, t+1]
        imagined_hidden_states: (B, horizon + 1, d_model)
            - hidden state aligned with imagined_latents
    """
    B = z_context.shape[0]
    D_s = z_context.shape[3]
    d_model = policy_head.d_model
    
    # Initialize context
    z_ctx_clean = z_context  # (B, context_length, n_spatial, d_spatial)
    actions_ctx = context_actions  # (B, context_length)
    
    # Pre-compute agent tokens for entire rollout (task doesn't change)
    agent_tokens_full = task_embedder(
        task_ids, B, context_length + horizon
    )  # (B, context_length + horizon, n_agent, d_model)
    
    # Compute step index from denoising step size d
    # This ensures context uses the same step size as the imagination schedule
    e = _step_idx_from_d(d, k_max)
    e_jax = jnp.int32(e)
    
    # Prepare step and signal indices for initial context dynamics call
    step_idx_ctx = jnp.full((B, context_length), e_jax, dtype=jnp.int32)
    signal_idx_ctx = jnp.full((B, context_length), k_max - 1, dtype=jnp.int32)  # tau=1.0
    
    # Get initial hidden state from context (before loop starts)
    _, h_ctx_init = dynamics(
        actions_ctx,
        step_idx_ctx,
        signal_idx_ctx,
        z_ctx_clean,
        agent_tokens=agent_tokens_full[:, :context_length, :, :],
        deterministic=True,
    )  # h_ctx_init: (B, context_length, n_agent, d_model)
    h_pooled_init = reduce(h_ctx_init, 'b t n_agent d_model -> b t d_model', 'mean')  # (B, context_length, d_model)
    h = h_pooled_init[:, -1, :]  # (B, d_model) - use last timestep

    # Pre-allocate output arrays (include starting state as index 0)
    imagined_latents = jnp.zeros((B, horizon + 1, n_spatial, D_s), dtype=z_context.dtype)
    imagined_actions = jnp.zeros((B, horizon), dtype=jnp.int32)
    imagined_hidden_states = jnp.zeros((B, horizon + 1, d_model), dtype=z_context.dtype)

    # Set starting state (last context latent and its hidden state) at index 0
    z_start = z_ctx_clean[:, -1, :, :]  # (B, n_spatial, D_s)
    imagined_latents = imagined_latents.at[:, 0, :, :].set(z_start)
    imagined_hidden_states = imagined_hidden_states.at[:, 0, :].set(h)
    
    rng = rng_key
    
    for t in range(horizon):
        # Use current hidden state h to predict next action
        h_for_policy = h[:, None, :]  # (B, 1, d_model)
        # Query policy head to get action logits
        pi_logits = policy_head(h_for_policy)  # (B, 1, L, A)
        
        # Sample action from the first predicted action (index 0 in L dimension)
        rng, action_key = jax.random.split(rng)
        logp = jax.nn.log_softmax(pi_logits[:, 0, 0, :], axis=-1)  # (B, A)
        action_curr = jax.random.categorical(action_key, logp, axis=-1)  # (B,)
        action_curr = action_curr[:, None]  # (B, 1)
        
        # Use denoise_single_latent to get next latent prediction
        rng, z0key = jax.random.split(rng)
        z0 = jax.random.normal(z0key, (B, 1, n_spatial, D_s), dtype=z_ctx_clean.dtype)
        z_t_init = z0
        
        rng, step_key = jax.random.split(rng)
        
        # Slice agent tokens for current sequence length: [0, context_length + t + 1]
        # This includes context + all imagined steps up to t, plus one more for the new timestep
        agent_tokens_seq = agent_tokens_full[:, t:context_length + t + 1, :, :]
        
        z_clean_pred, h_t_pred = denoise_single_latent(
            dynamics=dynamics,
            actions_ctx=actions_ctx,
            action_curr=action_curr,
            z_ctx_clean=z_ctx_clean,
            z_t_init=z_t_init,
            k_max=k_max,
            d=d,
            start_mode=start_mode,
            tau0_fixed=tau0_fixed,
            rng_key=step_key,
            agent_tokens=agent_tokens_seq,
        )  # z_clean_pred: (B, 1, n_spatial, d_spatial), h_t_pred: (B, 1, n_agent, d_model) or None
        
        # Pool hidden state from denoising step for next iteration
        h_pooled_pred = reduce(h_t_pred, 'b t n_agent d_model -> b t d_model', 'mean')  # (B, 1, d_model)
        h_next = h_pooled_pred[:, 0, :]  # (B, d_model)
        
        # Store results in pre-allocated arrays
        # Latents/hidden are shifted by +1 because index 0 holds the starting context state
        imagined_latents = imagined_latents.at[:, t + 1, :, :].set(z_clean_pred[:, 0, :, :])
        imagined_actions = imagined_actions.at[:, t].set(action_curr[:, 0])
        imagined_hidden_states = imagined_hidden_states.at[:, t + 1, :].set(h_next)
        
        # Update context autoregressively
        z_ctx_clean = jnp.concatenate([z_ctx_clean, z_clean_pred], axis=1)[:, -context_length:, :, :]
        actions_ctx = jnp.concatenate([actions_ctx, action_curr], axis=1)[:, -context_length:]
        
        # Update h for next iteration (use h_next from denoising)
        h = h_next
    
    return imagined_latents, imagined_actions, imagined_hidden_states
