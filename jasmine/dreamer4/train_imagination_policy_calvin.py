"""Train CALVIN policy/value heads inside imagination with a frozen Dreamer4 world model."""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Optional

import flax.nnx as nnx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
import wandb
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jasmine.dreamer4.calvin_bc_validation import (
    CalvinEnvValidationState,
    close_calvin_env_validation,
    init_calvin_env_validation,
    run_calvin_env_validation,
)
from jasmine.dreamer4.sampler import imagine_rollouts_continuous, squash_calvin_actions
from jasmine.models.dreamer4_models import (
    DynamicsDreamer4,
    PolicyHeadContinuousMTP,
    RewardHeadMTP,
    TokenizerDreamer4,
    ValueHead,
    restore_dreamer4_tokenizer,
)
from jasmine.utils.calvin_dataloader import get_calvin_lang_dataloader
from jasmine.utils.dreamer4_utils import pack_bottleneck_to_spatial
from jasmine.utils.train_utils import count_parameters_by_component, get_lr_schedule


@dataclass
class Args:
    num_steps: int = 50_000
    seed: int = 0
    seq_len: int = 96
    batch_size: int = 32

    train_data_dirs: list[str] = field(
        default_factory=lambda: ["/home/4bkang/rl/calvin/dataset/task_ABCD_D/training"]
    )
    val_data_dirs: list[str] = field(
        default_factory=lambda: ["/home/4bkang/rl/calvin/dataset/task_ABCD_D/validation"]
    )
    lang_folder: str = "lang_all-mpnet-base-v2"
    reward_at_end: bool = True

    context_length: int = 16
    horizon: int = 32
    imagination_d: float = 0.25
    imagination_start_mode: str = "pure"
    imagination_tau0_fixed: float = 0.5
    ctx_noise_tau: Optional[float] = None

    gamma: float = 0.997
    lambda_: float = 0.95
    actor_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    bc_prior_weight: float = 0.3
    init_policy_from_bc: bool = True

    init_lr: float = 0.0
    max_lr: float = 3e-4
    decay_end: float = 0.0
    wsd_decay_steps: int = 10_000
    warmup_steps: int = 5_000
    lr_schedule: str = "wsd"
    weight_decay: float = 1e-4

    image_channels: int = 3
    image_height: int = 96
    image_width: int = 96
    time_every: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True
    patch_size: int = 16
    pos_emb_type: str = "rope"

    d_latent: int = 32
    n_latent: int = 32
    tokenizer_enc_model_dim: int = 512
    tokenizer_enc_mlp_ratio: int = 4
    tokenizer_enc_time_every: int = 3
    tokenizer_enc_n_block: int = 6
    tokenizer_enc_n_head: int = 8
    tokenizer_dec_model_dim: int = 512
    tokenizer_dec_mlp_ratio: int = 4
    tokenizer_dec_time_every: int = 2
    tokenizer_dec_n_block: int = 4
    tokenizer_dec_n_head: int = 8
    tokenizer_checkpoint: str = "ckpts/calvin/dreamer4/tokenizer_96p"

    dyna_d_model: int = 768
    dyna_packing_factor: int = 1
    dyna_d_spatial: int = 32
    dyna_n_spatial: int = 32
    dyna_n_register: int = 4
    dyna_n_agent: int = 1
    dyna_n_block: int = 8
    dyna_n_head: int = 12
    dyna_k_max: int = 128
    lang_emb_dim: int = 768

    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0
    reward_log_high: float = 3.0
    num_value_bins: int = 101
    value_log_low: float = -6.0
    value_log_high: float = 6.0

    pretrained_bc_rew_ckpt: str = "ckpts/calvin/dreamer4/bc_rew_heads_96p"
    pretrained_bc_rew_step: int = 30000

    save_ckpt: bool = True
    restore_ckpt: bool = False
    restore_step: int = 0
    ckpt_dir: str = "ckpts/calvin/dreamer4/imagination_policy_96p"
    log_checkpoint_interval: int = 1_000
    log_checkpoint_keep_period: int = 10_000

    val_interval: int = 5_000
    val_num_rollouts: int = 4
    val_num_videos: int = 4
    val_max_steps: int = 120
    val_context_len: Optional[int] = None
    val_show_gui: bool = False
    val_worker_timeout_sec: float = 120.0
    calvin_repo_root: str = field(default_factory=lambda: os.environ.get("CALVIN_ROOT", "/home/4bkang/rl/calvin"))
    calvin_python_bin: str = field(default_factory=lambda: os.environ.get("CALVIN_PYTHON_BIN", ""))

    log: bool = True
    entity: str = "4bkang"
    project: str = "jasmine"
    name: str = "imagination_policy_dreamer4_calvin_mpnet"
    tags: list[str] = field(default_factory=lambda: ["policy", "dreamer4", "calvin", "imagination"])
    log_interval: int = 50
    wandb_id: str = ""


class FrozenBCRewardModel(nnx.Module):
    def __init__(
        self,
        dynamics: DynamicsDreamer4,
        policy_head: PolicyHeadContinuousMTP,
        reward_head: RewardHeadMTP,
    ):
        self.dynamics = dynamics
        self.policy_head = policy_head
        self.reward_head = reward_head


class PolicyValueModel(nnx.Module):
    def __init__(
        self,
        policy_head: PolicyHeadContinuousMTP,
        value_head: ValueHead,
    ):
        self.policy_head = policy_head
        self.value_head = value_head


class ValidationPolicyWrapper:
    def __init__(self, dynamics: DynamicsDreamer4, policy_head: PolicyHeadContinuousMTP):
        self.dynamics = dynamics
        self.policy_head = policy_head


def _symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


def _twohot_symlog_targets(values: jnp.ndarray, centers_log: jnp.ndarray) -> jnp.ndarray:
    y = _symlog(values)
    num_bins = centers_log.shape[0]

    idx_r = jnp.searchsorted(centers_log, y, side="right")
    idx_l = jnp.maximum(idx_r - 1, 0)
    idx_r = jnp.minimum(idx_r, num_bins - 1)
    idx_l = jnp.minimum(idx_l, num_bins - 1)

    c_l = jnp.take(centers_log, idx_l)
    c_r = jnp.take(centers_log, idx_r)
    denom = jnp.maximum(c_r - c_l, 1e-8)
    frac = jnp.where(idx_r == idx_l, 0.0, (y - c_l) / denom)

    oh_l = jax.nn.one_hot(idx_l, num_bins)
    oh_r = jax.nn.one_hot(idx_r, num_bins)
    return oh_l * (1.0 - frac)[..., None] + oh_r * frac[..., None]


def _expected_symlog_value(logits: jnp.ndarray, centers_log: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    exp_symlog = jnp.sum(probs * centers_log[None, None, :], axis=-1)
    return _symexp(exp_symlog)


def _compute_lambda_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    gamma: float,
    lambda_: float,
) -> jnp.ndarray:
    def step(carry, inputs):
        g_next = carry
        r_t1, v_t1 = inputs
        g_t = r_t1 + gamma * ((1.0 - lambda_) * v_t1 + lambda_ * g_next)
        return g_t, g_t

    r_rev = rewards[:, ::-1]
    v_next_rev = values[:, 1:][:, ::-1]
    _, g_rev = jax.lax.scan(step, values[:, -1], (r_rev.T, v_next_rev.T))
    return g_rev[::-1].T


@partial(jax.jit, static_argnames=("context_length",))
def sample_contexts(
    videos: jnp.ndarray,
    actions: jnp.ndarray,
    valid_mask: jnp.ndarray,
    rng: jax.Array,
    context_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    b, t = videos.shape[:2]
    valid_counts = valid_mask.astype(jnp.int32).sum(axis=1)
    first_valid = t - valid_counts
    fallback_start = jnp.maximum(0, t - context_length)
    start_min = jnp.where(valid_counts >= context_length, first_valid, fallback_start)
    start_max = jnp.full((b,), fallback_start, dtype=jnp.int32)
    num_choices = jnp.maximum(start_max - start_min + 1, 1)
    offsets = jnp.floor(jax.random.uniform(rng, (b,)) * num_choices).astype(jnp.int32)
    start_indices = start_min + offsets

    def slice_video(video_seq, start_idx):
        return jax.lax.dynamic_slice(
            video_seq,
            start_indices=(start_idx, 0, 0, 0),
            slice_sizes=(context_length, videos.shape[2], videos.shape[3], videos.shape[4]),
        )

    def slice_action(action_seq, start_idx):
        return jax.lax.dynamic_slice(
            action_seq,
            start_indices=(start_idx, 0),
            slice_sizes=(context_length, actions.shape[2]),
        )

    context_videos = jax.vmap(slice_video, in_axes=(0, 0))(videos, start_indices)
    context_actions = jax.vmap(slice_action, in_axes=(0, 0))(actions, start_indices)
    return context_videos, context_actions


def build_frozen_model(args: Args, rngs: nnx.Rngs) -> FrozenBCRewardModel:
    dynamics = DynamicsDreamer4(
        d_model=args.dyna_d_model,
        d_spatial=args.dyna_d_spatial,
        n_spatial=args.dyna_n_spatial,
        n_register=args.dyna_n_register,
        n_agent=args.dyna_n_agent,
        n_heads=args.dyna_n_head,
        n_actions=1,
        n_camera=None,
        calvin_actions=True,
        depth=args.dyna_n_block,
        k_max=args.dyna_k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode="wm_agent",
        dtype=args.dtype,
        param_dtype=args.param_dtype,
        use_flash_attention=args.use_flash_attention,
        rngs=rngs,
        pos_emb_type=args.pos_emb_type,
    )
    policy_head = PolicyHeadContinuousMTP(
        d_model=args.dyna_d_model,
        action_dim=7,
        L=args.L,
        dtype=jnp.float32,
        param_dtype=args.param_dtype,
        rngs=rngs,
    )
    reward_head = RewardHeadMTP(
        d_model=args.dyna_d_model,
        L=args.L,
        num_bins=args.num_reward_bins,
        log_low=args.reward_log_low,
        log_high=args.reward_log_high,
        dtype=jnp.float32,
        param_dtype=args.param_dtype,
        rngs=rngs,
    )
    return FrozenBCRewardModel(dynamics, policy_head, reward_head)


def build_trainable_model(args: Args, rngs: nnx.Rngs) -> PolicyValueModel:
    policy_head = PolicyHeadContinuousMTP(
        d_model=args.dyna_d_model,
        action_dim=7,
        L=args.L,
        dtype=jnp.float32,
        param_dtype=args.param_dtype,
        rngs=rngs,
    )
    value_head = ValueHead(
        d_model=args.dyna_d_model,
        num_bins=args.num_value_bins,
        log_low=args.value_log_low,
        log_high=args.value_log_high,
        dtype=jnp.float32,
        param_dtype=args.param_dtype,
        rngs=rngs,
    )
    return PolicyValueModel(policy_head, value_head)


def build_optimizer(model: PolicyValueModel, args: Args) -> nnx.ModelAndOptimizer:
    lr_schedule = get_lr_schedule(
        args.lr_schedule,
        args.init_lr,
        args.max_lr,
        args.decay_end,
        args.num_steps,
        args.warmup_steps,
        args.wsd_decay_steps,
    )
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.95,
        weight_decay=args.weight_decay,
        mu_dtype=args.param_dtype,
    )
    return nnx.ModelAndOptimizer(model, tx)


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated = NamedSharding(mesh, PartitionSpec())
    videos = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    actions = NamedSharding(mesh, PartitionSpec("data", None, None))
    vectors = NamedSharding(mesh, PartitionSpec("data", None))
    masks = NamedSharding(mesh, PartitionSpec("data", None))
    return mesh, replicated, videos, actions, vectors, masks


def shard_optimizer_states(optimizer: nnx.ModelAndOptimizer, replicated: NamedSharding) -> None:
    model_state = nnx.state(optimizer.model)
    nnx.update(optimizer.model, jax.lax.with_sharding_constraint(model_state, replicated))
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    nnx.update(optimizer, jax.lax.with_sharding_constraint(optimizer_state, replicated))


def shard_module_state(module: nnx.Module, replicated: NamedSharding) -> None:
    state = nnx.state(module)
    nnx.update(module, jax.lax.with_sharding_constraint(state, replicated))


def build_train_dataloader(args: Args) -> grain.DataLoaderIterator:
    dataloader = get_calvin_lang_dataloader(
        data_dirs=args.train_data_dirs,
        seq_len=args.seq_len,
        global_batch_size=args.batch_size,
        lang_folder=args.lang_folder,
        image_key="rgb_static",
        image_h=args.image_height,
        image_w=args.image_width,
        num_workers=8,
        prefetch_buffer_size=8,
        seed=args.seed,
        action_key="rel_actions",
        reward_at_end=args.reward_at_end,
    )
    initial_state = dataloader._create_initial_state()
    return grain.DataLoaderIterator(dataloader, initial_state)


def build_checkpoint_manager(args: Args) -> Optional[ocp.CheckpointManager]:
    if not (args.save_ckpt or args.restore_ckpt):
        return None

    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler)
    handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=args.log_checkpoint_interval,
        max_to_keep=3,
        keep_period=args.log_checkpoint_keep_period,
        step_format_fixed_length=6,
        cleanup_tmp_directories=True,
    )
    return ocp.CheckpointManager(
        os.path.abspath(args.ckpt_dir),
        options=options,
        handler_registry=handler_registry,
    )


def restore_pretrained_bc_reward(
    args: Args,
    model: FrozenBCRewardModel,
    replicated: NamedSharding,
) -> None:
    dummy_optimizer = nnx.ModelAndOptimizer(model, optax.adamw(1e-4))

    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(args.pretrained_bc_rew_ckpt),
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=handler_registry,
    )

    restore_step = args.pretrained_bc_rew_step or checkpoint_manager.latest_step()
    if restore_step is None:
        raise FileNotFoundError(
            f"No pretrained BC/reward checkpoint found in {args.pretrained_bc_rew_ckpt}"
        )

    abstract_optimizer = nnx.eval_shape(lambda: dummy_optimizer)
    abstract_state = nnx.state(abstract_optimizer)
    restore_args_tree = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(sharding=replicated),
        abstract_state,
    )
    restored = checkpoint_manager.restore(
        restore_step,
        args=ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(
                abstract_state,
                partial_restore=True,
                restore_args=restore_args_tree,
            )
        ),
    )
    nnx.update(dummy_optimizer, restored["model_state"])
    checkpoint_manager.close()
    print(
        f"Restored pretrained BC/reward checkpoint from step {restore_step} "
        f"({args.pretrained_bc_rew_ckpt})"
    )


def restore_or_initialize(
    args: Args,
    checkpoint_manager: Optional[ocp.CheckpointManager],
    tokenizer: TokenizerDreamer4,
    frozen_model: FrozenBCRewardModel,
    optimizer: nnx.ModelAndOptimizer,
    replicated: NamedSharding,
    rng: jax.Array,
) -> tuple[int, TokenizerDreamer4]:
    start_step = 0
    restored = False

    if args.restore_ckpt and checkpoint_manager is not None:
        restore_step = args.restore_step or checkpoint_manager.latest_step()
        if restore_step is not None:
            abstract_optimizer = nnx.eval_shape(lambda: optimizer)
            abstract_state = nnx.state(abstract_optimizer)
            restore_args_tree = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=replicated),
                abstract_state,
            )
            restored_state = checkpoint_manager.restore(
                restore_step,
                args=ocp.args.Composite(
                    model_state=ocp.args.PyTreeRestore(
                        abstract_state,
                        partial_restore=True,
                        restore_args=restore_args_tree,
                    )
                ),
            )
            nnx.update(optimizer, restored_state["model_state"])
            start_step = restore_step + 1
            restored = True
            print(f"Restored imagination policy checkpoint from step {restore_step} ({args.ckpt_dir})")

    rng, tok_rng = jax.random.split(rng)
    tokenizer = restore_dreamer4_tokenizer(replicated, tok_rng, args)
    restore_pretrained_bc_reward(args, frozen_model, replicated)

    if args.init_policy_from_bc and not restored:
        nnx.update(optimizer.model.policy_head, nnx.state(frozen_model.policy_head))
        print("Initialized trainable policy head from the BC prior head.")

    return start_step, tokenizer


def main(args: Args) -> None:
    if args.dyna_d_model != args.lang_emb_dim:
        raise ValueError(
            f"dyna_d_model ({args.dyna_d_model}) must equal lang_emb_dim ({args.lang_emb_dim}) "
            "to use CALVIN mpnet embeddings directly."
        )
    if args.context_length <= 0 or args.horizon <= 0:
        raise ValueError("context_length and horizon must be positive.")
    if args.context_length > args.seq_len:
        raise ValueError(
            f"context_length ({args.context_length}) must be <= seq_len ({args.seq_len})."
        )
    if args.val_context_len is None:
        args.val_context_len = args.context_length

    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} device(s).")

    rngs_frozen = nnx.Rngs(args.seed)
    rngs_trainable = nnx.Rngs(args.seed + 1)
    rng = jax.random.PRNGKey(args.seed)

    tokenizer = TokenizerDreamer4(
        in_dim=args.image_channels,
        image_height=args.image_height,
        image_width=args.image_width,
        enc_model_dim=args.tokenizer_enc_model_dim,
        enc_mlp_ratio=args.tokenizer_enc_mlp_ratio,
        enc_time_every=args.tokenizer_enc_time_every,
        enc_num_blocks=args.tokenizer_enc_n_block,
        enc_num_heads=args.tokenizer_enc_n_head,
        dec_model_dim=args.tokenizer_dec_model_dim,
        dec_mlp_ratio=args.tokenizer_dec_mlp_ratio,
        dec_time_every=args.tokenizer_dec_time_every,
        dec_num_blocks=args.tokenizer_dec_n_block,
        dec_num_heads=args.tokenizer_dec_n_head,
        latent_dim=args.d_latent,
        num_latent_tokens=args.n_latent,
        patch_size=args.patch_size,
        dropout=args.dropout,
        max_mask_ratio=0.0,
        param_dtype=args.param_dtype,
        dtype=args.dtype,
        use_flash_attention=args.use_flash_attention,
        rngs=rngs_frozen,
        pos_emb_type=args.pos_emb_type,
    )
    frozen_model = build_frozen_model(args, rngs_frozen)
    trainable_model = build_trainable_model(args, rngs_trainable)
    optimizer = build_optimizer(trainable_model, args)

    _, frozen_params, _ = nnx.split(frozen_model, nnx.Param, ...)
    _, trainable_params, _ = nnx.split(trainable_model, nnx.Param, ...)
    print("Frozen parameter counts:")
    print(count_parameters_by_component(frozen_params))
    print("Trainable parameter counts:")
    print(count_parameters_by_component(trainable_params))

    mesh, replicated, videos_sharding, actions_sharding, vector_sharding, mask_sharding = (
        build_mesh_and_sharding(num_devices)
    )
    shard_optimizer_states(optimizer, replicated)

    checkpoint_manager = build_checkpoint_manager(args)
    start_step, tokenizer = restore_or_initialize(
        args, checkpoint_manager, tokenizer, frozen_model, optimizer, replicated, rng
    )
    shard_module_state(tokenizer, replicated)
    shard_module_state(frozen_model, replicated)

    # Split frozen_model into (graphdef, state) so we can pass the state as a
    # plain JAX pytree arg to train_step rather than an NNX graph node.  NNX's
    # JIT output extraction calls _check_valid_context on every Pytree (Module)
    # found in args_out; frozen modules carry an outer-trace PytreeState and
    # therefore fail that check.  Passing the State object (which is NOT a
    # Pytree subclass) sidesteps the check entirely.
    frozen_graphdef, frozen_state_const = nnx.split(frozen_model)

    train_iterator = build_train_dataloader(args)
    env_val_state: Optional[CalvinEnvValidationState] = None
    if args.val_data_dirs and args.val_interval > 0 and jax.process_index() == 0:
        env_val_state = init_calvin_env_validation(args)

    @partial(nnx.jit, donate_argnums=0, static_argnames=("B", "T", "context_length", "horizon"))
    def train_step(
        optimizer: nnx.ModelAndOptimizer,
        tokenizer: TokenizerDreamer4,
        frozen_state: nnx.State,
        batch: dict,
        B: int,
        T: int,
        context_length: int,
        horizon: int,
        master_key: jax.Array,
        step: int,
    ) -> tuple[jax.Array, dict]:
        frozen = nnx.merge(frozen_graphdef, frozen_state)
        step_key = jax.random.fold_in(master_key, step)
        key_ctx, key_imag = jax.random.split(step_key)

        videos = jnp.asarray(batch["videos"], dtype=jnp.float32) / 255.0
        raw_actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
        valid_mask = jnp.asarray(batch["valid_mask"], dtype=jnp.bool_)
        task_embedding = jnp.asarray(batch["task_embedding"], dtype=args.dtype)

        context_frames, context_actions = sample_contexts(
            videos, raw_actions, valid_mask, key_ctx, context_length
        )

        z_btld = tokenizer.mask_and_encode(context_frames.astype(args.dtype), rng=None, training=False)["z"]
        z_context = pack_bottleneck_to_spatial(
            z_btld,
            n_spatial=args.dyna_n_spatial,
            k=args.dyna_packing_factor,
        ).astype(args.dtype)

        agent_tokens = jnp.broadcast_to(
            task_embedding[:, None, None, :],
            (B, context_length + horizon, args.dyna_n_agent, args.dyna_d_model),
        )

        def loss_and_aux(model: PolicyValueModel) -> tuple[jax.Array, dict]:
            _, imagined_actions, imagined_hidden_states = imagine_rollouts_continuous(
                dynamics=frozen.dynamics,
                policy_head=model.policy_head,
                z_context=z_context,
                context_actions=context_actions.astype(args.dtype),
                agent_tokens=agent_tokens,
                k_max=args.dyna_k_max,
                horizon=horizon,
                context_length=context_length,
                n_spatial=args.dyna_n_spatial,
                d=args.imagination_d,
                start_mode=args.imagination_start_mode,
                tau0_fixed=args.imagination_tau0_fixed,
                rng_key=key_imag,
                ctx_noise_tau=args.ctx_noise_tau,
            )

            imagined_hidden_states_sg = jax.lax.stop_gradient(imagined_hidden_states)

            rew_logits, centers_log_rw = frozen.reward_head(
                imagined_hidden_states[:, 1:],
                deterministic=True,
            )
            rewards = _expected_symlog_value(rew_logits[:, :, 0, :], centers_log_rw)

            val_logits, centers_log_val = model.value_head(
                imagined_hidden_states_sg,
                deterministic=False,
            )
            values = _expected_symlog_value(val_logits, centers_log_val)
            td_returns = _compute_lambda_returns(
                rewards,
                jax.lax.stop_gradient(values),
                args.gamma,
                args.lambda_,
            )

            value_targets = jax.lax.stop_gradient(_twohot_symlog_targets(td_returns, centers_log_val))
            value_log_probs = jax.nn.log_softmax(val_logits[:, :-1], axis=-1)
            value_ce = -jnp.sum(value_targets * value_log_probs, axis=-1)
            value_loss = jnp.mean(value_ce)

            bc_prior = frozen.policy_head(
                imagined_hidden_states_sg[:, :horizon],
                deterministic=True,
            )
            bc_prior_actions = squash_calvin_actions(bc_prior[:, :, 0, :]).astype(jnp.float32)
            imagined_actions_f32 = imagined_actions.astype(jnp.float32)
            bc_prior_loss = jnp.mean(
                (imagined_actions_f32 - jax.lax.stop_gradient(bc_prior_actions)) ** 2
            )

            actor_return = jnp.mean(td_returns)
            actor_loss = -actor_return

            total_loss = (
                args.actor_loss_weight * actor_loss
                + args.value_loss_weight * value_loss
                + args.bc_prior_weight * bc_prior_loss
            )
            metrics = {
                "loss": total_loss,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "bc_prior_loss": bc_prior_loss,
                "return_mean": actor_return,
                "reward_mean": jnp.mean(rewards),
                "value_mean": jnp.mean(values[:, :-1]),
                "action_abs_mean": jnp.mean(jnp.abs(imagined_actions_f32)),
            }
            return total_loss, metrics

        graphdef, state = nnx.split(optimizer.model)

        def loss_fn(state: nnx.State) -> tuple[jax.Array, dict]:
            model = nnx.merge(graphdef, state)
            return loss_and_aux(model)

        (loss, metrics), grads_state = jax.value_and_grad(loss_fn, has_aux=True)(state)
        grads = nnx.merge(graphdef, grads_state)
        optimizer.update(grads)
        return loss, metrics

    def _make_sharded_batch(elem: dict) -> dict:
        return {
            "videos": jax.make_array_from_process_local_data(videos_sharding, local_data=elem["videos"]),
            "actions": jax.make_array_from_process_local_data(actions_sharding, local_data=elem["actions"]),
            "valid_mask": jax.make_array_from_process_local_data(mask_sharding, local_data=elem["valid_mask"]),
            "task_embedding": jax.make_array_from_process_local_data(
                vector_sharding, local_data=elem["task_embedding"]
            ),
        }

    dataloader_train = (_make_sharded_batch(elem) for elem in train_iterator)

    if args.log and jax.process_index() == 0:
        wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.name,
            tags=args.tags,
            id=args.wandb_id or None,
            resume="allow",
            config=asdict(args),
        )

    train_rng = jax.random.PRNGKey(args.seed + 11)
    start_wall = time.time()

    with mesh:
        for step in range(start_step, args.num_steps + 1):
            batch = next(dataloader_train)
            train_rng, master_key = jax.random.split(train_rng)
            loss, metrics = train_step(
                optimizer,
                tokenizer,
                frozen_state_const,
                batch,
                B=args.batch_size,
                T=args.seq_len,
                context_length=args.context_length,
                horizon=args.horizon,
                master_key=master_key,
                step=step,
            )

            if step % args.log_interval == 0 and jax.process_index() == 0:
                metrics_np = jax.device_get(metrics)
                elapsed = time.time() - start_wall
                steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
                print(
                    f"[step {step:06d}] loss={float(metrics_np['loss']):.4f} "
                    f"actor={float(metrics_np['actor_loss']):.4f} "
                    f"value={float(metrics_np['value_loss']):.4f} "
                    f"bc={float(metrics_np['bc_prior_loss']):.4f} "
                    f"ret={float(metrics_np['return_mean']):.4f} "
                    f"({steps_per_sec:.2f} steps/s)"
                )
                if args.log:
                    wandb.log(
                        {
                            "train/loss": float(metrics_np["loss"]),
                            "train/actor_loss": float(metrics_np["actor_loss"]),
                            "train/value_loss": float(metrics_np["value_loss"]),
                            "train/bc_prior_loss": float(metrics_np["bc_prior_loss"]),
                            "train/return_mean": float(metrics_np["return_mean"]),
                            "train/reward_mean": float(metrics_np["reward_mean"]),
                            "train/value_mean": float(metrics_np["value_mean"]),
                            "train/action_abs_mean": float(metrics_np["action_abs_mean"]),
                            "train/step": step,
                        },
                        step=step,
                    )

            if (
                env_val_state is not None
                and args.val_interval > 0
                and step > 0
                and step % args.val_interval == 0
                and jax.process_index() == 0
            ):
                print("Running CALVIN environment validation...")
                val_model = ValidationPolicyWrapper(
                    dynamics=frozen_model.dynamics,
                    policy_head=optimizer.model.policy_head,
                )
                val_metrics, val_videos = run_calvin_env_validation(
                    args=args,
                    tokenizer=tokenizer,
                    model=val_model,
                    step=step,
                    state=env_val_state,
                )
                print(
                    f"[step {step:06d}] val_success={val_metrics['val/env_success_rate']:.4f} "
                    f"val_steps={val_metrics['val/env_mean_steps']:.2f}"
                )
                if args.log:
                    log_dict = dict(val_metrics)
                    for key, video in val_videos.items():
                        log_dict[f"val/videos/{key}"] = wandb.Video(
                            np.transpose(video, (0, 3, 1, 2)),
                            fps=10,
                            format="mp4",
                        )
                    wandb.log(log_dict, step=step)

            if (
                checkpoint_manager is not None
                and args.save_ckpt
                and step > 0
                and step % args.log_checkpoint_interval == 0
            ):
                optimizer_state = nnx.state(optimizer)
                checkpoint_manager.save(
                    step,
                    args=ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(optimizer_state),
                    ),
                )
                if jax.process_index() == 0:
                    print(f"Saved checkpoint to {args.ckpt_dir} at step {step}")

    close_calvin_env_validation(env_val_state)
    if checkpoint_manager is not None:
        checkpoint_manager.close()
    if args.log and jax.process_index() == 0:
        wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
