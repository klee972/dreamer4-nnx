"""Train CALVIN BC and reward heads with mpnet sentence embeddings."""

# Full access mode apply_patch test comment.

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
from einops import reduce
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from jasmine.models.dreamer4_models import (
    DynamicsDreamer4,
    PolicyHeadContinuousMTP,
    RewardHeadMTP,
    TokenizerDreamer4,
    restore_dreamer4_tokenizer,
)
from jasmine.utils.calvin_dataloader import get_calvin_lang_dataloader
from jasmine.utils.dreamer4_utils import pack_bottleneck_to_spatial
from jasmine.utils.train_utils import count_parameters_by_component, get_lr_schedule
from jasmine.dreamer4.calvin_bc_validation import (
    CalvinEnvValidationState,
    close_calvin_env_validation,
    init_calvin_env_validation,
    run_calvin_env_validation,
)


@dataclass
class Args:
    num_steps: int = 50_000
    seed: int = 0
    seq_len: int = 96
    batch_size: int = 32
    batch_size_self: Optional[int] = None

    train_data_dirs: list[str] = field(
        default_factory=lambda: ["/home/4bkang/rl/calvin/dataset/task_ABCD_D/training"]
    )
    val_data_dirs: list[str] = field(
        default_factory=lambda: ["/home/4bkang/rl/calvin/dataset/task_ABCD_D/validation"]
    )
    lang_folder: str = "lang_all-mpnet-base-v2"
    reward_at_end: bool = True
    val_interval: int = 5000
    val_num_rollouts: int = 4
    val_num_videos: int = 4
    val_max_steps: int = 120
    val_context_len: int = 1
    val_show_gui: bool = False
    val_worker_timeout_sec: float = 120.0
    calvin_repo_root: str = field(default_factory=lambda: os.environ.get("CALVIN_ROOT", "/home/4bkang/rl/calvin"))
    calvin_python_bin: str = field(default_factory=lambda: os.environ.get("CALVIN_PYTHON_BIN", ""))

    init_lr: float = 0.0
    max_lr: float = 3e-4
    decay_end: float = 0.0
    wsd_decay_steps: int = 10_000
    warmup_steps: int = 5_000
    lr_schedule: str = "wsd"
    bootstrap_start: int = 5_000
    weight_decay: float = 1e-4

    L: int = 2
    num_reward_bins: int = 101
    reward_log_low: float = -3.0
    reward_log_high: float = 3.0
    loss_weight_shortcut: float = 1.0
    loss_weight_policy: float = 1.0
    loss_weight_reward: float = 0.3

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
    pretrained_dyn_ckpt: str = "ckpts/calvin/dreamer4/dynamics_96p_s"
    pretrained_dyn_step: int = 0
    lang_emb_dim: int = 768

    save_ckpt: bool = True
    restore_ckpt: bool = False
    restore_step: int = 0
    ckpt_dir: str = "ckpts/calvin/dreamer4/bc_rew_heads_96p"
    log_checkpoint_interval: int = 1_000
    log_checkpoint_keep_period: int = 10_000

    log: bool = True
    entity: str = "4bkang"
    project: str = "jasmine"
    name: str = "bc_rew_heads_dreamer4_calvin_mpnet"
    tags: list[str] = field(default_factory=lambda: ["bc", "reward", "dreamer4", "calvin"])
    log_interval: int = 50
    wandb_id: str = ""


class BCRewardModel(nnx.Module):
    def __init__(
        self,
        dynamics: DynamicsDreamer4,
        policy_head: PolicyHeadContinuousMTP,
        reward_head: RewardHeadMTP,
    ):
        self.dynamics = dynamics
        self.policy_head = policy_head
        self.reward_head = reward_head


def resolve_batch_size_self(args: Args) -> int:
    batch_size_self = args.batch_size // 2 if args.batch_size_self is None else args.batch_size_self
    if not (0 <= batch_size_self <= args.batch_size):
        raise ValueError(
            f"batch_size_self must be in [0, batch_size], got {batch_size_self} for batch_size={args.batch_size}"
        )
    return batch_size_self


def _symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


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


def _shift_actions(raw_actions: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    shifted = raw_actions[:, :-1].astype(dtype)
    sentinel = jnp.full((raw_actions.shape[0], 1, raw_actions.shape[-1]), jnp.nan, dtype=dtype)
    return jnp.concatenate([sentinel, shifted], axis=1)


def _sample_tau_for_step(
    rng: jax.Array,
    shape_bt: tuple[int, int],
    k_max: int,
    step_idx: jnp.ndarray,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, seq_len = shape_bt
    K = (1 << step_idx)
    u = jax.random.uniform(rng, (batch_size, seq_len), dtype=dtype)
    j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
    tau = j_idx.astype(dtype) / K.astype(dtype)
    tau_idx = j_idx * (k_max // K)
    return tau, tau_idx


def _sample_step_excluding_dmin(
    rng: jax.Array,
    shape_bt: tuple[int, int],
    k_max: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, seq_len = shape_bt
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jax.random.randint(rng, (batch_size, seq_len), 0, emax, dtype=jnp.int32)
    d = 1.0 / (1 << step_idx).astype(jnp.float32)
    return d, step_idx


def _gather_action_targets(
    actions_bta: jnp.ndarray,
    valid_bt: jnp.ndarray,
    L: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, seq_len = valid_bt.shape
    padded_actions = jnp.pad(actions_bta, ((0, 0), (0, L - 1), (0, 0)), constant_values=0.0)
    padded_valid = jnp.pad(valid_bt, ((0, 0), (0, L - 1)), constant_values=False)
    offsets = jnp.arange(L)
    indices = jnp.arange(seq_len)[:, None] + offsets[None, :]
    targets = padded_actions[:, indices, :]
    future_valid = padded_valid[:, indices]
    valid = valid_bt[:, :, None] & future_valid
    return targets, valid


def _gather_reward_targets(
    rewards_bt: jnp.ndarray,
    valid_bt: jnp.ndarray,
    L: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size, seq_len = valid_bt.shape
    padded_rewards = jnp.pad(rewards_bt, ((0, 0), (0, L - 1)), constant_values=0.0)
    padded_valid = jnp.pad(valid_bt, ((0, 0), (0, L - 1)), constant_values=False)
    offsets = jnp.arange(L)
    indices = jnp.arange(seq_len)[:, None] + offsets[None, :]
    targets = padded_rewards[:, indices]
    future_valid = padded_valid[:, indices]
    valid = valid_bt[:, :, None] & future_valid
    return targets, valid


def build_model(args: Args, rngs: nnx.Rngs) -> tuple[TokenizerDreamer4, BCRewardModel]:
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
        rngs=rngs,
        pos_emb_type=args.pos_emb_type,
    )
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
    return tokenizer, BCRewardModel(dynamics, policy_head, reward_head)


def build_optimizer(model: BCRewardModel, args: Args) -> nnx.ModelAndOptimizer:
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
) -> tuple[Mesh, NamedSharding, NamedSharding, NamedSharding, NamedSharding, NamedSharding]:
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


def restore_pretrained_dynamics(
    args: Args,
    dynamics: DynamicsDreamer4,
    replicated: NamedSharding,
) -> None:
    tx = optax.adamw(1e-4)
    dyn_optimizer = nnx.ModelAndOptimizer(dynamics, tx)

    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(args.pretrained_dyn_ckpt),
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=handler_registry,
    )
    restore_step = args.pretrained_dyn_step or checkpoint_manager.latest_step()
    if restore_step is None:
        raise FileNotFoundError(f"No pretrained dynamics checkpoint found in {args.pretrained_dyn_ckpt}")

    abstract_optimizer = nnx.eval_shape(lambda: dyn_optimizer)
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
    nnx.update(dyn_optimizer, restored["model_state"])
    checkpoint_manager.close()
    print(f"Restored pretrained dynamics from step {restore_step} ({args.pretrained_dyn_ckpt})")


def restore_or_initialize(
    args: Args,
    checkpoint_manager: Optional[ocp.CheckpointManager],
    optimizer: nnx.ModelAndOptimizer,
    tokenizer: TokenizerDreamer4,
    replicated: NamedSharding,
    rng: jax.Array,
) -> tuple[int, TokenizerDreamer4, nnx.ModelAndOptimizer]:
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
            print(f"Restored BC/reward checkpoint from step {restore_step} ({args.ckpt_dir})")

    rng, tok_rng = jax.random.split(rng)
    tokenizer = restore_dreamer4_tokenizer(replicated, tok_rng, args)

    if not restored:
        restore_pretrained_dynamics(args, optimizer.model.dynamics, replicated)

    return start_step, tokenizer, optimizer


def main(args: Args) -> None:
    if args.dyna_d_model != args.lang_emb_dim:
        raise ValueError(
            f"dyna_d_model ({args.dyna_d_model}) must equal lang_emb_dim ({args.lang_emb_dim}) "
            "to use CALVIN mpnet embeddings directly."
        )
    batch_size_self = resolve_batch_size_self(args)

    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} device(s).")

    rngs = nnx.Rngs(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    tokenizer, model = build_model(args, rngs)
    optimizer = build_optimizer(model, args)

    _, params, _ = nnx.split(model, nnx.Param, ...)
    print("Parameter counts:")
    print(count_parameters_by_component(params))

    mesh, replicated, videos_sharding, actions_sharding, vector_sharding, mask_sharding = (
        build_mesh_and_sharding(num_devices)
    )
    shard_optimizer_states(optimizer, replicated)

    checkpoint_manager = build_checkpoint_manager(args)
    start_step, tokenizer, optimizer = restore_or_initialize(
        args, checkpoint_manager, optimizer, tokenizer, replicated, rng
    )
    shard_module_state(tokenizer, replicated)

    train_iterator = build_train_dataloader(args)
    env_val_state: Optional[CalvinEnvValidationState] = None
    if args.val_data_dirs and args.val_interval > 0 and jax.process_index() == 0:
        env_val_state = init_calvin_env_validation(args)


    @partial(nnx.jit, donate_argnums=0, static_argnames=("B", "T", "B_self", "L"))
    def train_step(
        optimizer: nnx.ModelAndOptimizer,
        tokenizer: TokenizerDreamer4,
        batch: dict,
        B: int,
        T: int,
        B_self: int,
        L: int,
        master_key: jax.Array,
        step: int,
    ) -> tuple[jax.Array, dict]:
        step_key = jax.random.fold_in(master_key, step)
        key_tau, key_step_self, key_noise = jax.random.split(step_key, 3)

        videos = jnp.asarray(batch["videos"], dtype=jnp.float32) / 255.0
        raw_actions = jnp.asarray(batch["actions"], dtype=jnp.float32)
        rewards = jnp.asarray(batch["rewards"], dtype=jnp.float32)
        valid_mask = jnp.asarray(batch["valid_mask"], dtype=jnp.bool_)
        action_mask = jnp.asarray(batch["action_mask"], dtype=jnp.bool_)
        task_embedding = jnp.asarray(batch["task_embedding"], dtype=args.dtype)

        z_btld = tokenizer.mask_and_encode(videos.astype(args.dtype), rng=None, training=False)["z"]
        z1 = pack_bottleneck_to_spatial(
            z_btld,
            n_spatial=args.dyna_n_spatial,
            k=args.dyna_packing_factor,
        ).astype(args.dtype)
        actions_in = _shift_actions(raw_actions, args.dtype)
        agent_tokens = jnp.broadcast_to(
            task_embedding[:, None, None, :],
            (B, T, args.dyna_n_agent, args.dyna_d_model),
        )

        B_emp = B - B_self
        emax = jnp.log2(args.dyna_k_max).astype(jnp.int32)
        step_idx_emp = jnp.full((B_emp, T), emax, dtype=jnp.int32)
        d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), args.dyna_k_max)
        d_self = d_self.astype(args.dtype)
        step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)

        tau_full, tau_idx_full = _sample_tau_for_step(
            key_tau, (B, T), args.dyna_k_max, step_idx_full, dtype=args.dtype
        )
        tau_emp = tau_full[:B_emp]
        tau_self = tau_full[B_emp:]
        tau_idx_self = tau_idx_full[B_emp:]

        z0_full = jax.random.normal(key_noise, z1.shape, dtype=z1.dtype)
        z_tilde_full = ((jnp.asarray(1.0, dtype=args.dtype) - tau_full)[..., None, None] * z0_full + tau_full[..., None, None] * z1).astype(args.dtype)
        z_tilde_self = z_tilde_full[B_emp:]

        w_emp = 0.9 * tau_emp + 0.1
        w_self = 0.9 * tau_self + 0.1
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        tau_plus = tau_self + d_half
        tau_idx_plus = tau_idx_self + (args.dyna_k_max * d_half).astype(jnp.int32)

        valid_mask_f = valid_mask.astype(jnp.float32)
        valid_emp_f = valid_mask_f[:B_emp]
        valid_self_f = valid_mask_f[B_emp:]

        def loss_and_aux(model: BCRewardModel) -> tuple[jax.Array, dict]:
            z1_hat_full, h_btnd = model.dynamics(
                actions_in,
                step_idx_full,
                tau_idx_full,
                z_tilde_full,
                agent_tokens=agent_tokens,
            )
            if h_btnd is None:
                raise ValueError("Dynamics did not return agent readouts.")
            h_pooled = reduce(h_btnd, "b t n_agent d_model -> b t d_model", "mean")

            z1_hat_emp = z1_hat_full[:B_emp]
            z1_hat_self = z1_hat_full[B_emp:]

            flow_per = jnp.mean((z1_hat_emp - z1[:B_emp]) ** 2, axis=(2, 3))
            flow_denom = jnp.maximum(valid_emp_f.sum(), 1.0)
            loss_emp = jnp.sum(flow_per * w_emp * valid_emp_f) / flow_denom
            flow_mse = jnp.sum(flow_per * valid_emp_f) / flow_denom

            do_boot = (B_self > 0) & (step >= args.bootstrap_start)
            zero = jnp.array(0.0, dtype=z1.dtype)
            boot_mse = zero
            loss_self = zero

            if B_self > 0:
                agent_tokens_self = agent_tokens[B_emp:]
                z1_hat_half1, _ = model.dynamics(
                    actions_in[B_emp:],
                    step_idx_half,
                    tau_idx_self,
                    z_tilde_self,
                    agent_tokens=agent_tokens_self,
                )
                b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - tau_self)[..., None, None]
                z_prime = z_tilde_self + b_prime * d_half[..., None, None]
                z1_hat_half2, _ = model.dynamics(
                    actions_in[B_emp:],
                    step_idx_half,
                    tau_idx_plus,
                    z_prime,
                    agent_tokens=agent_tokens_self,
                )
                b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - tau_plus)[..., None, None]
                vhat_tau = (z1_hat_self - z_tilde_self) / (1.0 - tau_self)[..., None, None]
                vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
                boot_per = (1.0 - tau_self) ** 2 * jnp.mean((vhat_tau - vbar_target) ** 2, axis=(2, 3))
                boot_denom = jnp.maximum(valid_self_f.sum(), 1.0)
                loss_self_raw = jnp.sum(boot_per * w_self * valid_self_f) / boot_denom
                boot_mse_raw = jnp.sum(boot_per * valid_self_f) / boot_denom
                loss_self = jnp.where(do_boot, loss_self_raw, zero)
                boot_mse = jnp.where(do_boot, boot_mse_raw, zero)

            action_pred = model.policy_head(h_pooled, deterministic=False)
            action_targets, action_valid = _gather_action_targets(raw_actions, action_mask, L)
            safe_action_targets = jnp.where(action_valid[..., None], action_targets, 0.0)
            safe_action_pred = jnp.where(action_valid[..., None], action_pred, 0.0)
            bc_per = jnp.mean((safe_action_pred - safe_action_targets) ** 2, axis=-1)
            action_valid_f = action_valid.astype(jnp.float32)
            bc_denom = jnp.maximum(action_valid_f.sum(), 1.0)
            bc_mse = jnp.sum(bc_per * action_valid_f) / bc_denom

            reward_targets, reward_valid = _gather_reward_targets(rewards, valid_mask, L)
            rew_logits, centers_log = model.reward_head(h_pooled, deterministic=False)
            twohot = _twohot_symlog_targets(reward_targets, centers_log)
            logq = jax.nn.log_softmax(rew_logits, axis=-1)
            reward_ce_per = -jnp.sum(twohot * logq, axis=-1)
            reward_valid_f = reward_valid.astype(jnp.float32)
            reward_denom = jnp.maximum(reward_valid_f.sum(), 1.0)
            rw_ce = jnp.sum(reward_ce_per * reward_valid_f) / reward_denom

            shortcut_loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B
            total_loss = (
                args.loss_weight_shortcut * shortcut_loss
                + args.loss_weight_policy * bc_mse
                + args.loss_weight_reward * rw_ce
            )

            metrics = {
                "loss": total_loss,
                "flow_mse": flow_mse,
                "bootstrap_mse": boot_mse,
                "bc_mse": bc_mse,
                "rw_ce": rw_ce,
                "shortcut_loss": shortcut_loss,
                "reward_pos_rate": rewards.mean(),
            }
            return total_loss, metrics

        (loss, metrics), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(optimizer.model)
        optimizer.update(grads)
        return loss, metrics

    def _make_sharded_batch(elem: dict) -> dict:
        return {
            "videos": jax.make_array_from_process_local_data(videos_sharding, local_data=elem["videos"]),
            "actions": jax.make_array_from_process_local_data(actions_sharding, local_data=elem["actions"]),
            "rewards": jax.make_array_from_process_local_data(mask_sharding, local_data=elem["rewards"]),
            "valid_mask": jax.make_array_from_process_local_data(mask_sharding, local_data=elem["valid_mask"]),
            "action_mask": jax.make_array_from_process_local_data(mask_sharding, local_data=elem["action_mask"]),
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

    train_rng = jax.random.PRNGKey(args.seed + 1)
    start_wall = time.time()

    with mesh:
        for step in range(start_step, args.num_steps + 1):
            batch = next(dataloader_train)
            train_rng, master_key = jax.random.split(train_rng)
            loss, metrics = train_step(
                optimizer,
                tokenizer,
                batch,
                B=args.batch_size,
                T=args.seq_len,
                B_self=batch_size_self,
                L=args.L,
                master_key=master_key,
                step=step,
            )

            if step % args.log_interval == 0 and jax.process_index() == 0:
                metrics_np = jax.device_get(metrics)
                elapsed = time.time() - start_wall
                steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
                print(
                    f"[step {step:06d}] loss={float(metrics_np['loss']):.4f} "
                    f"flow={float(metrics_np['flow_mse']):.4f} "
                    f"boot={float(metrics_np['bootstrap_mse']):.4f} "
                    f"bc={float(metrics_np['bc_mse']):.4f} "
                    f"rew={float(metrics_np['rw_ce']):.4f} "
                    f"({steps_per_sec:.2f} steps/s)"
                )
                if args.log:
                    wandb.log(
                        {
                            "train/loss": float(metrics_np["loss"]),
                            "train/flow_mse": float(metrics_np["flow_mse"]),
                            "train/bootstrap_mse": float(metrics_np["bootstrap_mse"]),
                            "train/bc_mse": float(metrics_np["bc_mse"]),
                            "train/rw_ce": float(metrics_np["rw_ce"]),
                            "train/shortcut_loss": float(metrics_np["shortcut_loss"]),
                            "train/reward_pos_rate": float(metrics_np["reward_pos_rate"]),
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
                val_metrics, val_videos = run_calvin_env_validation(
                    args=args,
                    tokenizer=tokenizer,
                    model=optimizer.model,
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
