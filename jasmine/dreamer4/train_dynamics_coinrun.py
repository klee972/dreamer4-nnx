"""
Parts of the diffusion training, sampling, and DiT implementation are adapted from:
https://github.com/kvfrans/shortcut-models

For diffusion-forcing training, we integrate several elements inspired by Dreamer 4
(https://arxiv.org/abs/2509.24527).

Flat-action variant: for environments with a single discrete action index per timestep
(e.g. CoinRun with 15 actions). Actions are stored as (B, T) int32 tensors.
For Minecraft's hierarchical (button + camera) action space, use train_dynamics.py instead.
"""

import os
import time

# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass, field
from functools import partial
import itertools
from typing import cast, Optional

import einops
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
import optax
import orbax.checkpoint as ocp
import numpy as np
import dm_pix as pix
import jax
import jax.numpy as jnp
import tyro
import wandb
import grain
import flax.nnx as nnx

from jasmine.models.dreamer4_models import DynamicsDreamer4, TokenizerDreamer4, restore_dreamer4_tokenizer
from jasmine.utils.dataloader import get_dataloader
from jasmine.utils.train_utils import (
    get_lr_schedule,
    count_parameters_by_component,
    print_mem_stats,
    print_compiled_memory_stats,
    print_compiled_cost_analysis,
)
from jasmine.dreamer4.sampler import sample_video, SamplerConfig
from jasmine.utils.dreamer4_utils import patchify, unpatchify, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck



@dataclass
class Args:
    # Experiment
    num_steps: int = 150_000
    seed: int = 0
    seq_len: int = 64
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64
    data_dir: str = "data/coinrun_episodes/train"
    save_ckpt: bool = True
    restore_ckpt: bool = False
    restore_step: int = 0
    # Optimization
    batch_size: int = 32
    init_lr: float = 0.0
    max_lr: float = 3e-4
    decay_end: float = 0.0
    wsd_decay_steps: int = (
        20_000  # NOTE: wsd_decay_steps will only be used when using a wsd-schedule
    )
    warmup_steps: int = 5000
    lr_schedule: str = "wsd"  # supported options: wsd, cos
    bootstrap_start: int = 5_000  # shotcut distillation start step
    optimizer: str = "muon"  # supported options: adamw, muon
    # Common
    time_every: int = 4
    mlp_ratio: int = 4
    num_actions: int = 15  # flat action vocab size (e.g. 15 for CoinRun)
    dropout: float = 0.0
    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True
    patch_size: int = 16
    pos_emb_type: str = "rope"
    # Latent tokens
    d_latent: int = 32
    n_latent: int = 16
    # Tokenizer
    tokenizer_enc_model_dim: int = 512
    tokenizer_enc_mlp_ratio: int = 4
    tokenizer_enc_time_every: int = 4
    tokenizer_enc_n_block: int = 4
    tokenizer_enc_n_head: int = 8
    tokenizer_dec_model_dim: int = 512
    tokenizer_dec_mlp_ratio: int = 4
    tokenizer_dec_time_every: int = 4
    tokenizer_dec_n_block: int = 4
    tokenizer_dec_n_head: int = 8
    tokenizer_checkpoint: str = "ckpts/coinrun/dreamer4/tokenizer"
    # Dynamics
    dyna_d_model: int = 768
    dyna_packing_factor: int = 1
    dyna_d_spatial: int = 32  # must equal dyna_packing_factor * d_latent
    dyna_n_spatial: int = 16 # should be dyna_n_spatial * dyna_packing_factor = n_latent
    dyna_n_register: int = 4
    dyna_n_agent: int = 1
    dyna_n_block: int = 12
    dyna_n_head: int = 12
    dyna_k_max: int = 128
    batch_size_self: int = batch_size // 2
    ctx_length: int = 8  # num. gt frames given when validating
    # Logging
    log: bool = True
    entity: str = "4bkang"
    project: str = "jasmine"
    name: str = "dynamics_dreamer4_coinrun_muon"
    tags: list[str] = field(default_factory=lambda: ["dynamics", "dreamer4"])
    log_interval: int = 50
    log_image_interval: int = 1000
    ckpt_dir: str = "ckpts/coinrun/dreamer4/dynamics_muon"
    log_checkpoint_interval: int = 5000
    log_checkpoint_keep_period: int = 10_000
    log_gradients: bool = False
    val_data_dir: str = "data/coinrun_episodes/val"
    val_interval: int = 10000
    val_steps: int = 10
    wandb_id: str = ""
    


def build_model(args: Args, rngs: nnx.Rngs) -> tuple[TokenizerDreamer4, DynamicsDreamer4]:
    print(f"Action space: flat, num_actions={args.num_actions}")

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
    # n_camera=None → uses flat ActionEncoder (single embedding table)
    dynamics = DynamicsDreamer4(
        d_model=args.dyna_d_model,
        d_spatial=args.dyna_d_spatial,
        n_spatial=args.dyna_n_spatial,
        n_register=args.dyna_n_register,
        n_agent=args.dyna_n_agent,
        n_heads=args.dyna_n_head,
        n_actions=args.num_actions,
        n_camera=None,
        depth=args.dyna_n_block,
        k_max=args.dyna_k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode="wm_agent_isolated",
        dtype=args.dtype,
        param_dtype=args.param_dtype,
        use_flash_attention=args.use_flash_attention,
        rngs=rngs,
        pos_emb_type=args.pos_emb_type,
    )
    return tokenizer, dynamics


def build_optimizer(dynamics: DynamicsDreamer4, args: Args) -> nnx.ModelAndOptimizer:
    lr_schedule = get_lr_schedule(
        args.lr_schedule,
        args.init_lr,
        args.max_lr,
        args.decay_end,
        args.num_steps,
        args.warmup_steps,
        args.wsd_decay_steps,
    )
    if args.optimizer == "adamw":
        tx = optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.9,
            weight_decay=1e-4,
            mu_dtype=args.param_dtype,
        )
    elif args.optimizer == "muon":
        muon_tx = optax.contrib.muon(learning_rate=lr_schedule, mu_dtype=args.param_dtype)
        # Wrap init to strip NNX Variables before calling muon's init.
        # muon uses combine.partition which creates Param(MaskedNode) when
        # params contain Variables, causing to_opt_state to fail.
        # Optimizer.update already strips Variables via nnx.to_arrays(nnx.pure(...)),
        # so we only need to fix init.
        original_init = muon_tx.init
        def _compat_init(params):
            return original_init(nnx.to_arrays(nnx.pure(params)))
        tx = optax.GradientTransformation(_compat_init, muon_tx.update)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}. Supported: adamw, muon")
    optimizer = nnx.ModelAndOptimizer(dynamics, tx)
    return optimizer


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    # Flat actions: (B, T) int32
    actions_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    return mesh, replicated_sharding, videos_sharding, actions_sharding


def shard_optimizer_states(
    optimizer: nnx.ModelAndOptimizer, replicated_sharding: NamedSharding
) -> None:
    model_state = nnx.state(optimizer.model)
    model_sharded_state = jax.lax.with_sharding_constraint(
        model_state, replicated_sharding
    )
    nnx.update(optimizer.model, model_sharded_state)
    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(
        optimizer_state, replicated_sharding
    )
    nnx.update(optimizer, optimizer_sharded_state)


def build_dataloader(args: Args, data_dir: str) -> grain.DataLoaderIterator:
    image_shape = (args.image_height, args.image_width, args.image_channels)
    array_record_files = [
        os.path.join(data_dir, x)
        for x in os.listdir(data_dir)
        if x.endswith(".array_record")
    ]
    grain_dataloader = get_dataloader(
        array_record_files,
        args.seq_len,
        # NOTE: We deliberately pass the global batch size
        # The dataloader shards the dataset across all processes
        args.batch_size,
        *image_shape,
        num_workers=8,
        prefetch_buffer_size=1,
        seed=args.seed,
    )
    initial_state = grain_dataloader._create_initial_state()
    grain_iterator = grain.DataLoaderIterator(grain_dataloader, initial_state)
    return grain_iterator


def build_checkpoint_manager(args: Args) -> Optional[ocp.CheckpointManager]:
    if args.restore_ckpt or args.save_ckpt:
        handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        handler_registry.add(
            "model_state", ocp.args.PyTreeSave, ocp.handlers.PyTreeCheckpointHandler
        )
        handler_registry.add(
            "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
        )
        handler_registry.add(
            "train_dataloader_state",
            grain.checkpoint.CheckpointSave,
            cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
        )
        handler_registry.add(
            "train_dataloader_state",
            grain.checkpoint.CheckpointRestore,
            cast(ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler),
        )
        if args.val_data_dir:
            handler_registry.add(
                "val_dataloader_state",
                grain.checkpoint.CheckpointSave,
                cast(
                    ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler
                ),
            )
            handler_registry.add(
                "val_dataloader_state",
                grain.checkpoint.CheckpointRestore,
                cast(
                    ocp.handlers.CheckpointHandler, grain.checkpoint.CheckpointHandler
                ),
            )
        checkpoint_options = ocp.CheckpointManagerOptions(
            save_interval_steps=args.log_checkpoint_interval,
            max_to_keep=3,
            keep_period=args.log_checkpoint_keep_period,
            step_format_fixed_length=6,
            cleanup_tmp_directories=True,
        )
        checkpoint_manager = ocp.CheckpointManager(
            os.path.abspath(args.ckpt_dir),
            options=checkpoint_options,
            handler_registry=handler_registry,
        )
        return checkpoint_manager
    else:
        return None


def restore_or_initialize_components(
    args: Args,
    checkpoint_manager: Optional[ocp.CheckpointManager],
    optimizer: nnx.ModelAndOptimizer,
    tokenizer: TokenizerDreamer4,
    train_iterator: grain.DataLoaderIterator,
    rng: jax.Array,
    replicated_sharding: NamedSharding,
    val_iterator: Optional[grain.DataLoaderIterator],
    restore_step: Optional[int] = None,
) -> tuple[
    int,
    nnx.ModelAndOptimizer,
    TokenizerDreamer4,
    grain.DataLoaderIterator,
    grain.DataLoaderIterator,
    jax.Array,
]:
    step = 0
    if checkpoint_manager and restore_step is None:
        restore_step = checkpoint_manager.latest_step()
    if args.restore_ckpt:
        assert checkpoint_manager is not None
        abstract_optimizer = nnx.eval_shape(lambda: optimizer)
        abstract_optimizer_state = nnx.state(abstract_optimizer)
        # Explicitly remap every array to replicated_sharding so that checkpoints
        # saved on a different number of devices (e.g. 4 GPUs → 1 GPU) restore cleanly.
        restore_args_tree = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(sharding=replicated_sharding),
            abstract_optimizer_state,
        )
        if val_iterator:
            restore_args = ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(abstract_optimizer_state, partial_restore=True, restore_args=restore_args_tree),  # type: ignore
                train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
                val_dataloader_state=grain.checkpoint.CheckpointRestore(val_iterator),  # type: ignore
            )
        else:
            restore_args = ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(abstract_optimizer_state, partial_restore=True, restore_args=restore_args_tree),  # type: ignore
                train_dataloader_state=grain.checkpoint.CheckpointRestore(train_iterator),  # type: ignore
            )
        restored = checkpoint_manager.restore(restore_step, args=restore_args)
        restored_optimizer_state = restored["model_state"]
        nnx.update(optimizer, restored_optimizer_state)
        train_iterator = restored["train_dataloader_state"]
        if val_iterator:
            val_iterator = restored["val_dataloader_state"]
        step = restore_step or 0
        print(f"Restored dataloader and model state from step {step}")
    # Tokenizer is always frozen (not trained), so always restore from its pretrained checkpoint.
    rng, _rng = jax.random.split(rng)
    tokenizer = restore_dreamer4_tokenizer(replicated_sharding, _rng, args)
    return step, optimizer, tokenizer, train_iterator, val_iterator, rng


def _eval_regimes_for_realism(cfg, *, ctx_length: int):
    common = dict(
        dyna_k_max=cfg.dyna_k_max,
        horizon=cfg.seq_len - cfg.ctx_length,
        ctx_length=ctx_length,
        image_height=cfg.image_height, image_width=cfg.image_width, image_channels=cfg.image_channels, patch_size=cfg.patch_size,
        dyna_n_spatial=cfg.dyna_n_spatial,
        dyna_packing_factor=cfg.dyna_packing_factor,
        start_mode="pure",
        rollout="autoregressive",
    )
    regs = []
    regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
    return regs



def main(args: Args) -> None:
    # jax.distributed.initialize()
    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} devices.")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by "
            f"number of devices {num_devices}."
        )

    rngs = nnx.Rngs(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # --- Initialize model ---
    tokenizer, dynamics = build_model(args, rngs)
    _, params, _ = nnx.split(dynamics, nnx.Param, ...)
    param_counts = count_parameters_by_component(params)

    if args.log and jax.process_index() == 0:
        wandb_init_kwargs = {
            "entity": args.entity,
            "project": args.project,
            "name": args.name,
            "tags": args.tags,
            "group": "debug",
            "config": args,
        }

        if args.wandb_id:
            wandb_init_kwargs.update(
                {
                    "id": args.wandb_id,
                    "resume": "allow",
                }
            )
        wandb.init(**wandb_init_kwargs)

        wandb.config.update({"model_param_count": param_counts})

    print("Parameter counts:")
    print(param_counts)

    # --- Initialize optimizer ---
    optimizer = build_optimizer(dynamics, args)
    del dynamics

    # FIXME: switch to create_hybrid_device_mesh for runs spanning multiple nodes
    _, replicated_sharding, videos_sharding, actions_sharding = build_mesh_and_sharding(
        num_devices
    )

    shard_optimizer_states(optimizer, replicated_sharding)

    # --- Initialize checkpoint manager ---
    checkpoint_manager = build_checkpoint_manager(args)

    # --- Create DataLoaderIterator from dataloader ---
    train_iterator = build_dataloader(args, args.data_dir)
    val_iterator = None
    if args.val_data_dir:
        val_iterator = build_dataloader(args, args.val_data_dir)

    # --- Restore checkpoint ---
    step, optimizer, tokenizer, train_iterator, val_iterator, rng = (
        restore_or_initialize_components(
            args,
            checkpoint_manager,
            optimizer,
            tokenizer,
            train_iterator,
            rng,
            replicated_sharding,
            val_iterator,
        )
    )


    @partial(nnx.jit, donate_argnums=0, static_argnames=("B", "T", "B_self"))
    def train_step(
        optimizer: nnx.ModelAndOptimizer, tokenizer: TokenizerDreamer4, inputs: dict,
        B: int, T: int, B_self: int,
        master_key: jnp.ndarray, step: int, bootstrap_start: int,
    ) -> tuple[jax.Array, jax.Array, dict]:
        """
        Deterministic two-branch training (one fused main forward):
        - first B_emp rows: empirical flow at d_min = 1/k_max
        - last  B_self rows: bootstrap self-consistency with d > d_min
        If step < bootstrap_start, the bootstrap contribution is masked to 0 (but we still
        execute one fused path to keep a single jit and stable shapes).
        """
        @partial(jax.jit, static_argnames=("shape_bt","k_max","dtype"))
        def _sample_tau_for_step(rng, shape_bt, k_max:int, step_idx:jnp.ndarray, *, dtype=jnp.float32):
            B_, T_ = shape_bt
            K = (1 << step_idx)
            u = jax.random.uniform(rng, (B_, T_), dtype=dtype)
            j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)
            tau = j_idx.astype(dtype) / K.astype(dtype)
            tau_idx = j_idx * (k_max // K)
            return tau, tau_idx

        @partial(jax.jit, static_argnames=("shape_bt","k_max","dtype"))
        def _sample_step_excluding_dmin(rng, shape_bt, k_max:int, *, dtype=jnp.float32):
            B_, T_ = shape_bt
            emax = jnp.log2(k_max).astype(jnp.int32)
            step_idx = jax.random.randint(rng, (B_, T_), 0, emax, dtype=jnp.int32)  # exclude emax
            d = 1.0 / (1 << step_idx).astype(dtype)
            return d, step_idx
        step_key = jax.random.fold_in(master_key, step)
        enc_key, key_sigma_full, key_step_self, key_noise_full, drop_key = jax.random.split(step_key, 5)

        dynamics = optimizer.model
        actions = inputs["actions"]  # (B, T) flat int32
        gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0

        latent_tokens_BTNlL = tokenizer.mask_and_encode(gt.astype(args.dtype), rng=None, training=False)['z']
        z1 = pack_bottleneck_to_spatial(latent_tokens_BTNlL, n_spatial=args.dyna_n_spatial, k=args.dyna_packing_factor)

        # Deterministic batch split
        B_emp  = B - B_self
        # Shift actions: frame t should receive a_{t-1}; frame 0 gets sentinel -1.
        sentinel = jnp.full((B, 1), -1, dtype=actions.dtype)
        actions_full = jnp.concatenate([sentinel, actions[:, :-1]], axis=1)  # (B, T)
        emax = jnp.log2(args.dyna_k_max).astype(jnp.int32)

        # --- Step indices (encode d) ---
        step_idx_emp  = jnp.full((B_emp,  T), emax, dtype=jnp.int32)             # d = d_min
        d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), args.dyna_k_max, dtype=args.dtype)
        step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)   # (B,T)

        # --- Signal levels on each row's grid (one call for whole batch) ---
        sigma_full, sigma_idx_full = _sample_tau_for_step(key_sigma_full, (B, T), args.dyna_k_max, step_idx_full, dtype=args.dtype)
        sigma_emp   = sigma_full[:B_emp]
        sigma_self  = sigma_full[B_emp:]
        sigma_idx_self = sigma_idx_full[B_emp:]

        # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 ---
        z0_full      = jax.random.normal(key_noise_full, z1.shape, dtype=z1.dtype)
        z_tilde_full = (1.0 - sigma_full)[...,None,None] * z0_full + sigma_full[...,None,None] * z1
        z_tilde_self = z_tilde_full[B_emp:]

        # --- Ramp weights ---
        w_emp  = 0.9 * sigma_emp  + 0.1
        w_self = 0.9 * sigma_self + 0.1

        # --- Half-step metadata for self rows ---
        d_half            = d_self / 2.0
        step_idx_half     = step_idx_self + 1
        sigma_plus        = sigma_self + d_half
        sigma_idx_plus    = sigma_idx_self + (args.dyna_k_max * d_half).astype(jnp.int32)

        def loss_and_aux(dynamics: DynamicsDreamer4):
            # Main forward (emp + self)
            z1_hat_full, _ = dynamics(
                actions_full, step_idx_full, sigma_idx_full, z_tilde_full,
            )  # (B,T,n_spatial,d_model)

            z1_hat_emp  = z1_hat_full[:B_emp]
            z1_hat_self = z1_hat_full[B_emp:]

            # Flow loss on empirical rows (to z1)
            flow_per = jnp.mean((z1_hat_emp - z1[:B_emp])**2, axis=(2,3))        # (B_emp,T)
            loss_emp = jnp.mean(flow_per * w_emp)

            do_boot = (B_self > 0) & (step >= bootstrap_start)

            z1_hat_half1, _ = dynamics(
                actions_full[B_emp:], step_idx_half, sigma_idx_self, z_tilde_self,
            )
            b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            z_prime = z_tilde_self + b_prime * d_half[...,None,None]
            z1_hat_half2, _ = dynamics(
                actions_full[B_emp:], step_idx_half, sigma_idx_plus, z_prime,
            )
            b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - sigma_plus)[...,None,None]
            vhat_sigma = (z1_hat_self - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
            vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)
            boot_per = (1.0 - sigma_self)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))  # (B_self,T)
            boot_loss_raw = jnp.mean(boot_per * w_self)
            boot_mse_raw = jnp.mean(boot_per)

            zero = jnp.array(0.0, dtype=z1.dtype)
            loss_self = jnp.where(do_boot, boot_loss_raw, zero)
            boot_mse = jnp.where(do_boot, boot_mse_raw, zero)

            loss = ((loss_emp * (B - B_self)) + (loss_self * B_self)) / B
            metrics = {
                "flow_mse": jnp.mean(flow_per),
                "bootstrap_mse": boot_mse,
            }
            return loss, (z1_hat_full, metrics)


        (loss, (z1_hat, metrics)), grads = nnx.value_and_grad(loss_and_aux, has_aux=True)(
            optimizer.model
        )
        optimizer.update(grads)
        if args.log_gradients:
            metrics["gradients_std/"] = jax.tree.map(
                lambda x: x.std(), grads["params"]["dynamics"]
            )
        return loss, z1_hat, metrics

    # @nnx.jit
    def val_step(dynamics: DynamicsDreamer4, tokenizer: TokenizerDreamer4, inputs: dict) -> dict:
        """Evaluate model and compute metrics"""
        dynamics.eval()
        gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
        actions = inputs["actions"]

        ctx_length = min(args.ctx_length, args.seq_len - 1)
        regimes = _eval_regimes_for_realism(args, ctx_length=ctx_length)

        val_output = {}
        for tag, sampler_conf in regimes:
            sampler_conf.rng_key = jax.random.PRNGKey(4242)
            t0 = time.time()

            pred_frames, floor_frames, gt_frames = sample_video(
                tokenizer, dynamics, gt.astype(args.dtype), actions, sampler_conf
            )

            dt = time.time() - t0
            HZ = sampler_conf.horizon
            mse = (jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2)).astype(float)
            psnr = (10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12))).astype(float)

            val_output.update(
                {
                    f"{tag}_recon": pred_frames,
                    f"{tag}_gt": gt_frames,
                    f"{tag}_floor": floor_frames,
                    f"{tag}_latency": dt,
                    f"{tag}_mse": mse,
                    f"{tag}_psnr": psnr,
                }
            )
        return val_output

    def calculate_validation_metrics(val_dataloader, dynamics, tokenizer, rng):
        ctx_length = min(args.ctx_length, args.seq_len - 1)
        regimes = _eval_regimes_for_realism(args, ctx_length=ctx_length)
        tags = [tag for tag, _ in regimes]

        N_VIZ = 10
        metrics_accum = {tag: {"latency": [], "mse": [], "psnr": []} for tag in tags}
        viz_gt    = {tag: [] for tag in tags}
        viz_recon = {tag: [] for tag in tags}
        viz_floor = {tag: [] for tag in tags}

        val_step_count = 0
        for batch in val_dataloader:
            val_outputs = val_step(dynamics, tokenizer, batch)

            for tag in tags:
                metrics_accum[tag]["latency"].append(val_outputs[f"{tag}_latency"])
                metrics_accum[tag]["mse"].append(val_outputs[f"{tag}_mse"])
                metrics_accum[tag]["psnr"].append(val_outputs[f"{tag}_psnr"])
                needed = N_VIZ - sum(a.shape[0] for a in viz_gt[tag])
                if needed > 0:
                    tag_gt    = np.asarray(val_outputs[f"{tag}_gt"])
                    tag_recon = np.asarray(val_outputs[f"{tag}_recon"])
                    tag_floor = np.asarray(val_outputs[f"{tag}_floor"])
                    take = min(needed, tag_gt.shape[0])
                    viz_gt[tag].append(tag_gt[:take])
                    viz_recon[tag].append(tag_recon[:take])
                    viz_floor[tag].append(tag_floor[:take])

            val_step_count += 1
            if val_step_count >= args.val_steps:
                break

        val_metrics = {}
        for tag in tags:
            val_metrics[f"{tag}_latency"] = np.mean(metrics_accum[tag]["latency"])
            val_metrics[f"{tag}_mse"] = np.mean(metrics_accum[tag]["mse"])
            val_metrics[f"{tag}_psnr"] = np.mean(metrics_accum[tag]["psnr"])

        val_videos = {}
        for tag in tags:
            val_videos[f"{tag}_gt"]    = np.concatenate(viz_gt[tag],    axis=0)[:N_VIZ] if viz_gt[tag]    else None
            val_videos[f"{tag}_recon"] = np.concatenate(viz_recon[tag], axis=0)[:N_VIZ] if viz_recon[tag] else None
            val_videos[f"{tag}_floor"] = np.concatenate(viz_floor[tag], axis=0)[:N_VIZ] if viz_floor[tag] else None

        return val_metrics, val_videos, tags

    # --- TRAIN LOOP ---
    dataloader_train = (
        {
            "videos": jax.make_array_from_process_local_data(
                videos_sharding, local_data=elem["videos"]
            ),
            "actions": (
                jax.make_array_from_process_local_data(
                    actions_sharding, elem["actions"]
                )
            ),
        }
        for elem in train_iterator
    )
    dataloader_val = None
    if val_iterator:
        dataloader_val = (
            {
                "videos": jax.make_array_from_process_local_data(
                    videos_sharding, elem["videos"]
                ),
                "actions": (
                    jax.make_array_from_process_local_data(
                        actions_sharding, elem["actions"]
                    )
                ),
            }
            for elem in val_iterator
        )
    if jax.process_index() == 0:
        first_batch = next(dataloader_train)
        first_batch["rng"] = rng  # type: ignore
        compiled = train_step.lower(
            optimizer, tokenizer, first_batch,
            B=args.batch_size, T=args.seq_len, B_self=args.batch_size_self,
            master_key=rng, step=0, bootstrap_start=args.bootstrap_start).compile()
        print_compiled_memory_stats(compiled.memory_analysis())
        print_compiled_cost_analysis(compiled.cost_analysis())
        # Do not skip the first batch during training
        dataloader_train = itertools.chain([first_batch], dataloader_train)
    print(f"Starting training from step {step}...")
    first_step = step
    while step < args.num_steps:
        for batch in dataloader_train:
            # --- Train step ---
            rng, _rng_mask = jax.random.split(rng, 2)
            batch["rng"] = _rng_mask
            loss, z1_hat, metrics = train_step(
                optimizer, tokenizer, batch,
                B=args.batch_size, T=args.seq_len, B_self=args.batch_size_self,
                master_key=rng, step=step, bootstrap_start=args.bootstrap_start
            )
            z1_hat = einops.rearrange(
                z1_hat,
                'b t n_spatial (packing_factor d_latent) -> b t (n_spatial packing_factor) d_latent',
                n_spatial=args.dyna_n_spatial,
                packing_factor=args.dyna_packing_factor
            )
            recon = tokenizer.decode(z1_hat, (args.image_height, args.image_width))
            if step == first_step:
                print_mem_stats("After params initialized")
            step += 1

            # --- Validation loss ---
            val_results = {}
            if dataloader_val and step % args.val_interval == 0:
                rng, _rng_mask_val = jax.random.split(rng, 2)
                print("Calculating validation metrics...")
                val_metrics, val_videos, val_tags = calculate_validation_metrics(
                    dataloader_val, optimizer.model, tokenizer, _rng_mask_val
                )
                first_tag = val_tags[0] if val_tags else None
                if first_tag:
                    print(f"Step {step}, validation PSNR ({first_tag}): {val_metrics[f'{first_tag}_psnr']:.2f}")
                val_results = {
                    "metrics": val_metrics,
                    "videos": val_videos,
                    "tags": val_tags,
                }

            # --- Logging ---
            if args.log:
                if step % args.log_interval == 0 and jax.process_index() == 0:
                    log_dict = {"loss": loss, "step": step, **metrics}
                    if val_results:
                        log_dict.update(val_results["metrics"])
                    wandb.log(log_dict)
                if step % args.log_image_interval == 0:
                    gt_seq = batch["videos"][0].astype(jnp.float32) / 255.0
                    recon_seq = recon[0].clip(0, 1)
                    comparison_seq = jnp.concatenate((gt_seq, recon_seq), axis=1)
                    comparison_seq = einops.rearrange(
                        comparison_seq * 255, "t h w c -> h (t w) c"
                    )

                    val_video_logs = {}
                    if val_results:
                        val_videos = val_results["videos"]
                        val_tags = val_results["tags"]
                        for tag in val_tags:
                            tag_gt    = val_videos[f"{tag}_gt"]    # (N, T, H, W, C) float32
                            tag_recon = val_videos[f"{tag}_recon"] # (N, T, H, W, C) float32
                            tag_floor = val_videos[f"{tag}_floor"] # (N, T, H, W, C) float32

                            if tag_gt is None or tag_recon is None:
                                continue

                            # Per-sample images (first sample, last frame)
                            val_video_logs[f"val/{tag}_gt"]    = tag_gt[0]
                            val_video_logs[f"val/{tag}_recon"] = tag_recon[0]
                            if tag_floor is not None:
                                val_video_logs[f"val/{tag}_floor"] = tag_floor[0]

                            # Per-sample GT|recon side-by-side videos (no floor)
                            N = tag_gt.shape[0]
                            for b in range(N):
                                gt_b    = np.clip(tag_gt[b],    0.0, 1.0)
                                recon_b = np.clip(tag_recon[b], 0.0, 1.0)
                                # (T, H, W*2, C) → (T, C, H, W*2) uint8 for wandb.Video
                                side_by_side = np.concatenate([gt_b, recon_b], axis=2)
                                side_by_side = (side_by_side * 255).astype(np.uint8)
                                side_by_side = np.transpose(side_by_side, (0, 3, 1, 2))
                                val_video_logs[f"val/{tag}_video/{b}"] = side_by_side

                    if jax.process_index() == 0:
                        log_images = {
                            "train/image": wandb.Image(np.asarray(gt_seq[args.seq_len - 1])),
                            "train/recon": wandb.Image(np.asarray(recon_seq[args.seq_len - 1])),
                            "train/true_vs_recon": wandb.Image(
                                np.asarray(comparison_seq.astype(np.uint8))
                            ),
                        }

                        for key, video in val_video_logs.items():
                            if "_video/" in key:
                                # (T, H, W*2, C) uint8 → wandb.Video
                                log_images[key] = wandb.Video(video, fps=10, format="mp4")
                            else:
                                # Individual frame: last frame of sequence
                                log_images[key] = wandb.Image(
                                    np.clip(np.asarray(video[args.seq_len - 1]) * 255, 0, 255).astype(np.uint8)
                                )

                        wandb.log(log_images)
            # --- Checkpointing ---
            if args.save_ckpt and step % args.log_checkpoint_interval == 0:
                assert checkpoint_manager is not None
                optimizer_state = nnx.state(optimizer)
                if val_iterator:
                    ckpt_manager_args = ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(optimizer_state),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            train_iterator  # type: ignore
                        ),
                        val_dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            val_iterator  # type: ignore
                        ),
                    )
                else:
                    ckpt_manager_args = ocp.args.Composite(
                        model_state=ocp.args.PyTreeSave(optimizer_state),  # type: ignore
                        train_dataloader_state=grain.checkpoint.CheckpointSave(  # type: ignore
                            train_iterator  # type: ignore
                        ),
                    )
                checkpoint_manager.save(step, args=ckpt_manager_args)
                print(f"Saved checkpoint at step {step}")
            if step >= args.num_steps:
                break

    if checkpoint_manager:
        checkpoint_manager.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
