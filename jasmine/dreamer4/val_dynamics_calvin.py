"""
Standalone validation script for CALVIN dynamics model.

Loads a dynamics checkpoint and evaluates on the validation dataset
using sample_video. Logs metrics and side-by-side videos to wandb.
"""

import glob
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import tyro
import wandb
import grain
import optax
import orbax.checkpoint as ocp
import flax.nnx as nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh

from jasmine.models.dreamer4_models import DynamicsDreamer4, TokenizerDreamer4, restore_dreamer4_tokenizer
from jasmine.utils.dataloader import get_dataloader
from jasmine.utils.train_utils import get_lr_schedule, count_parameters_by_component
from jasmine.dreamer4.sampler import sample_video, SamplerConfig


@dataclass
class Args:
    seed: int = 0
    seq_len: int = 96
    image_channels: int = 3
    image_height: int = 96
    image_width: int = 96
    batch_size: int = 1
    # Common model
    time_every: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.0

    param_dtype = jnp.float32
    dtype = jnp.bfloat16
    use_flash_attention: bool = True
    patch_size: int = 16
    pos_emb_type: str = "rope"
    # Latent tokens
    d_latent: int = 32
    n_latent: int = 32
    # Tokenizer
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
    # Dynamics
    dyna_d_model: int = 768
    dyna_packing_factor: int = 1
    dyna_d_spatial: int = 32
    dyna_n_spatial: int = 32
    dyna_n_register: int = 4
    dyna_n_agent: int = 1
    dyna_n_block: int = 8
    dyna_n_head: int = 12
    dyna_k_max: int = 128
    ctx_length: int = 8
    ctx_noise_tau: float = 0.9
    # Checkpoint
    ckpt_dir: str = "ckpts/calvin/dreamer4/dynamics_96p_s"
    restore_step: int = 0  # 0 = latest
    # Validation data
    val_data_dir: str = "data/calvin_96p_clips/val"
    val_steps: int = 2
    # Logging
    log: bool = True
    entity: str = "4bkang"
    project: str = "jasmine"
    name: str = "val_dynamics_dreamer4_calvin_96p_s"
    tags: list[str] = field(default_factory=lambda: ["val", "dynamics", "dreamer4"])
    wandb_id: str = ""
    # Validation regimes: any subset of ["shortcut_d4", "finest"]
    # eval_regimes: list[str] = field(default_factory=lambda: ["shortcut_d4", "finest"])
    # eval_regimes: list[str] = field(default_factory=lambda: ["finest"])
    eval_regimes: list[str] = field(default_factory=lambda: ["shortcut_d4"])




def build_model(args: Args, rngs: nnx.Rngs) -> tuple[TokenizerDreamer4, DynamicsDreamer4]:
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
        n_actions=1,  # dummy, not used by CALVINActionEncoderContinuous
        n_camera=None,
        calvin_actions=True,
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
        decode=True,
    )
    return tokenizer, dynamics


def build_mesh_and_sharding(
    num_devices: int,
) -> tuple[Mesh, NamedSharding, NamedSharding, NamedSharding]:
    device_mesh_arr = create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh_arr, axis_names=("data",))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    videos_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None, None))
    # CALVIN actions: (B, T, 7) float32 continuous [arm x/y/z, ori a/b/c, gripper]; NaN = sentinel
    actions_sharding = NamedSharding(mesh, PartitionSpec("data", None, None))
    return mesh, replicated_sharding, videos_sharding, actions_sharding


def restore_dynamics(
    args: Args,
    optimizer: nnx.ModelAndOptimizer,
    replicated_sharding: NamedSharding,
) -> DynamicsDreamer4:
    handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    handler_registry.add(
        "model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler
    )
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(args.ckpt_dir),
        options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
        handler_registry=handler_registry,
    )
    restore_step = args.restore_step or checkpoint_manager.latest_step()

    abstract_optimizer = nnx.eval_shape(lambda: optimizer)
    abstract_optimizer_state = nnx.state(abstract_optimizer)
    restore_args_tree = jax.tree.map(
        lambda _: ocp.ArrayRestoreArgs(sharding=replicated_sharding),
        abstract_optimizer_state,
    )
    restored = checkpoint_manager.restore(
        restore_step,
        args=ocp.args.Composite(
            model_state=ocp.args.PyTreeRestore(  # type: ignore
                abstract_optimizer_state,
                partial_restore=True,
                restore_args=restore_args_tree,
            )
        ),
    )
    nnx.update(optimizer, restored["model_state"])
    checkpoint_manager.close()
    print(f"Restored dynamics from step {restore_step} ({args.ckpt_dir})")
    return optimizer.model


def build_val_dataloader(args: Args) -> grain.DataLoaderIterator:
    array_record_paths = sorted(
        glob.glob(os.path.join(args.val_data_dir, "*.array_record"))
    )
    grain_dataloader = get_dataloader(
        array_record_paths=array_record_paths,
        seq_len=args.seq_len,
        global_batch_size=args.batch_size,
        image_h=args.image_height,
        image_w=args.image_width,
        image_c=args.image_channels,
        num_workers=8,
        prefetch_buffer_size=8,
        seed=args.seed,
    )
    initial_state = grain_dataloader._create_initial_state()
    return grain.DataLoaderIterator(grain_dataloader, initial_state)


def _eval_regimes_for_realism(cfg, *, ctx_length: int):
    common = dict(
        dyna_k_max=cfg.dyna_k_max,
        horizon=cfg.seq_len - cfg.ctx_length,
        ctx_length=ctx_length,
        ctx_noise_tau=cfg.ctx_noise_tau,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        image_channels=cfg.image_channels,
        patch_size=cfg.patch_size,
        dyna_n_spatial=cfg.dyna_n_spatial,
        dyna_packing_factor=cfg.dyna_packing_factor,
        start_mode="pure",
        rollout="autoregressive",
    )
    regs = []
    if "shortcut_d4" in cfg.eval_regimes:
        regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
    if "finest" in cfg.eval_regimes:
        regs.append(("finest_pure_AR", SamplerConfig(schedule="finest", **common)))
    return regs


def main(args: Args) -> None:
    num_devices = jax.device_count()
    if num_devices == 0:
        raise ValueError("No JAX devices found.")
    print(f"Running on {num_devices} devices.")

    rngs = nnx.Rngs(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    # --- Build model ---
    tokenizer, dynamics = build_model(args, rngs)
    _, params, _ = nnx.split(dynamics, nnx.Param, ...)
    param_counts = count_parameters_by_component(params)
    print("Parameter counts:")
    print(param_counts)

    # Dummy optimizer to match checkpoint state structure
    lr_schedule = get_lr_schedule("wsd", 0.0, 3e-4, 0.0, 150_000, 5000, 30_000)
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.9, weight_decay=1e-4,
                     mu_dtype=args.param_dtype)
    optimizer = nnx.ModelAndOptimizer(dynamics, tx)
    del dynamics

    _, replicated_sharding, videos_sharding, actions_sharding = build_mesh_and_sharding(num_devices)

    # Shard model state
    model_state = nnx.state(optimizer.model)
    nnx.update(optimizer.model, jax.lax.with_sharding_constraint(model_state, replicated_sharding))

    # --- Restore dynamics checkpoint ---
    dynamics = restore_dynamics(args, optimizer, replicated_sharding)

    # --- Restore tokenizer ---
    rng, _rng = jax.random.split(rng)
    tokenizer = restore_dreamer4_tokenizer(replicated_sharding, _rng, args)

    # --- wandb ---
    if args.log and jax.process_index() == 0:
        wandb_init_kwargs = dict(
            entity=args.entity,
            project=args.project,
            name=args.name,
            tags=args.tags,
            config=args,
        )
        if args.wandb_id:
            wandb_init_kwargs.update({"id": args.wandb_id, "resume": "allow"})
        wandb.init(**wandb_init_kwargs)
        wandb.config.update({"model_param_count": param_counts})

    # --- Val dataloader ---
    val_iterator = build_val_dataloader(args)

    def _stack_actions(elem: dict) -> np.ndarray:
        """Shift CALVIN continuous rel_actions: frame 0 gets NaN sentinel, frame t gets action_{t-1}.

        Returns (B, T, 7) bfloat16 — matches model dtype to avoid mixed-precision collective-permute.
        """
        actions = elem["actions"][:, :-1].astype(args.dtype)           # (B, T-1, 7) bf16
        sentinel = np.full((actions.shape[0], 1, 7), np.nan, dtype=args.dtype)
        return np.concatenate([sentinel, actions], axis=1)             # (B, T, 7) bf16

    dataloader_val = (
        {
            "videos": jax.make_array_from_process_local_data(
                videos_sharding, elem["videos"]
            ),
            "actions": jax.make_array_from_process_local_data(
                actions_sharding, _stack_actions(elem)
            ),
        }
        for elem in val_iterator
    )

    # --- val_step ---
    def val_step(dynamics: DynamicsDreamer4, tokenizer: TokenizerDreamer4, inputs: dict) -> dict:
        dynamics.eval()
        gt = jnp.asarray(inputs["videos"], dtype=jnp.float32) / 255.0
        actions = inputs["actions"]

        ctx_length = min(args.ctx_length, args.seq_len - 1)
        regimes = _eval_regimes_for_realism(args, ctx_length=ctx_length)

        val_output = {}
        for tag, sampler_conf in regimes:
            sampler_conf.rng_key = jax.random.PRNGKey(4242)
            t0 = time.perf_counter()

            pred_frames, floor_frames, gt_frames = sample_video(
                tokenizer, dynamics, gt.astype(args.dtype), actions, sampler_conf
            )

            # JAX dispatch is asynchronous, so block before stopping the timer.
            pred_frames, floor_frames = jax.block_until_ready((pred_frames, floor_frames))
            dt = time.perf_counter() - t0
            HZ = sampler_conf.horizon
            mse = (jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2)).astype(float)
            psnr = (10.0 * jnp.log10(1.0 / jnp.maximum(mse, 1e-12))).astype(float)

            val_output.update({
                f"{tag}_recon": pred_frames,
                f"{tag}_gt": gt_frames,
                f"{tag}_floor": floor_frames,
                f"{tag}_latency": dt,
                f"{tag}_mse": mse,
                f"{tag}_psnr": psnr,
            })
        return val_output

    # --- calculate_validation_metrics ---
    def calculate_validation_metrics(val_dataloader):
        ctx_length = min(args.ctx_length, args.seq_len - 1)
        regimes = _eval_regimes_for_realism(args, ctx_length=ctx_length)
        tags = [tag for tag, _ in regimes]

        N_VIZ = 10
        metrics_accum = {
            tag: {
                "latency": [],
                "latency_post_compile": [],
                "mse": [],
                "psnr": [],
            }
            for tag in tags
        }
        viz_gt    = {tag: [] for tag in tags}
        viz_recon = {tag: [] for tag in tags}
        viz_floor = {tag: [] for tag in tags}

        for val_step_count, batch in enumerate(val_dataloader):
            val_outputs = val_step(dynamics, tokenizer, batch)

            for tag in tags:
                latency = float(val_outputs[f"{tag}_latency"])
                mse = float(val_outputs[f"{tag}_mse"])
                psnr = float(val_outputs[f"{tag}_psnr"])

                metrics_accum[tag]["latency"].append(latency)
                metrics_accum[tag]["mse"].append(mse)
                metrics_accum[tag]["psnr"].append(psnr)
                if val_step_count > 0:
                    metrics_accum[tag]["latency_post_compile"].append(latency)

                phase = "compile+run" if val_step_count == 0 else "post-compile"
                print(
                    f"[val][step {val_step_count + 1}/{args.val_steps}][{tag}] "
                    f"latency={latency:.3f}s phase={phase} mse={mse:.6f} psnr={psnr:.2f}"
                )

                needed = N_VIZ - sum(a.shape[0] for a in viz_gt[tag])
                if needed > 0:
                    tag_gt    = np.asarray(val_outputs[f"{tag}_gt"])
                    tag_recon = np.asarray(val_outputs[f"{tag}_recon"])
                    tag_floor = np.asarray(val_outputs[f"{tag}_floor"])
                    take = min(needed, tag_gt.shape[0])
                    viz_gt[tag].append(tag_gt[:take])
                    viz_recon[tag].append(tag_recon[:take])
                    viz_floor[tag].append(tag_floor[:take])

            if val_step_count + 1 >= args.val_steps:
                break

        val_metrics = {}
        for tag in tags:
            val_metrics[f"{tag}_latency"] = np.mean(metrics_accum[tag]["latency"])
            post_compile = metrics_accum[tag]["latency_post_compile"]
            val_metrics[f"{tag}_latency_post_compile"] = (
                np.mean(post_compile) if post_compile else np.nan
            )
            val_metrics[f"{tag}_mse"]     = np.mean(metrics_accum[tag]["mse"])
            val_metrics[f"{tag}_psnr"]    = np.mean(metrics_accum[tag]["psnr"])

        val_videos = {}
        for tag in tags:
            val_videos[f"{tag}_gt"]    = np.concatenate(viz_gt[tag],    axis=0)[:N_VIZ] if viz_gt[tag]    else None
            val_videos[f"{tag}_recon"] = np.concatenate(viz_recon[tag], axis=0)[:N_VIZ] if viz_recon[tag] else None
            val_videos[f"{tag}_floor"] = np.concatenate(viz_floor[tag], axis=0)[:N_VIZ] if viz_floor[tag] else None

        return val_metrics, val_videos, tags

    # --- Run validation ---
    print("Calculating validation metrics...")
    val_metrics, val_videos, val_tags = calculate_validation_metrics(dataloader_val)

    for tag in val_tags:
        print(f"  [{tag}] PSNR: {val_metrics[f'{tag}_psnr']:.2f}  "
              f"MSE: {val_metrics[f'{tag}_mse']:.4f}  "
              f"Latency(all): {val_metrics[f'{tag}_latency']:.1f}s  "
              f"Latency(post-compile): {val_metrics[f'{tag}_latency_post_compile']:.1f}s")

    # --- Log to wandb ---
    if args.log and jax.process_index() == 0:
        log_dict = {**val_metrics}
        log_media = {}

        for tag in val_tags:
            tag_gt    = val_videos[f"{tag}_gt"]
            tag_recon = val_videos[f"{tag}_recon"]
            tag_floor = val_videos[f"{tag}_floor"]

            if tag_gt is None or tag_recon is None:
                continue

            # Per-sample images (first sample, last frame)
            log_media[f"val/{tag}_gt"]    = wandb.Image(
                np.clip(tag_gt[0][args.seq_len - 1] * 255, 0, 255).astype(np.uint8)
            )
            log_media[f"val/{tag}_recon"] = wandb.Image(
                np.clip(tag_recon[0][args.seq_len - 1] * 255, 0, 255).astype(np.uint8)
            )
            if tag_floor is not None:
                log_media[f"val/{tag}_floor"] = wandb.Image(
                    np.clip(tag_floor[0][args.seq_len - 1] * 255, 0, 255).astype(np.uint8)
                )

            # Per-sample GT|recon side-by-side videos (no floor)
            N = tag_gt.shape[0]
            for b in range(N):
                gt_b    = np.clip(tag_gt[b],    0.0, 1.0)
                recon_b = np.clip(tag_recon[b], 0.0, 1.0)
                # (T, H, W*2, C) → (T, C, H, W*2) uint8 for wandb.Video
                side_by_side = np.concatenate([gt_b, recon_b], axis=2)
                side_by_side = (side_by_side * 255).astype(np.uint8)
                side_by_side = np.transpose(side_by_side, (0, 3, 1, 2))
                log_media[f"val/{tag}_video/{b}"] = wandb.Video(side_by_side, fps=30, format="mp4")

        wandb.log({**log_dict, **log_media})
        print("Logged to wandb.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
