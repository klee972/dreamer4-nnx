"""
Coinrun tokenizer reconstruction test.
Loads the latest checkpoint and saves a GT vs. recon comparison image.

Usage:
    cd /home/4bkang/rl/jasmine
    python test_tokenizer_coinrun.py
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import numpy as np
from PIL import Image
import einops

from jasmine.models.dreamer4_models import TokenizerDreamer4
from jasmine.utils.dataloader import get_dataloader
import grain

# ── Config (must match train_tokenizer_coinrun.py) ──────────────────────────
CKPT_DIR    = "/home/4bkang/rl/jasmine/ckpts/coinrun/dreamer4/tokenizer"
DATA_DIR    = "/home/4bkang/rl/jasmine/data/coinrun_episodes/val"
OUT_PATH    = "/home/4bkang/rl/jasmine/tokenizer_recon_test.png"

IMAGE_H, IMAGE_W, IMAGE_C = 64, 64, 3
SEQ_LEN        = 16          # shorter than training seq_len is fine
BATCH_SIZE     = 2

MODEL_DIM      = 512
MLP_RATIO      = 4
LATENT_DIM     = 32
N_LATENT       = 16
TIME_EVERY     = 4
PATCH_SIZE     = 16
N_BLOCKS       = 4
N_HEADS        = 8
DROPOUT        = 0.0
PARAM_DTYPE    = jnp.float32
DTYPE          = jnp.bfloat16
FLASH_ATTN     = True
POS_EMB_TYPE   = "rope"

# ── Build tokenizer ──────────────────────────────────────────────────────────
print("Building tokenizer...")
rngs = nnx.Rngs(0)
tokenizer = TokenizerDreamer4(
    in_dim=IMAGE_C,
    image_height=IMAGE_H,
    image_width=IMAGE_W,
    model_dim=MODEL_DIM,
    mlp_ratio=MLP_RATIO,
    latent_dim=LATENT_DIM,
    num_latent_tokens=N_LATENT,
    time_every=TIME_EVERY,
    patch_size=PATCH_SIZE,
    num_blocks=N_BLOCKS,
    num_heads=N_HEADS,
    dropout=DROPOUT,
    max_mask_ratio=0.0,          # no masking at eval
    param_dtype=PARAM_DTYPE,
    dtype=DTYPE,
    use_flash_attention=FLASH_ATTN,
    rngs=rngs,
    pos_emb_type=POS_EMB_TYPE,
)

# ── Restore checkpoint ───────────────────────────────────────────────────────
# Checkpoint was saved as nnx.state(ModelAndOptimizer), so wrap in a dummy optimizer
print("Restoring checkpoint...")
tx = optax.adamw(learning_rate=1e-4)
dummy_optimizer = nnx.ModelAndOptimizer(tokenizer, tx)
dummy_state = nnx.state(dummy_optimizer)

# replicated sharding (single device or multi-device)
devices = jax.local_devices()
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.mesh_utils import create_device_mesh
mesh = Mesh(create_device_mesh((len(devices),)), axis_names=("data",))
sharding = NamedSharding(mesh, PartitionSpec())

restore_args = jax.tree_util.tree_map(
    lambda _: ocp.ArrayRestoreArgs(sharding=sharding),
    dummy_state,
)

handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
handler_registry.add("model_state", ocp.args.PyTreeRestore, ocp.handlers.PyTreeCheckpointHandler)
ckpt_mgr = ocp.CheckpointManager(
    CKPT_DIR,
    options=ocp.CheckpointManagerOptions(step_format_fixed_length=6),
    handler_registry=handler_registry,
)
latest = ckpt_mgr.latest_step()
print(f"Latest checkpoint step: {latest}")

restored = ckpt_mgr.restore(
    latest,
    args=ocp.args.Composite(
        model_state=ocp.args.PyTreeRestore(dummy_state, partial_restore=True, restore_args=restore_args),
    ),
)["model_state"]
nnx.update(dummy_optimizer, restored)
tokenizer = dummy_optimizer.model
ckpt_mgr.close()
print("Checkpoint restored.")

# ── Load a batch of data ─────────────────────────────────────────────────────
print("Loading data...")
array_record_files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".array_record")
])
loader = get_dataloader(
    array_record_files, SEQ_LEN, BATCH_SIZE,
    IMAGE_H, IMAGE_W, IMAGE_C,
    num_workers=0, prefetch_buffer_size=1, seed=0,
)
state = loader._create_initial_state()
iterator = grain.DataLoaderIterator(loader, state)
batch = next(iterator)
videos = batch["videos"]  # (B, T, H, W, C) uint8

# ── Encode + Decode ──────────────────────────────────────────────────────────
print("Running encode/decode...")
tokenizer.eval()

@nnx.jit
def encode_decode(model, videos_BTHWC):
    gt = videos_BTHWC.astype(DTYPE) / 255.0
    outputs = model({"videos": gt, "rng": jax.random.PRNGKey(0)}, training=False)
    return outputs["recon"].astype(jnp.float32)

gt_f32 = jnp.asarray(videos, dtype=jnp.float32) / 255.0
recon = encode_decode(tokenizer, jnp.asarray(videos))

# ── Save comparison image ────────────────────────────────────────────────────
# Layout: each row = one sample; columns alternate GT | Recon frames
# rows = B samples, columns = T frames interleaved (GT top half, Recon bottom half per row)
print("Saving comparison image...")

B, T = videos.shape[:2]
rows = []
for b in range(B):
    gt_row   = np.array(gt_f32[b])    # (T, H, W, C)
    recon_row = np.array(recon[b]).clip(0, 1)  # (T, H, W, C)
    # stack GT and recon vertically per frame, then tile horizontally
    combined = np.concatenate([gt_row, recon_row], axis=1)  # (T, 2H, W, C)
    combined = einops.rearrange(combined, "t h w c -> h (t w) c")
    rows.append(combined)

grid = np.concatenate(rows, axis=0)  # (B*2H, T*W, C)
grid_uint8 = (grid * 255).clip(0, 255).astype(np.uint8)
Image.fromarray(grid_uint8).save(OUT_PATH)

print(f"Saved: {OUT_PATH}")
print(f"GT range:    [{gt_f32.min():.3f}, {gt_f32.max():.3f}]")
print(f"Recon range: [{float(recon.min()):.3f}, {float(recon.max()):.3f}]")
mse = float(jnp.mean((gt_f32 - recon) ** 2))
psnr = -10 * np.log10(mse + 1e-12)
print(f"MSE: {mse:.5f}  |  PSNR: {psnr:.2f} dB")
