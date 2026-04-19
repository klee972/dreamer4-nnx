# Dreamer4-NNX

A JAX + flax NNX-based implementation of the Dreamer4 world model. Trains a tokenizer and dynamics model using coinrun, VPT(minecraft), and CALVIN dataset.

## References

This project is based on the following two repos:
- [edwhu/dreamer4-jax](https://github.com/edwhu/dreamer4-jax)
- [p-doom/jasmine](https://github.com/p-doom/jasmine)

## Requirements

Key dependencies:

- Python 3.10+
- JAX 0.7.2 (CUDA 12)
- Flax 0.11.2
- Optax 0.2.6
- Orbax-checkpoint 0.11.32
- Grain 0.2.10
- einops, tyro, wandb, ffmpeg

Install packages:

```bash
uv sync
```

## Project Structure

```
jasmine/
├── jasmine/
│   ├── dreamer4/                 # Dreamer4 training scripts
│   │   ├── train_tokenizer.py
│   │   ├── train_dynamics.py
│   │   └── ...
│   ├── models/                   # Model definitions
│   └── utils/                    # Utilities
├── data/
│   └── jasmine_data/
│       └── minecraft/
│           └── openai/           # Data download scripts
├── run_tok_{dataset_name}.sh     # Tokenizer training entrypoint
└── run_dyna_{dataset_name}.sh    # Dynamics model training entrypoint
```

## Data Pipeline

Following is the instruction for Minecraft data preparation. For CALVIN, please refer to their repo(https://github.com/mees/calvin).

Prepare the Minecraft dataset by running the following steps in order.

All scripts should be run from the project root.

### 1. Download Index Files

Download JSON index files containing metadata for the OpenAI VPT dataset.

```bash
bash data/jasmine_data/minecraft/openai/download_index_files.sh
```

Default output path: `data/open_ai_index_files/`

### 2. Download Videos

Download Minecraft gameplay MP4 videos based on the index files. Supports parallel downloads and automatically resumes interrupted downloads.

```bash
uv run python data/jasmine_data/minecraft/openai/download_videos.py
```

Default output path: `data/minecraft_videos_10/`

Key options:

```bash
uv run python data/jasmine_data/minecraft/openai/download_videos.py \
    --index_file_path data/open_ai_index_files/all_10xx_Jun_29.json \
    --output_dir data/minecraft_videos_10/ \
    --num_workers 8 \
    --dry_run  # check total size without downloading
```

### 3. Download Action Files

Download the action sequence JSONL files corresponding to each video.

```bash
uv run python data/jasmine_data/minecraft/openai/download_actions_files.py
```

Default output path: `data/open_ai_minecraft_actions_files/`

### 4. Chunk Videos

Split long MP4 videos into 64-frame clips and chunk the corresponding action files in parallel. Uses ffmpeg for H.264 re-encoding and resolution scaling (384x224).

> **Warning**: Setting `DELETE_AFTER_CHUNK = True` will delete original files after chunking.

```bash
uv run python data/jasmine_data/chunk_mp4_to_mp4.py
```

Output path: `data/minecraft_chunk64_224p/`

### 5. Split Dataset

Split the chunked clips into train/val/test sets using hash-based deterministic assignment.

```bash
uv run python data/jasmine_data/split_dataset.py
```

Input path: `data/minecraft_chunk64_224p/`
Output path: `data/minecraft_chunk64_224p_split/{train,val,test}/`

Key options:

```bash
uv run python data/jasmine_data/split_dataset.py \
    --source_dir data/minecraft_chunk64_224p \
    --output_dir data/minecraft_chunk64_224p_split \
    --val_size 1000 \
    --test_size 1000 \
    --dry_run  # preview split counts without moving files
```

> **Note**: By default (`--move True`), files are moved (not copied) to save disk space.

### 6. Train Tokenizer

Train the Dreamer4 tokenizer on the chunked video data.

```bash
bash run_tok_{dataset_name}.sh
```

Internally runs `jasmine/dreamer4/train_tokenizer.py`.

### 7. Train Dynamics Model

Train the world model dynamics on top of the learned tokenizer.

```bash
bash run_dyna_{dataset_name}.sh
```

Internally runs `jasmine/dreamer4/train_dynamics.py`.

## Training Monitoring

Training logs are recorded with [Weights & Biases](https://wandb.ai). Log in before running:

```bash
wandb login
```
