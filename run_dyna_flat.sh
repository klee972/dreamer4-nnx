#!/bin/bash

# uv run python jasmine/baselines/diffusion/train_dynamics_diffusion.py \
#     --data_dir data/data/coinrun_episodes/train \
#     --val_data_dir data/data/coinrun_episodes/val


uv run python jasmine/dreamer4/train_dynamics_flat_action.py
