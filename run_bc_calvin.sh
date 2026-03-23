#!/bin/bash

#SBATCH -J bc_calvin
#SBATCH -w a5k01
#SBATCH --gres=gpu:4
#SBATCH -t 2-00

export CALVIN_ROOT=/home/4bkang/rl/calvin
export CALVIN_PYTHON_BIN=$HOME/.conda/envs/calvin_venv/bin/python

mkdir -p /home/4bkang/rl/worldmodel/logs
cd /home/4bkang/rl/worldmodel

uv run python jasmine/dreamer4/train_bc_rew_heads_calvin.py
