#!/bin/bash

GAME=$1
SEED=$SLURM_ARRAY_TASK_ID

source env_gpu/bin/activate

RLU_DIR=experiments/atari/datasets/rlu_dataset
NUMPY_DIR=experiments/atari/datasets/numpy_dataset
SLIM_DIR=experiments/atari/datasets/slim_dataset

echo "Downloading data..."
mkdir -p $RLU_DIR/$GAME
gsutil -m cp -R gs://rl_unplugged/atari_episodes_ordered/$GAME/run_$SEED* $RLU_DIR/$GAME

echo "Converting to Numpy..."
python3 experiments/atari/rlu_to_numpy.py --game $GAME --run $SEED --src_dir $RLU_DIR --dest_dir $NUMPY_DIR

echo "Preparing Replay Buffers..."
python3 experiments/atari/prepare_replay_buffers.py --game $GAME --run $SEED --src_dir $NUMPY_DIR --dest_dir $SLIM_DIR

echo "SEED processed successfully."
