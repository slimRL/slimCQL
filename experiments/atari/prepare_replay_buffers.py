import os
import shutil
import gzip
from tqdm import tqdm
import numpy as np

from slimcql.sample_collection.replay_buffer import ReplayBuffer
from slimcql.sample_collection.samplers import Uniform


GAME = "VideoPinball"
RUN = 1
SRC_DIR = "experiments/atari/datasets/numpy_dataset"
DEST_DIR = "experiments/atari/datasets/slim_dataset"


for ckpt in tqdm(range(0, 50)):
    os.makedirs(f"{DEST_DIR}/{GAME}/{RUN}/{ckpt}", exist_ok=True)
    shutil.copyfile(f"{SRC_DIR}/{GAME}/{RUN}/observations_{ckpt}.gz", f"{DEST_DIR}/{GAME}/{RUN}/{ckpt}/observations.gz")

    arrays = {}
    for attr in ["observations", "actions", "rewards", "is_terminals", "episode_ends"]:
        arrays[attr] = np.load(gzip.GzipFile(fileobj=open(f"{SRC_DIR}/{GAME}/{RUN}/{attr}_{ckpt}.gz", "rb")))

    # Change replay buffer variables here for different `update_horizon`/`gamma`
    replay_buffer = ReplayBuffer(
        sampling_distribution=Uniform(RUN),
        max_capacity=1_000_000,
        batch_size=32,
        stack_size=4,
        update_horizon=1,
        gamma=0.99,
        clipping=None,
    )
    replay_buffer.save(
        f"{DEST_DIR}/{GAME}/{RUN}",
        ckpt,
        arrays["actions"],
        arrays["rewards"],
        arrays["is_terminals"],
        arrays["episode_ends"],
    )
    del replay_buffer
print(f"---------Game = {GAME}; Finished seed = {RUN}---------")
