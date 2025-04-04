import os
import shutil
import time
import gzip
from tqdm import tqdm
import numpy as np

from slimdqn.sample_collection.replay_buffer import TransitionElement, ReplayBuffer
from slimdqn.sample_collection.samplers import UniformSamplingDistribution

STORE_FILENAME_PREFIX = "$store$_"
ELEMS = ["observation", "action", "reward", "terminal"]

GAME = "Pong"
seed = 1
source_data_dir = "/home/yogesh/RL/dqn-dataset"
destination_data_dir = "/home/yogesh/RL/offline-dataset"


def transform_single_checkpoint(idx_checkpoint, data_dir):
    data = {}

    for elem in ELEMS:
        filename = f"{data_dir}/{STORE_FILENAME_PREFIX}{elem}_ckpt.{idx_checkpoint}.gz"
        with open(filename, "rb") as f:
            with gzip.GzipFile(fileobj=f) as infile:
                data[elem] = np.load(infile)

    rb = ReplayBuffer(
        sampling_distribution=UniformSamplingDistribution(),
        batch_size=32,
        replay_buffer_capacity=1_000_000,
        stack_size=4,
        update_horizon=1,
        gamma=0.99,
        compress=True,
    )

    for index, (_, action, reward, terminal) in enumerate(
        zip(data["observation"], data["action"], data["reward"], data["terminal"])
    ):
        rb.add(TransitionElement(index, action, reward, terminal))

    return rb


data_dir = f"{source_data_dir}/{GAME}/{seed}/replay_logs"
destination_dir = f"{destination_data_dir}/{GAME}/{seed}"
os.makedirs(destination_dir)
for idx_iteration in tqdm(range(0, 50)):
    os.makedirs(os.path.join(destination_dir, str(idx_iteration)))
    shutil.copyfile(
        f"{data_dir}/{STORE_FILENAME_PREFIX}observation_ckpt.{idx_iteration}.gz",
        f"{destination_dir}/{idx_iteration}/observation.gz",
    )
    rb = transform_single_checkpoint(idx_iteration, data_dir)
    rb.save(checkpoint_dir=destination_dir, idx_iteration=idx_iteration)
    del rb
    print(f"---------Game = {GAME}, Finished seed = {seed}, ckpt = {idx_iteration}---------")
