import os
import gzip
import numpy as np

from slimfqi.sample_collection.replay_buffer import TransitionElement, ReplayBuffer
from slimfqi.sample_collection.samplers import UniformSamplingDistribution

STORE_FILENAME_PREFIX = '$store$_'
ELEMS = ['observation', 'action', 'reward', 'terminal']

GAME='Pong'
seed=5


def transform_single_checkpoint(idx_checkpoint, data_dir):
    data = {}
    
    for elem in ELEMS:
        filename = f'{data_dir}/{STORE_FILENAME_PREFIX}{elem}_ckpt.{idx_checkpoint}.gz'
        with open(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as infile:
                data[elem] = np.load(infile)
    
    rb = ReplayBuffer(sampling_distribution=UniformSamplingDistribution(seed=0),
                    batch_size=32,
                    max_capacity=1_000_000,
                    stack_size=4,
                    update_horizon=1,
                    gamma=0.99,
                    checkpoint_duration=1,
                    compress=True)
    
    for observation, action, reward, terminal in zip(data["observation"], data["action"], data["reward"], data["terminal"]):
        rb.add(TransitionElement(observation, action, reward, terminal))
        
    return rb
    
data_dir = f'/home/stud_tripathi/rl/dqn_dataset/{GAME}/{seed}/replay_logs'
destination_dir = f'/home/stud_tripathi/rl/slimdqn_dataset/{GAME}/{seed}'
os.makedirs(destination_dir)
for idx_checkpoint in range(0, 50):
    
    rb = transform_single_checkpoint(idx_checkpoint, data_dir)
    rb.save(checkpoint_dir=destination_dir, iteration_number=idx_checkpoint)
    del rb
    print(f"---------Finished seed = {seed}, ckpt = {idx_checkpoint}---------")
    