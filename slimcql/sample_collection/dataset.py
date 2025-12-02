# inspired from batch_rl FixedReplayBuffer: https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/replay_memory/fixed_replay_buffer.py

import os
from functools import partial
from concurrent import futures
import numpy as np
import jax
import jax.numpy as jnp

from slimcql.sample_collection.replay_buffer import ReplayBuffer
from slimcql.sample_collection.samplers import Uniform, Prioritized


class Dataset:
    """Dataset is constructed from multiple ReplayBuffer objects in offline RL"""

    def __init__(
        self,
        data_dir: os.PathLike,
        n_buffers_to_load: int,
        single_replay_buffer_capacity: int,  # Controls the percentage of DQN data used
        sampling_distribution: Uniform | Prioritized,
        batch_size: int,
        stack_size: int,
        update_horizon: int,
        gamma: float,
        clipping: callable,
    ):

        self.data_dir = data_dir  # Path where the replay buffers for a given seed are logged
        self.n_buffers_to_load = n_buffers_to_load

        self.single_replay_buffer_args = (
            sampling_distribution,
            single_replay_buffer_capacity,
            batch_size,
            stack_size,
            update_horizon,
            gamma,
            clipping,
        )
        self.key = jax.random.PRNGKey(int(data_dir.split("/")[-1]))
        self.all_replay_checkpoints = sorted([int(checkpoint) for checkpoint in os.listdir(self.data_dir)])

    def load_dataset(self):
        """Samples and loads `n_buffers_to_load` ReplayBuffers into the RAM"""
        load_replay_key, self.key = jax.random.split(self.key)
        load_replay_checkpoints = jax.random.choice(
            load_replay_key, a=np.array(self.all_replay_checkpoints), shape=(self.n_buffers_to_load,), replace=False
        )

        def load_single_buffer(idx_buffer, load_replay_checkpoint):
            replay_buffer = ReplayBuffer(*self.single_replay_buffer_args)
            replay_buffer.load(self.data_dir, load_replay_checkpoint)
            return (idx_buffer, replay_buffer)

        with futures.ThreadPoolExecutor(max_workers=self.n_buffers_to_load) as thread_pool_executor:
            batches_futures = [
                thread_pool_executor.submit(load_single_buffer, idx_buffer, load_replay_checkpoint)
                for idx_buffer, load_replay_checkpoint in enumerate(load_replay_checkpoints)
            ]

        idx_and_buffers = []
        for f in batches_futures:
            idx_and_buffers.append(f.result())

        self.loaded_replay_buffers = [
            idx_and_buffer[1] for idx_and_buffer in sorted(idx_and_buffers, key=lambda x: x[0])
        ]

    def sample_single_batch(self, idx_batch, replay_index, batch_key):
        return (idx_batch, self.loaded_replay_buffers[replay_index].sample(key=batch_key))

    @partial(jax.jit, static_argnames=("self", "n_batches"))
    def sample_replay_indices_and_batch_keys(self, key, n_batches):
        replay_indices_key, batch_samples_keys = jax.random.split(key)
        replay_indices = jax.random.randint(replay_indices_key, (n_batches,), 0, self.n_buffers_to_load)
        batch_keys = jax.random.split(batch_samples_keys, n_batches)
        return replay_indices, batch_keys

    def sample(self, n_batches):
        sample_key, self.key = jax.random.split(self.key)
        replay_indices, batch_keys = self.sample_replay_indices_and_batch_keys(sample_key, n_batches)
        with futures.ThreadPoolExecutor(max_workers=n_batches) as thread_pool_executor:
            batches_futures = [
                thread_pool_executor.submit(
                    self.sample_single_batch, idx_batch, replay_indices[idx_batch], batch_keys[idx_batch]
                )
                for idx_batch in range(n_batches)
            ]

        idx_and_batches = []
        for f in batches_futures:
            idx_and_batches.append(f.result())

        # idx_and_batches is in the form (idx, (batch, (sample_key, importance_weight))
        # sort the list of idx_and_batches on their index (needed for determinism) and output (batch, (sample_key, importance_weight))
        batches_and_weights = [idx_and_batch[1] for idx_and_batch in sorted(idx_and_batches, key=lambda x: x[0])]

        batches = []
        sample_keys = []
        importance_weights = []

        for batch, (sample_key, importance_weight) in batches_and_weights:
            batches.append(batch)
            sample_keys.append(sample_key)
            if importance_weight is not None:
                importance_weights.append(importance_weight)

        # Convert the list of batch to a list single batch where each element
        # has the shape (n_batch, batch_size) + (element_shape,)
        return jax.tree.map(lambda *batch: jnp.stack(batch), *batches), (
            replay_indices,
            jnp.array(sample_keys),
            jnp.array(importance_weights) if len(importance_weights) > 0 else None,
        )

    def update(self, replay_index, keys, loss):
        # update function for Prioritized sampler
        self.loaded_replay_buffers[replay_index].update(keys, loss)

    def clear(self):
        for replay_buffer in self.loaded_replay_buffers:
            replay_buffer.clear()
