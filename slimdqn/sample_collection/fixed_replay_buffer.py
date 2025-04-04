# inspired from batch_rl FixedReplayBuffer: https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/replay_memory/fixed_replay_buffer.py

"""Logged Replay Buffer."""

import os
from functools import partial
from concurrent import futures
import operator
import numpy as np
import jax

from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.samplers import UniformSamplingDistribution


class FixedReplayBuffer:
    """For working with multiple ReplayBuffer objects in offline scenario"""

    def __init__(
        self,
        data_dir,
        n_buffers_to_load,
        sampler_seed,
        *args,
        replay_checkpoint=None,
        replay_file_start_index=0,
        replay_file_end_index=None,
        replay_transitions_start_index=0,
        **kwargs
    ):
        self._args = args
        self._kwargs = kwargs

        self.data_dir = data_dir  # path where the replay buffers are logged

        self.replay_checkpoint = replay_checkpoint  # to load a specific checkpoint replay buffer
        self.n_buffers_to_load = n_buffers_to_load
        self.replay_transitions_start_index = replay_transitions_start_index
        self.key = jax.random.PRNGKey(seed=sampler_seed)

        assert not self.replay_checkpoint or (
            self.replay_checkpoint and self.n_buffers_to_load == 1
        ), "When providing a checkpoint, n_buffers_to_load should be 1"

        self._replay_indices = self._get_checkpoint_indices(replay_file_start_index, replay_file_end_index)

        if replay_checkpoint is not None:
            self._replay_buffers = [self.load_single_buffer(replay_checkpoint)]

    def load_single_buffer(self, checkpoint):
        """Load a specific checkpoint"""

        replay_buffer = ReplayBuffer(sampling_distribution=UniformSamplingDistribution(), *self._args, **self._kwargs)
        replay_buffer.load(self.data_dir, checkpoint, self.replay_transitions_start_index)
        return replay_buffer

    def _get_checkpoint_indices(self, replay_file_start_index, replay_file_end_index):
        """Get subset of all checkpoints to sample from"""
        all_ckpts = os.listdir(self.data_dir)
        all_ckpts = [int(x) for x in all_ckpts]
        all_ckpts = sorted(all_ckpts)
        if replay_file_end_index is None:
            replay_file_end_index = len(all_ckpts)
        replay_ckpts = all_ckpts[replay_file_start_index:replay_file_end_index]
        if len(replay_ckpts) == 1:
            self.replay_checkpoint = replay_ckpts[0]
        return replay_ckpts

    def _load_replay_buffers(self):
        """Loads multiple checkpoints into a list of replay buffers"""
        replay_ckpts_key, self.key = jax.random.split(self.key)
        replay_ckpts = jax.random.choice(
            replay_ckpts_key, a=np.array(self._replay_indices), shape=(self.n_buffers_to_load,), replace=False
        )
        self._replay_buffers = [self.load_single_buffer(ckpt) for ckpt in replay_ckpts]

    def sample_single_batch(self, idx_batch, rb_index, sample_indices_key):
        return (idx_batch, self._replay_buffers[rb_index].sample(key=sample_indices_key))

    @partial(jax.jit, static_argnames=("self", "n_batches"))
    def sample_replay_indices_and_batch_keys(self, key, n_batches):
        batches_key, sample_indices_key = jax.random.split(key)
        replay_indices = jax.random.randint(batches_key, (n_batches,), 0, self.n_buffers_to_load)
        batch_keys = jax.random.split(sample_indices_key, n_batches)
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

        # sort the list of idx_and_batches from their index and output the batches only (idx_and_batches[1])
        return list(map(lambda x: x[1], sorted(idx_and_batches, key=lambda x: x[0])))

    def reload_data(self):
        if self.replay_checkpoint is None:
            self._load_replay_buffers()

    def clear(self):
        if self.replay_checkpoint is None:
            for replay_buffer in self._replay_buffers:
                replay_buffer.clear()
