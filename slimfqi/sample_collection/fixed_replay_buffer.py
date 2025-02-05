# inspired from batch_rl FixedReplayBuffer: https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/replay_memory/fixed_replay_buffer.py

"""Logged Replay Buffer."""

import os
from concurrent import futures
import numpy as np

from slimfqi.sample_collection.replay_buffer import ReplayBuffer


class FixedReplayBuffer(object):
    """For working with multiple ReplayBuffer objects in offline scenario"""

    def __init__(
        self,
        data_dir,
        n_buffers_to_load,
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

        assert not self.replay_checkpoint or (
            self.replay_checkpoint and self.n_buffers_to_load == 1
        ), "When providing a checkpoint, n_buffers_to_load should be 1"

        self._replay_indices = self._get_checkpoint_indices(replay_file_start_index, replay_file_end_index)

        if replay_checkpoint is not None:
            self.load_single_buffer(replay_checkpoint)

    def load_single_buffer(self, checkpoint):
        """Load a specific checkpoint"""

        replay_buffer = ReplayBuffer(*self._args, **self._kwargs)
        replay_buffer.load(self.data_dir, checkpoint)
        print(len(replay_buffer._memory))
        print(replay_buffer._replay_buffer_capacity)

        replay_buffer._memory = replay_buffer._memory[
            self.replay_transitions_start_index : self.replay_transitions_start_index + replay_buffer._replay_buffer_capacity
        ].copy()
        print(len(replay_buffer._memory))
        print(replay_buffer._replay_buffer_capacity)

        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]
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
        replay_ckpts = np.random.choice(self._replay_indices, self.n_buffers_to_load, replace=False)
        self._replay_buffers = []
        with futures.ThreadPoolExecutor(max_workers=self.n_buffers_to_load) as thread_pool_executor:
            replay_futures = [thread_pool_executor.submit(self.load_single_buffer, ckpt) for ckpt in replay_ckpts]
        for f in replay_futures:
            replay_buffer = f.result()
            if replay_buffer is not None:
                self._replay_buffers.append(replay_buffer)

    def sample(self, size=None):
        buffer_index = np.random.randint(len(self._replay_buffers))
        return self._replay_buffers[buffer_index].sample(size=size)

    def reload_data(self):
        if self.replay_checkpoint is None:
            self._load_replay_buffers()
