# inspired from batch_rl FixedReplayBuffer: https://github.com/google-research/batch_rl/blob/master/batch_rl/fixed_replay/replay_memory/fixed_replay_buffer.py

"""Logged Replay Buffer."""

import os
import collections
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
    ):  # to change between exploratory vs. expert data
        # For individual ReplayBuffer object parameters
        self._args = args
        self._kwargs = kwargs

        self.data_dir = data_dir  # path where the replay buffers are logged

        self.replay_checkpoint = replay_checkpoint  # to load a specific checkpoint replay buffer
        self.n_buffers_to_load = n_buffers_to_load

        self._loaded_buffers = False

        assert not self.replay_checkpoint or (
            self.replay_checkpoint and self.n_buffers_to_load == 1
        ), "When providing a checkpoint, n_buffers_to_load should be 1"

        self._replay_indices = self._get_checkpoint_indices(replay_file_start_index, replay_file_end_index)

        while not self._loaded_buffers:
            if replay_checkpoint is not None:
                self.load_single_buffer(replay_checkpoint)
            else:
                self._load_replay_buffers()

    def load_single_buffer(self, checkpoint):
        """Load a specific checkpoint"""
        try:
            replay_buffer = ReplayBuffer(*self._args, **self._kwargs)
            replay_buffer.load(self._data_dir, checkpoint)
            print(len(replay_buffer._memory))
            # check that load loads all 1M transitions (irrespective of replay_capacity value)
            # if replay capacity is less than a million, need to take only [replay_transitions_start_index: replay_transitions_start_index+replay_capacity+stack_size]

        except Exception:
            return None
        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]
            self._loaded_buffers = True
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
            self._replay_checkpoint = replay_ckpts[0]
        return replay_ckpts

    def _load_replay_buffers(self):
        """Loads multiple checkpoints into a list of replay buffers"""
        if not self._loaded_buffers:
            replay_ckpts = np.random.choice(self._replay_indices, self.n_buffers_to_load, replace=False)
            self._replay_buffers = []
            with futures.ThreadPoolExecutor(max_workers=self.n_buffers_to_load) as thread_pool_executor:
                replay_futures = [thread_pool_executor.submit(self.load_single_buffer, ckpt) for ckpt in replay_ckpts]
            for f in replay_futures:
                replay_buffer = f.result()
                if replay_buffer is not None:
                    self._replay_buffers.append(replay_buffer)
            if len(self._replay_buffers):
                self._loaded_buffers = True

    def sample_transition_batch(self, batch_size=None, indices=None):
        buffer_index = np.random.randint(self._num_replay_buffers)
        return self._replay_buffers[buffer_index].sample_transition_batch(batch_size=batch_size, indices=indices)

    def reload_data(self):
        if self._replay_checkpoint is None:
            self._loaded_buffers = False
            self._load_replay_buffers()
