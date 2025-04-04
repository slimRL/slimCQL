# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the standard DQN replay memory."""
import os
import collections
import operator
import typing
from typing import Any, Iterable
import gzip

import jax
import numpy as np
import numpy.typing as npt

from flax import struct
import snappy

from slimdqn.sample_collection import ReplayItemID


class TransitionElement(typing.NamedTuple):
    observation_index: int
    action: int
    reward: float
    is_terminal: bool
    episode_end: bool = False


class ReplayElement(struct.PyTreeNode):
    """A single replay transition element supporting compression."""

    state: npt.NDArray[np.float64]
    action: int
    reward: float
    next_state: npt.NDArray[np.float64]
    is_terminal: bool
    episode_end: bool = False

    @staticmethod
    def compress(buffer: npt.NDArray) -> npt.NDArray:
        if not buffer.flags["C_CONTIGUOUS"]:
            buffer = buffer.copy(order="C")
        compressed = np.frombuffer(snappy.compress(buffer), dtype=np.uint8)

        return np.array(
            (compressed, buffer.shape, buffer.dtype.str),
            dtype=[
                ("data", "u1", compressed.shape),
                ("shape", "i4", (len(buffer.shape),)),
                ("dtype", f"S{len(buffer.dtype.str)}"),
            ],
        )

    @staticmethod
    def uncompress(compressed: npt.NDArray) -> npt.NDArray:
        shape = tuple(compressed["shape"])
        dtype = compressed["dtype"].item()
        compressed_bytes = compressed["data"].tobytes()
        uncompressed = snappy.uncompress(compressed_bytes)
        return np.ndarray(shape=shape, dtype=dtype, buffer=uncompressed)

    def pack(self) -> "ReplayElement":
        return self.replace(
            state=ReplayElement.compress(self.state),
            next_state=ReplayElement.compress(self.next_state),
        )

    def unpack(self) -> "ReplayElement":
        return self.replace(
            state=ReplayElement.uncompress(self.state),
            next_state=ReplayElement.uncompress(self.next_state),
        )


class ReplayBuffer:
    def __init__(
        self,
        sampling_distribution,
        batch_size: int,
        replay_buffer_capacity: int,
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        compress: bool = True,
        clipping: callable = None,
    ):
        self.add_count = 0
        self._replay_buffer_capacity = replay_buffer_capacity
        self._compress = compress
        self._memory = collections.OrderedDict[ReplayItemID, ReplayElement]()

        self._sampling_distribution = sampling_distribution
        self._batch_size = batch_size

        self._stack_size = stack_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._clipping = clipping
        self._trajectory = collections.deque[TransitionElement](maxlen=self._update_horizon + self._stack_size)
        self.dataset_components = {
            "observation": [],
            "o_tm1_indices": [],
            "o_tm1_stack_size": [],
            "o_t_indices": [],
            "o_t_stack_size": [],
            "action": [],
            "reward": [],
            "is_terminal": [],
        }

    def _make_replay_element(self) -> ReplayElement:
        trajectory_len = len(self._trajectory)
        last_transition = self._trajectory[-1]
        # Check if we have a valid transition, i.e. we either
        #   1) have accumulated more transitions than the update horizon
        #   2) have a trajectory shorter than the update horizon, but the
        #      last element is terminal
        if not (trajectory_len > self._update_horizon or (trajectory_len > 1 and last_transition.is_terminal)):
            return None

        # Calculate effective horizon, this can differ from the update horizon
        # when we have n-step transitions where the last observation is terminal.
        effective_horizon = self._update_horizon
        if last_transition.is_terminal and trajectory_len <= self._update_horizon:
            effective_horizon = trajectory_len - 1

        # Initialize the slice for which this observation is valid.
        # The start index for o_tm1 is the start of the n-step trajectory.
        # The end index for o_tm1 is just moving over `stack size`.
        o_tm1_slice = slice(
            trajectory_len - effective_horizon - self._stack_size,
            trajectory_len - effective_horizon - 1,
        )
        # The action chosen will be the last transition in the stack.
        a_tm1 = self._trajectory[o_tm1_slice.stop].action

        # Initialize the slice for which this observation is valid.
        # The start index for o_t is just moving backwards `stack size`.
        # The end index for o_t is just the last index of the n-step trajectory.
        o_t_slice = slice(
            trajectory_len - self._stack_size,
            trajectory_len - 1,
        )
        # Terminal information will come from the last transition in the stack
        is_terminal = self._trajectory[o_t_slice.stop].is_terminal

        # Slice to accumulate n-step returns. This will be the end
        # transition of o_tm1 plus the effective horizon.
        # This might over-run the trajectory length in the case of n-step
        # returns where the last transition is terminal.
        gamma_slice = slice(
            o_tm1_slice.stop,
            o_tm1_slice.stop + self._update_horizon - 1,
        )
        assert o_t_slice.stop - o_tm1_slice.stop == effective_horizon
        assert o_t_slice.stop - 1 >= o_tm1_slice.stop

        # Now we'll iterate through the n-step trajectory and compute the
        # cumulant and insert the observations into the appropriate stacks
        r_t = 0.0
        o_tm1_indices = []
        o_t_indices = []
        for t, transition_t in enumerate(self._trajectory):
            # If we should be accumulating reward for an n-step return?
            if gamma_slice.start <= t <= gamma_slice.stop:
                r_t += transition_t.reward * (self._gamma ** (t - gamma_slice.start))

            # If we should be accumulating frames for the frame-stack?
            if o_tm1_slice.start <= t <= o_tm1_slice.stop:
                o_tm1_indices.append(transition_t.observation_index)
            if o_t_slice.start <= t <= o_t_slice.stop:
                o_t_indices.append(transition_t.observation_index)

        return (
            o_tm1_indices[0],
            len(o_tm1_indices),
            o_t_indices[0],
            len(o_t_indices),
            a_tm1,
            r_t,
            is_terminal,
        )

    def accumulate(self, transition: TransitionElement) -> Iterable[ReplayElement]:
        """Add a transition to the accumulator, maybe receive valid ReplayElements.

        If the transition has a terminal or end of episode signal, it will create a
        new trajectory and yield multiple elements.
        """
        self._trajectory.append(transition)

        if transition.is_terminal:
            while replay_element_components := self._make_replay_element():
                yield replay_element_components
                self._trajectory.popleft()
            self._trajectory.clear()
        else:
            if replay_element_components := self._make_replay_element():
                yield replay_element_components
            # If the transition truncates the trajectory then clear it
            if transition.episode_end:
                self._trajectory.clear()

    def add(self, transition: TransitionElement, **kwargs: Any) -> None:

        for replay_element_components in self.accumulate(transition):
            self.dataset_components["o_tm1_indices"].append(replay_element_components[0])
            self.dataset_components["o_tm1_stack_size"].append(replay_element_components[1])
            self.dataset_components["o_t_indices"].append(replay_element_components[2])
            self.dataset_components["o_t_stack_size"].append(replay_element_components[3])
            self.dataset_components["action"].append(replay_element_components[4])
            self.dataset_components["reward"].append(replay_element_components[5])
            self.dataset_components["is_terminal"].append(replay_element_components[6])
            self.add_count += 1

    def sample(self, key, size=None) -> ReplayElement | tuple[ReplayElement]:
        """Sample a batch of elements from the replay buffer."""
        assert self.add_count, ValueError("No samples in replay buffer!")

        if size is None:
            size = self._batch_size

        samples = self._sampling_distribution.sample(size, key)
        replay_elements = operator.itemgetter(*samples)(self._memory)
        if not isinstance(replay_elements, tuple):
            replay_elements = (replay_elements,)
        if self._compress:
            replay_elements = map(operator.methodcaller("unpack"), replay_elements)

        batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *replay_elements)
        return batch

    def clear(self) -> None:
        """Clear the replay buffer."""
        self.add_count = 0
        self._memory.clear()
        self._sampling_distribution.clear()
        self._trajectory.clear()

    def _generate_filename(self, checkpoint_dir, attr, idx_iteration, extension):
        return os.path.join(checkpoint_dir, str(idx_iteration), f"{attr}.{extension}")

    def save(self, checkpoint_dir, idx_iteration):

        self.dataset_components["o_tm1_indices"] = np.array(self.dataset_components["o_tm1_indices"], dtype=np.int32)
        self.dataset_components["o_tm1_stack_size"] = np.array(
            self.dataset_components["o_tm1_stack_size"], dtype=np.int8
        )
        self.dataset_components["o_t_indices"] = np.array(self.dataset_components["o_t_indices"], dtype=np.int32)
        self.dataset_components["o_t_stack_size"] = np.array(self.dataset_components["o_t_stack_size"], dtype=np.int8)
        self.dataset_components["action"] = np.array(self.dataset_components["action"], dtype=np.int32)
        self.dataset_components["reward"] = np.array(self.dataset_components["reward"], dtype=np.float32)
        self.dataset_components["is_terminal"] = np.array(self.dataset_components["is_terminal"], dtype=np.int8)

        os.makedirs(os.path.join(checkpoint_dir, str(idx_iteration)), exist_ok=True)
        for attr in [
            "o_tm1_indices",
            "o_tm1_stack_size",
            "o_t_indices",
            "o_t_stack_size",
            "action",
            "reward",
            "is_terminal",
        ]:
            filename = self._generate_filename(checkpoint_dir, attr, idx_iteration, "gz")
            with open(filename, "wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb") as outfile:
                    np.save(outfile, self.dataset_components[attr])

    def _load_transitions(self, replay_transitions_start_index=None):

        replay_transitions_end_index = len(self.dataset_components["action"])
        if replay_transitions_start_index is not None:
            replay_transitions_end_index = min(
                replay_transitions_start_index + self._replay_buffer_capacity, replay_transitions_end_index
            )
        else:
            replay_transitions_start_index = 0

        for attr in self.dataset_components:
            self.dataset_components[attr] = np.array(self.dataset_components[attr])

        for attr in [
            "o_tm1_indices",
            "o_tm1_stack_size",
            "o_t_indices",
            "o_t_stack_size",
            "action",
            "reward",
            "is_terminal",
        ]:
            self.dataset_components[attr] = self.dataset_components[attr][
                replay_transitions_start_index:replay_transitions_end_index
            ]
        self.dataset_components["observation"] = self.dataset_components["observation"][
            self.dataset_components["o_tm1_indices"].min() : self.dataset_components["o_t_indices"].max()
            + self.dataset_components["o_t_stack_size"].max()
            + 1
        ]

        for o_tm1_start_index, o_tm1_stack_size, o_t_start_index, o_t_stack_size, action, reward, is_terminal in zip(
            self.dataset_components["o_tm1_indices"],
            self.dataset_components["o_tm1_stack_size"],
            self.dataset_components["o_t_indices"],
            self.dataset_components["o_t_stack_size"],
            self.dataset_components["action"],
            self.dataset_components["reward"],
            self.dataset_components["is_terminal"],
        ):
            key = ReplayItemID(self.add_count)
            o_tm1 = np.zeros(shape=self.dataset_components["observation"][0].shape + (self._stack_size,))
            o_t = np.zeros(shape=self.dataset_components["observation"][0].shape + (self._stack_size,))

            o_tm1[..., :o_tm1_stack_size] = np.moveaxis(
                self.dataset_components["observation"][o_tm1_start_index : o_tm1_start_index + o_tm1_stack_size, ...],
                0,
                -1,
            )
            o_t[..., :o_t_stack_size] = np.moveaxis(
                self.dataset_components["observation"][o_t_start_index : o_t_start_index + o_t_stack_size, ...], 0, -1
            )
            replay_element = ReplayElement(o_tm1, action, reward, o_t, is_terminal)
            self._memory[key] = replay_element.pack() if self._compress else replay_element

            self._sampling_distribution.add(key)
            self.add_count += 1
            if self.add_count > self._replay_buffer_capacity:
                oldest_key, _ = self._memory.popitem(last=False)
                self._sampling_distribution.remove(oldest_key)
        del self.dataset_components["observation"]

    def load(self, checkpoint_dir, idx_iteration, replay_transitions_start_index):
        self.clear()
        self.dataset_components = {}

        for attr in [
            "observation",
            "o_tm1_indices",
            "o_tm1_stack_size",
            "o_t_indices",
            "o_t_stack_size",
            "action",
            "reward",
            "is_terminal",
        ]:
            filename = self._generate_filename(checkpoint_dir, attr, idx_iteration, "gz")
            with open(filename, "rb") as f:
                with gzip.GzipFile(fileobj=f) as infile:
                    self.dataset_components[attr] = np.load(infile)

        self._load_transitions(replay_transitions_start_index)
