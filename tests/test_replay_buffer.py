# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

from absl.testing import absltest
from absl.testing import parameterized
from absl import flags
from etils import epath
import msgpack
import numpy as np
import jax

from slimdqn.sample_collection import replay_buffer
from slimdqn.sample_collection.replay_buffer import ReplayElement, TransitionElement
from slimdqn.sample_collection import samplers


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32

flags.FLAGS(["--test_tmpdir", "/tmpdir"])


class ReplayBufferTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = epath.Path(self.create_tempdir("checkpoint").full_path)
        self._obs = np.ones((4, 3))

        self._sampling_distribution = samplers.UniformSamplingDistribution()

    def test_element_pack_unpack(self) -> None:
        """Simple test case that packs and unpacks a replay element."""
        state = np.zeros(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)
        next_state = np.ones(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)
        action = 1
        reward = 1.0
        episode_end = False

        element = ReplayElement(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            is_terminal=episode_end,
            episode_end=episode_end,
        )

        packed = element.pack()
        assert packed.action == action
        assert packed.reward == reward
        assert packed.is_terminal == packed.episode_end == episode_end

        unpacked = packed.unpack()
        assert unpacked.action == action
        assert unpacked.reward == reward
        assert unpacked.is_terminal == unpacked.episode_end == episode_end

        np.testing.assert_array_equal(unpacked.state, state)
        np.testing.assert_array_equal(unpacked.next_state, next_state)

    def testAddUpToCapacity(self):
        capacity = 10
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(),
            batch_size=BATCH_SIZE,
            replay_buffer_capacity=capacity,
            stack_size=STACK_SIZE,
            update_horizon=1,
            gamma=1.0,
            compress=False,
        )

        transitions = []
        for i in range(16):
            transitions.append(TransitionElement(i, i, i, False, False))
            rb.dataset_components["observation"].append(np.full(OBSERVATION_SHAPE, i))
            rb.add(transitions[-1])
        # Since we created the ReplayBuffer with a capacity of 10, it should have
        # gotten rid of the first 5 elements added.
        rb._load_transitions()
        self.assertLen(rb._memory, capacity)
        expected_keys = list(range(5, 5 + capacity))
        self.assertEqual(list(rb._memory.keys()), expected_keys)
        for i in expected_keys:
            self.assertEqual(rb._memory[i].action, transitions[i].action)
            self.assertEqual(rb._memory[i].reward, transitions[i].reward)
            self.assertEqual(rb._memory[i].is_terminal, int(transitions[i].is_terminal))
            self.assertEqual(rb._memory[i].episode_end, int(transitions[i].episode_end))

    def testNSteprewards(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(),
            batch_size=BATCH_SIZE,
            replay_buffer_capacity=10,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
            compress=False,
        )

        for i in range(50):
            # add non-terminating observations with reward 2
            rb.add(TransitionElement(i, 0, 2.0, False))
            rb.dataset_components["observation"].append(np.full(OBSERVATION_SHAPE, i))
        rb._load_transitions()

        test_key = jax.random.PRNGKey(0)
        for _ in range(100):
            batch_key, test_key = jax.random.split(test_key)
            batch = rb.sample(batch_key)
            # Make sure the total reward is reward per step x update_horizon.
            np.testing.assert_array_equal(batch.reward, np.ones(BATCH_SIZE) * 10.0)

    def testGetStack(self):
        zero_state = np.zeros(OBSERVATION_SHAPE + (3,))

        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(),
            batch_size=BATCH_SIZE,
            replay_buffer_capacity=50,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
            compress=False,
        )
        for i in range(11):
            rb.add(TransitionElement(i, 0, 0, False))
            rb.dataset_components["observation"].append(np.full(OBSERVATION_SHAPE, i))

        rb._load_transitions()
        # ensure that the returned shapes are always correct
        for i in rb._memory:
            np.testing.assert_array_equal(rb._memory[i].state.shape, OBSERVATION_SHAPE + (4,))

        # ensure that there is the necessary 0 padding
        state = rb._memory[0].state
        np.testing.assert_array_equal(zero_state, state[:, :, :3])

        # ensure that after the padding the contents are properly stored
        state = rb._memory[3].state
        for i in range(4):
            np.testing.assert_array_equal(np.full(OBSERVATION_SHAPE, i), state[:, :, i])

    def testSampleTransitionBatch(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(),
            batch_size=2,
            replay_buffer_capacity=10,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            compress=False,
        )
        num_adds = 50  # The number of transitions to add to the memory.

        # terminal transitions are not valid trajectories
        index_to_id = []
        for i in range(num_adds):
            terminal = i % 4 == 0  # Every 4 transitions is terminal.
            rb.add(TransitionElement(i, 0, 0, terminal, False))
            rb.dataset_components["observation"].append(np.full(OBSERVATION_SHAPE, i))

            index_to_id.append(i)

        rb._load_transitions()
        # Verify we sample the expected indices by using the same rng state.
        self._rng_key = jax.random.PRNGKey(seed=0)
        key, self._rng_key = jax.random.split(self._rng_key)
        indices = jax.random.randint(
            key, shape=(rb._batch_size,), minval=0, maxval=len(rb._sampling_distribution._index_to_key)
        )

        # Replicating the formula used above to determine what transitions are terminal
        expected_terminal = np.array(
            [int(((index_to_id[rb._sampling_distribution._index_to_key[i]]) % 4) == 0) for i in indices]
        )
        batch = rb.sample(key, size=len(indices))
        np.testing.assert_array_equal(batch.action, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.reward, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.is_terminal, expected_terminal)

    def testSamplingWithTerminalInTrajectory(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(),
            batch_size=2,
            replay_buffer_capacity=10,
            stack_size=1,
            update_horizon=3,
            gamma=1.0,
            compress=False,
        )
        for i in range(rb._replay_buffer_capacity):
            rb.add(
                TransitionElement(
                    i,
                    action=i * 2,
                    reward=i,
                    is_terminal=i == 3,
                    episode_end=False,
                )
            )
            rb.dataset_components["observation"].append(np.full(OBSERVATION_SHAPE, i))

        rb._load_transitions()
        # Verify we sample the expected indices, using the same rng.
        self._rng_key = jax.random.PRNGKey(seed=0)
        key, self._rng_key = jax.random.split(self._rng_key)
        indices = jax.random.randint(key, shape=(5,), minval=0, maxval=rb.add_count)

        batch = rb.sample(key, size=5)

        # Since index 3 is terminal, it will not be a valid transition so renumber.
        expected_states = np.array([np.full(OBSERVATION_SHAPE + (1,), i) for i in indices])
        expected_actions = np.array([i * 2 for i in indices])
        # The reward in the replay buffer will be (an asterisk marks the terminal
        # state):
        #   [0 1 2 3* 4 5 6 7 8 9]
        # Since we're setting the update_horizon to 3, the accumulated trajectory
        # reward starting at each of the replay buffer positions will be (a '_'
        # marks an invalid transition to sample):
        #   [3 6 5 _ 15 18 21 24]
        expected_rewards = np.array([3, 6, 5, 3, 15, 18, 21, 24])
        # Because update_horizon = 3, indices 0, 1 and 2 include terminal.
        expected_terminals = np.array([0, 1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(batch.state, expected_states)
        np.testing.assert_array_equal(batch.action, expected_actions)
        np.testing.assert_array_equal(batch.reward, expected_rewards[indices])
        np.testing.assert_array_equal(batch.is_terminal, expected_terminals[indices])

    def testKeyMappingsForSampling(self):
        capacity = 10
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(),
            batch_size=BATCH_SIZE,
            replay_buffer_capacity=capacity,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            compress=False,
        )
        sampler = rb._sampling_distribution

        for i in range(capacity + 1):
            rb.add(TransitionElement(i, i, i, False, False))
            rb.dataset_components["observation"].append(np.full(OBSERVATION_SHAPE, i))

        rb._load_transitions()
        # While we haven't overwritten any elements we should have
        # global indices as being equivalent to local indices
        for i in range(capacity):
            self.assertIn(i, sampler._key_to_index)
            index = sampler._key_to_index[i]
            self.assertEqual(i, index)
            self.assertEqual(i, sampler._index_to_key[index])
