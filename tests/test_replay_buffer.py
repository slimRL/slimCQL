# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replaymemory/replay_buffer_test.py
from absl.testing import parameterized
import os
import gzip
import shutil
import numpy as np
import jax

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement, TransitionElement
from slimdqn.sample_collection.samplers import Uniform


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4


class ReplayBufferTest(parameterized.TestCase):

    def test_element_pack_unpack(self) -> None:
        """Pack and unpack a replay element."""
        state = np.zeros(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)
        next_state = np.ones(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)

        element = ReplayElement(state=state, action=0, reward=0, next_state=next_state, is_terminal=False)
        unpacked = element.pack().unpack()

        np.testing.assert_array_equal(unpacked.state, state)
        np.testing.assert_array_equal(unpacked.next_state, next_state)
        assert unpacked.action == 0
        assert unpacked.reward == 0
        assert unpacked.is_terminal == False

    def testSaveAndLoad(self):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/atari/datasets/_test_save")
        rb = ReplayBuffer(
            sampling_distribution=Uniform(),
            max_capacity=10,
            batch_size=32,
            stack_size=STACK_SIZE,
            update_horizon=1,
            gamma=1.0,
            clipping=None,
        )

        rb.save(save_path, 1, np.arange(16), np.arange(16), np.full((16,), False), np.full((16,), False))
        np.save(
            gzip.GzipFile(fileobj=open(os.path.join(save_path, "1/observations.gz"), "wb"), mode="wb"),
            np.array([np.full(OBSERVATION_SHAPE, i) for i in range(16)]),
        )
        rb.load(save_path, 1)
        # Since we created the ReplayBuffer with a capacity of 10, it should have first 10 elements added
        self.assertLen(rb.memory, 10)
        self.assertEqual(list(rb.memory.keys()), list(range(0, 10)))

        transitions = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for i in range(10):
            np.testing.assert_array_equal(
                ReplayElement.uncompress(rb.memory[i].state)[0, 0, :], transitions[i : i + STACK_SIZE]
            )
            np.testing.assert_array_equal(
                ReplayElement.uncompress(rb.memory[i].next_state)[0, 0, :], transitions[i + 1 : i + 1 + STACK_SIZE]
            )
            self.assertEqual(rb.memory[i].action, i)
            self.assertEqual(rb.memory[i].reward, i)
            self.assertEqual(rb.memory[i].is_terminal, False)

        shutil.rmtree(save_path)

    def testNSteprewards(self):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/atari/datasets/_test_save")
        rb = ReplayBuffer(
            sampling_distribution=Uniform(),
            max_capacity=10,
            batch_size=32,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
            clipping=None,
        )

        rb.save(save_path, 1, np.zeros((50,)), np.full((50,), 2.0), np.full((50,), False), np.full((50,), False))
        np.save(
            gzip.GzipFile(fileobj=open(os.path.join(save_path, "1/observations.gz"), "wb"), mode="wb"),
            np.array([np.full(OBSERVATION_SHAPE, i) for i in range(50)]),
        )
        rb.load(save_path, 1)

        batch, _ = rb.sample(jax.random.PRNGKey(np.random.randint(1000)))
        # Make sure the total reward is reward per step x update_horizon.
        np.testing.assert_array_equal(batch.reward, np.ones(32) * 10.0)
        shutil.rmtree(save_path)

    def testSampleTransitionBatch(self):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiments/atari/datasets/_test_save")
        rb = ReplayBuffer(
            sampling_distribution=Uniform(),
            max_capacity=100,
            batch_size=32,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            clipping=None,
        )

        rb.save(save_path, 1, np.zeros((21,)), np.zeros((21,)), np.arange(21) % 4 == 0, np.full((21,), False))
        np.save(
            gzip.GzipFile(fileobj=open(os.path.join(save_path, "1/observations.gz"), "wb"), mode="wb"),
            np.array([np.full(OBSERVATION_SHAPE, i) for i in range(21)]),
        )
        rb.load(save_path, 1)

        # Verify we sample the expected indices by using the same rng state.
        self.sample_key = jax.random.PRNGKey(np.random.randint(1000))
        indices = jax.random.randint(
            self.sample_key, shape=(32,), minval=0, maxval=len(rb.sampling_distribution.index_to_key)
        )

        expected_states = [
            ReplayElement.uncompress(rb.memory[rb.sampling_distribution.index_to_key[i]].state) for i in indices
        ]
        expected_next_states = [
            ReplayElement.uncompress(rb.memory[rb.sampling_distribution.index_to_key[i]].next_state) for i in indices
        ]
        expected_terminal = [rb.memory[rb.sampling_distribution.index_to_key[i]].is_terminal for i in indices]

        batch, _ = rb.sample(self.sample_key, 32)
        np.testing.assert_array_equal(batch.state, expected_states)
        np.testing.assert_array_equal(batch.action, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.reward, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.next_state, expected_next_states)
        np.testing.assert_array_equal(batch.is_terminal, expected_terminal)
        shutil.rmtree(save_path)
