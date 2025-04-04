# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/samplers_test.py
"""Testing samplers."""

from absl.testing import absltest
from absl.testing import parameterized
from slimdqn.sample_collection import samplers
import numpy as np
import jax


class SamplersTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.uniform_sampler = samplers.UniformSamplingDistribution()
        self.prioritized_sampler = samplers.PrioritizedSamplingDistribution(replay_buffer_capacity=10)

    def test_prioritized_sample(self):
        keys = [0, 1, 2, 3, 4]
        priorities = [1.0, 2.0, 3.0, 4.0, 0.0]

        for key, priority in zip(keys, priorities):
            self.prioritized_sampler.add(key, priority=priority)

        # test if zero priority absent
        samples = self.prioritized_sampler.sample(5, jax.random.PRNGKey(0))
        np.testing.assert_array_less(samples, 4)

        self.prioritized_sampler.update(keys=np.array([2, 3]), priorities=np.array([0.0, 0.0]))

        # test if priority updated properly
        samples = self.prioritized_sampler.sample(5, jax.random.PRNGKey(0))
        np.testing.assert_array_less(samples, 2)

        self.prioritized_sampler.remove(0)
        samples = self.prioritized_sampler.sample(5, jax.random.PRNGKey(0))
        np.testing.assert_array_almost_equal(samples, 1)

    def test_clear_uniform_sampler(self):
        self.uniform_sampler.add(1)
        self.assertNotEmpty(self.uniform_sampler._key_to_index)
        self.assertNotEmpty(self.uniform_sampler._index_to_key)
        self.uniform_sampler.clear()
        self.assertEmpty(self.uniform_sampler._key_to_index)
        self.assertEmpty(self.uniform_sampler._index_to_key)

    def test_clear_prioritized_sampler(self):
        for index in range(3):
            self.prioritized_sampler.add(index, priority=1.0)
        self.assertNotEmpty(self.prioritized_sampler._key_to_index)
        self.assertNotEmpty(self.prioritized_sampler._index_to_key)
        self.prioritized_sampler.clear()
        self.assertEmpty(self.prioritized_sampler._key_to_index)
        self.assertEmpty(self.prioritized_sampler._index_to_key)
        self.assertEqual(self.prioritized_sampler._sum_tree.root, 0.0)
        for index in range(3):
            self.assertEqual(self.prioritized_sampler._sum_tree.get(index), 0.0)


if __name__ == "__main__":
    absltest.main()
