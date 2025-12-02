# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/samplers.py
"""Uniform and Prioritized sampling distributions."""

import numpy as np
import jax

from slimcql.sample_collection.sum_tree import SumTree


class Uniform:
    def __init__(self):
        self.uniform_sampler = jax.jit(
            lambda key, size, max_size: jax.random.randint(key, shape=(int(size),), minval=0, maxval=max_size),
            static_argnames="size",
        )

        # This is maintained to enable using `rng.integers` while sampling
        self.index_to_key = []
        # This intertwined mapping with `index_to_key` is needed for efficient O(1) pop
        # instead of O(n) pop from `index_to_key` list
        self.key_to_index = {}

    def add(self, key):
        self.key_to_index[key] = len(self.index_to_key)
        self.index_to_key.append(key)

    def remove(self, key):
        # To remove `key` from the sampler in O(1), we do:
        # (1) Fetch the index of `key` in the `index_to_key` list using `key_to_index` dictionary
        # (2) Swap the last element in `index_to_key` list with `key` present at the fetched `index`
        #     (`key` becomes the last element in `index_to_key` list by this operation)
        # (3) Update the `key_to_index` dictionary to reflect the effect of swapping
        # (4) Pop `key` in O(1) from `index_to_key` list (as it is the last element), and from `key_to_index` dictionary
        index = self.key_to_index[key]
        self.index_to_key[index], self.index_to_key[-1] = (self.index_to_key[-1], self.index_to_key[index])
        self.key_to_index[self.index_to_key[index]] = index
        self.key_to_index.pop(self.index_to_key.pop())

    def sample(self, samples_key, size):
        indices = np.asarray(self.uniform_sampler(samples_key, size, len(self.index_to_key)))
        return np.array([self.index_to_key[index] for index in indices], dtype=np.int32), None

    def clear(self) -> None:
        self.index_to_key.clear()
        self.key_to_index.clear()


class Prioritized(Uniform):
    def __init__(self, max_capacity: int, alpha: float = 1.0):

        self.max_capacity = max_capacity
        self.alpha = alpha
        self.sum_tree = SumTree(self.max_capacity)

        self.prioritized_sampler = jax.jit(
            lambda key, maxval, size: jax.random.uniform(key, minval=0.0, maxval=maxval, shape=(int(size),)),
            static_argnames="size",
        )

        super().__init__()

    def add(self, key):
        super().add(key)
        self.sum_tree.set(self.key_to_index[key], self.sum_tree.max_recorded_priority)

    def update(self, keys, loss):
        keys = np.atleast_1d(keys).astype(np.int32)
        self.sum_tree.set([self.key_to_index[key] for key in keys], loss**self.alpha)

    def remove(self, key) -> None:
        index = self.key_to_index[key]
        last_index = len(self.index_to_key) - 1
        if index == last_index:
            # If index and last_index are the same, simply set the priority to 0.0.
            self.sum_tree.set(index, 0.0)
        else:
            # Otherwise, swap priorities of `index` with `last_index`, and set priority of `last_index` to 0.
            # This keeps the remove() on sum_tree consistent with remove() on `index_to_key`.
            self.sum_tree.set([index, last_index], [self.sum_tree.get(last_index), 0.0])
        super().remove(key)

    def sample(self, key, size):
        if self.sum_tree.root == 0.0:
            return super().sample(size)

        targets = self.prioritized_sampler(key, self.sum_tree.root, size)
        indices = self.sum_tree.query(targets)
        probabilities = self.sum_tree.get(indices)
        importance_weights = 1.0 / np.sqrt(probabilities + 1e-10)  # beta = 0.5
        importance_weights /= np.max(importance_weights)

        return np.array([self.index_to_key[index] for index in indices], dtype=np.int32), importance_weights

    def clear(self):
        self.sum_tree.clear()
        super().clear()
