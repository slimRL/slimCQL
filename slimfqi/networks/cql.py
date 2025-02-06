from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimfqi.networks.architectures.dqn import DQNNet
from slimfqi.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class CQL:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        alpha_cql: float,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, architecture_type, n_actions)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.alpha_cql = alpha_cql

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample()

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        q_values = jax.vmap(self.compute_q, in_axes=(None, 0))(params, samples)
        q_targets = jax.vmap(self.compute_target, in_axes=(None, 0))(params_target, samples)
        chosen_actions = jax.vmap(lambda x: x.action)(samples)
        replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, chosen_actions)

        mean_td_loss = jnp.mean(jnp.square(q_targets - replay_chosen_q))
        cql_loss = jnp.mean(jax.scipy.special.logsumexp(q_values, axis=-1) - replay_chosen_q)
        return mean_td_loss + self.alpha_cql * cql_loss

    def compute_q(self, params: FrozenDict, sample):
        return self.network.apply(params, sample.state)

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state))

    def get_model(self):
        return {"params": self.params}
