from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.algorithms.architectures.dqn import DQNNet
from slimdqn.sample_collection.dataset import Dataset
from slimdqn.sample_collection.replay_buffer import ReplayElement


class DQN:
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
        target_update_period: int,
        adam_eps: float = 1e-8,
    ):
        self.network = DQNNet(features, architecture_type, n_actions)
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.target_update_period = target_update_period
        self.cumulated_loss = 0

    @partial(jax.jit, static_argnames="self")
    def apply_multiple_updates(self, params, params_target, optimizer_state, batches):
        # Applies gradient updates over multiple batches using jax.lax.scan
        def apply_single_update(state, batch):
            params, optimizer_state, loss = self.learn_on_batch(state[0], params_target, state[1], batch)
            return (params, optimizer_state), loss

        (final_params, final_optimizer_state), losses = jax.lax.scan(
            apply_single_update, (params, optimizer_state), batches
        )
        return final_params, final_optimizer_state, jnp.sum(losses)

    def n_updates_online_params(self, n_updates: int, dataset: Dataset):
        batches, _ = dataset.sample(n_updates)

        self.params, self.optimizer_state, loss = self.apply_multiple_updates(
            self.params, self.target_params, self.optimizer_state, batches
        )
        self.cumulated_loss += loss

    def update_target_params(self):
        self.target_params = self.params.copy()

        logs = {"loss": self.cumulated_loss / self.target_update_period}
        self.cumulated_loss = 0

        return logs

    def learn_on_batch(self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        return jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples).mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.network.apply(params, sample.state)[sample.action]
        return jnp.square(q_value - target)

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
