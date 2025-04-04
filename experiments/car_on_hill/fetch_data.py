import os
import json
import optax
import jax
import jax.numpy as jnp
import numpy as np

from slimdqn.sample_collection.replay_buffer import TransitionElement
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.environments.car_on_hill import CAR_ON_HILL_DEFAULT_HORIZON


def update_replay_buffer(key, env, rb: ReplayBuffer, p):
    rb_path = os.path.join(p["save_path"], f"replay_buffer/{p['seed']}")
    if os.path.exists(rb_path):
        print("Replay buffer already exists. Loading...")
        rb.load(checkpoint_dir=rb_path, iteration_number=0)

    else:
        n_positive_reward_samples = 0

        env.reset()
        for _ in range(p["replay_buffer_capacity"]):
            key, sample_key = jax.random.split(key)
            action = jax.random.randint(sample_key, (), 0, env.n_actions).item()
            obs = env.observation
            reward, absorbing = env.step(action)

            episode_end = absorbing or env.n_steps >= CAR_ON_HILL_DEFAULT_HORIZON
            rb.add(
                TransitionElement(
                    observation=obs,
                    action=action,
                    reward=reward if rb._clipping is None else rb._clipping(reward),
                    is_terminal=absorbing,
                    episode_end=episode_end,
                )
            )
            n_positive_reward_samples += reward > 0

            if episode_end:
                env.reset()

        print(f"NOTE: Replay buffer filled with {n_positive_reward_samples} success samples.")
        rb.save(checkpoint_dir=rb_path, iteration_number=0)
