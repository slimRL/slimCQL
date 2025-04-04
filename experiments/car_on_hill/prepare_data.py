import os
import gzip
import jax
import numpy as np

from slimdqn.sample_collection.replay_buffer import TransitionElement, ReplayBuffer
from slimdqn.sample_collection.samplers import UniformSamplingDistribution
from slimdqn.environments.car_on_hill import CAR_ON_HILL_DEFAULT_HORIZON


def generate_replay_buffer(p, env):

    rb_path = os.path.join(f"{p['save_path']}/../../../replay_buffer/uniform_{p['replay_buffer_capacity']}")
    rb = ReplayBuffer(
        sampling_distribution=UniformSamplingDistribution(),
        batch_size=-1,
        replay_buffer_capacity=p["replay_buffer_capacity"],
        stack_size=1,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        compress=False,
    )
    if not os.path.exists(rb_path):
        print("Replay buffer does not exist. Creating one...")

        key = jax.random.PRNGKey(seed=0)
        n_positive_reward_samples = 0

        env.reset()
        observations = []
        idx_observation = 0

        while rb.add_count < p["replay_buffer_capacity"]:
            observations.append(env.observation)

            key, sample_key = jax.random.split(key)
            action = jax.random.randint(sample_key, (), 0, env.n_actions).item()
            reward, absorbing = env.step(action)

            episode_end = absorbing or env.n_steps >= CAR_ON_HILL_DEFAULT_HORIZON
            rb.add(TransitionElement(idx_observation, action, reward, episode_end))

            idx_observation += 1
            n_positive_reward_samples += reward > 0

            if episode_end:
                env.reset()

        print(f"NOTE: Replay buffer filled with {n_positive_reward_samples} success samples.")

        # assert n_positive_reward_samples > 0, "Replay buffer has no success samples! Restart.."

        rb.save(checkpoint_dir=rb_path, idx_iteration=0)
        filename = rb._generate_filename(rb_path, "observation", 0, "gz")
        with open(filename, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb") as outfile:
                np.save(outfile, np.array(observations))
