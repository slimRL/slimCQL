import os
import sys

import jax

from experiments.base.fqi import train
from experiments.base.utils import prepare_logs
from experiments.car_on_hill.utils import update_replay_buffer
from slimfqi.environments.car_on_hill import CarOnHill
from slimfqi.networks.dqn import DQN
from slimfqi.sample_collection.replay_buffer import ReplayBuffer
from slimfqi.sample_collection.samplers import UniformSamplingDistribution


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (
        os.path.abspath(__file__).split("/")[-2],
        os.path.abspath(__file__).split("/")[-1][:-3],
    )
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key, sample_key = jax.random.split(jax.random.PRNGKey(p["seed"]), 3)

    env = CarOnHill()
    rb = ReplayBuffer(
        sampling_distribution=UniformSamplingDistribution(p["seed"]),
        batch_size=p["batch_size"],
        max_capacity=p["replay_buffer_capacity"],
        stack_size=1,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        compress=True,
    )
    update_replay_buffer(sample_key, env, rb, p)

    agent = DQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=-1,
        target_update_frequency=-1,
    )

    # need a function to store (update) the replay buffer

    train(train_key, p, agent, rb)


if __name__ == "__main__":
    run()
