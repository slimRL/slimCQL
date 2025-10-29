import os
import sys

import jax
import numpy as np

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.atari import AtariEnv
from slimdqn.algorithms.dqn import DQN
from slimdqn.sample_collection.dataset import Dataset
from slimdqn.sample_collection.samplers import Uniform


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3])
    p = prepare_logs(env_name, algo_name, argvs)

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    dataset = Dataset(
        data_dir=f"{p['data_dir']}/{p['experiment_name'].split('_')[-1]}/{p['seed']}",
        n_buffers_to_load=p["n_buffers_to_load"],
        single_replay_buffer_capacity=p["replay_buffer_capacity"],
        sampling_distribution=Uniform(),
        batch_size=p["batch_size"],
        stack_size=4,
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: np.clip(x, -1, 1),
    )
    agent = DQN(
        jax.random.PRNGKey(p["seed"]),
        env.observation_shape,
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        target_update_period=p["target_update_period"],
        adam_eps=1.5e-4,
    )
    train(p, agent, dataset)


if __name__ == "__main__":
    run()
