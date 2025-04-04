import os
import sys

import jax
import numpy as np

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.car_on_hill import CarOnHill
from slimdqn.networks.cql import CQL
from slimdqn.sample_collection.fixed_replay_buffer import FixedReplayBuffer


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3])
    p = prepare_logs(env_name, algo_name, argvs)

    env = CarOnHill()
    rb = FixedReplayBuffer(
        data_dir=f"{p['data_dir']}/{p['seed']}",
        n_buffers_to_load=p["n_buffers_to_load"],
        replay_buffer_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: np.clip(x, -1, 1),
        stack_size=4,
        compress=True,
        sampler_seed=p["seed"],
    )
    agent = CQL(
        jax.random.PRNGKey(p["seed"]),
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        target_update_frequency=p["target_update_frequency"],
        adam_eps=0.0003125,
        alpha_cql=p["alpha_cql"],
    )
    train(p, agent, rb)


if __name__ == "__main__":
    run()
