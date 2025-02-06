import os
import sys

import jax
import numpy as np

from experiments.base.fqi import train_and_eval
from experiments.base.utils import prepare_logs
from slimfqi.environments.atari import AtariEnv
from slimfqi.networks.cql import CQL
from slimfqi.sample_collection.fixed_replay_buffer import FixedReplayBuffer


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (
        os.path.abspath(__file__).split("/")[-2],
        os.path.abspath(__file__).split("/")[-1][:-3],
    )
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    rb = FixedReplayBuffer(
        data_dir=p["data_dir"],
        n_buffers_to_load=p["n_buffers_to_load"],
        replay_checkpoint=p["replay_checkpoint"],
        replay_file_start_index=p["replay_file_start_index"],
        replay_file_end_index=p["replay_file_end_index"],
        replay_transitions_start_index=p["replay_transitions_start_index"],
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
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=1,
        target_update_frequency=p["target_update_frequency"],
        adam_eps=1.5e-4,
        alpha_cql=p["alpha_cql"],
    )
    train_and_eval(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
