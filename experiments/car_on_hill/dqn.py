import os
import sys

import jax

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from experiments.car_on_hill.prepare_data import generate_replay_buffer
from slimdqn.environments.car_on_hill import CarOnHill
from slimdqn.networks.dqn import DQN
from slimdqn.sample_collection.fixed_replay_buffer import FixedReplayBuffer


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (
        os.path.abspath(__file__).split("/")[-2],
        os.path.abspath(__file__).split("/")[-1][:-3],
    )
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = CarOnHill()

    if p["data_dir"] is None:
        generate_replay_buffer(p, env)
        p["data_dir"] = f"{p['save_path']}/../../../replay_buffer/uniform_{p['replay_buffer_capacity']}"

    rb = FixedReplayBuffer(
        data_dir=p["data_dir"],
        n_buffers_to_load=1,
        replay_buffer_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=None,
        stack_size=1,
        compress=False,
        sampler_seed=p["seed"],
        replay_checkpoint=0,
    )
    agent = DQN(
        q_key,
        env.observation_shape,
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        target_update_frequency=p["target_update_frequency"],
        adam_eps=1.5e-4,
    )
    train(p, agent, rb)


if __name__ == "__main__":
    run()
