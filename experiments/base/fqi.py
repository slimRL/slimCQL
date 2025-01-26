import jax
from tqdm import tqdm

from experiments.base.utils import save_data
from slimfqi.networks.dqn import DQN
from slimfqi.sample_collection.replay_buffer import ReplayBuffer


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: DQN,
    rb: ReplayBuffer,
):
    n_grad_steps = int((p["n_fitting_steps"] * p["replay_buffer_capacity"]) / p["batch_size"])

    save_data(p, agent.get_model())

    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        for _ in range(n_grad_steps):
            key, grad_key = jax.random.split(key)
            agent.update_online_params(0, rb)
        _, logs = agent.update_target_params(0)
        p["wandb"].log({"idx_bellman_iteration": idx_bellman_iteration, **logs})

        save_data(p, agent.get_model())
