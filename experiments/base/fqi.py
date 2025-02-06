import jax
from tqdm import tqdm

from experiments.base.utils import save_model
from slimfqi.networks.dqn import DQN
from slimfqi.sample_collection.fixed_replay_buffer import FixedReplayBuffer


def train(
    p: dict,
    agent: DQN,
    fixed_rb: FixedReplayBuffer,
):
    n_training_steps = 0
    save_model(p, agent.get_model(), 0)

    for idx_iteration in tqdm(range(1, p["n_iterations"] + 1)):
        fixed_rb.reload_data()

        for _ in range(p["n_fitting_steps"]):
            agent.update_online_params(n_training_steps, fixed_rb)
            target_updated, logs = agent.update_target_params(n_training_steps)

            if target_updated:
                p["wandb"].log({"n_training_steps": n_training_steps, **logs})

            n_training_steps += 1

        save_model(p, agent.get_model(), idx_iteration)
