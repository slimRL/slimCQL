import time
from experiments.base.utils import save_model
from slimdqn.sample_collection.fixed_replay_buffer import FixedReplayBuffer


def train(
    p: dict,
    agent,
    fixed_rb: FixedReplayBuffer,
):
    save_model(p, agent.get_model(), 0)

    n_steps_remaining_to_target_update = p["target_update_frequency"]

    for idx_epoch in range(p["n_epochs"]):
        time_epoch = time.time()
        fixed_rb.reload_data()
        n_steps_remaining_to_epoch = p["n_fitting_steps"]
        while n_steps_remaining_to_epoch > 0:
            # 500 has to be set to maximize vRAM utilization
            n_steps_todo = min(n_steps_remaining_to_epoch, n_steps_remaining_to_target_update, 500)
            agent.n_updates_online_params(n_steps_todo, fixed_rb)

            n_steps_remaining_to_epoch -= n_steps_todo
            n_steps_remaining_to_target_update -= n_steps_todo

            if n_steps_remaining_to_target_update == 0:
                n_training_steps = idx_epoch * p["n_fitting_steps"] + p["n_fitting_steps"] - n_steps_remaining_to_epoch
                logs = agent.update_target_params(step=n_training_steps)
                p["wandb"].log({"n_training_steps": n_training_steps, **logs})

                n_steps_remaining_to_target_update = p["target_update_frequency"]

        fixed_rb.clear()
        save_model(p, agent.get_model(), idx_epoch + 1)
        print(f"--- Epoch {idx_epoch + 1} completed in {round((time.time() - time_epoch)/60, 4)} mins ---")
