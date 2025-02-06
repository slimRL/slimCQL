import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial

from experiments.base.utils import save_data
from slimfqi.networks.dqn import DQN
from slimfqi.sample_collection.fixed_replay_buffer import FixedReplayBuffer


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_eval"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_eval):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_eval,  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


def evaluation_per_iteration(
    key,
    agent,
    env,
    p,
    idx_iteration,
    eval_episode_returns_per_iteration,
    eval_episode_lengths_per_iteration,
):
    n_evaluation_steps_iteration = 0
    env.reset()
    has_reset = False

    while n_evaluation_steps_iteration < p["n_evaluation_steps_per_iteration"] or not has_reset:
        key, action_key = jax.random.split(key)
        action = select_action(
            agent.best_action,
            agent.params,
            env.state,
            action_key,
            env.n_actions,
            p["epsilon_eval"],
        ).item()

        reward, absorbing = env.step(action)

        n_evaluation_steps_iteration += 1

        has_reset = absorbing or env.n_steps >= p["max_steps_per_episode"]
        if has_reset:
            env.reset()

        eval_episode_returns_per_iteration[idx_iteration][-1] += reward
        eval_episode_lengths_per_iteration[idx_iteration][-1] += 1
        if has_reset and n_evaluation_steps_iteration < p["n_evaluation_steps_per_iteration"]:
            eval_episode_returns_per_iteration[idx_iteration].append(0)
            eval_episode_lengths_per_iteration[idx_iteration].append(0)


def train_and_eval(
    key: jax.random.PRNGKey,
    p: dict,
    agent: DQN,
    env,
    fixed_rb: FixedReplayBuffer,
):
    n_training_steps = 0
    eval_episode_returns_per_iteration = [[0]]
    eval_episode_lengths_per_iteration = [[0]]

    evaluation_per_iteration(
        key,
        agent,
        env,
        p,
        0,
        eval_episode_returns_per_iteration,
        eval_episode_lengths_per_iteration,
    )

    avg_return = np.mean(eval_episode_returns_per_iteration[0])
    avg_length_episode = np.mean(eval_episode_lengths_per_iteration[0])
    n_episodes = len(eval_episode_lengths_per_iteration[0])
    print(
        f"\nIteration 0: Return {avg_return} averaged on {n_episodes} episodes.\n",
        flush=True,
    )

    p["wandb"].log(
        {
            "iteration": 0,
            "n_training_steps": n_training_steps,
            "avg_return": avg_return,
            "avg_length_episode": avg_length_episode,
        }
    )

    eval_episode_returns_per_iteration.append([0])
    eval_episode_lengths_per_iteration.append([0])

    for idx_iteration in tqdm(range(p["n_iterations"])):
        fixed_rb.reload_data()

        for _ in range(p["n_fitting_steps"]):
            agent.update_online_params(n_training_steps, fixed_rb)
            target_updated, logs = agent.update_target_params(n_training_steps)

            if target_updated:
                p["wandb"].log({"n_training_steps": n_training_steps, **logs})

            n_training_steps += 1

        evaluation_per_iteration(
            key,
            agent,
            env,
            p,
            idx_iteration + 1,
            eval_episode_returns_per_iteration,
            eval_episode_lengths_per_iteration,
        )

        avg_return = np.mean(eval_episode_returns_per_iteration[idx_iteration])
        avg_length_episode = np.mean(eval_episode_lengths_per_iteration[idx_iteration])
        n_episodes = len(eval_episode_lengths_per_iteration[idx_iteration])
        print(
            f"\nIteration {idx_iteration}: Return {avg_return} averaged on {n_episodes} episodes.\n",
            flush=True,
        )

        p["wandb"].log(
            {
                "iteration": idx_iteration + 1,
                "n_training_steps": n_training_steps,
                "avg_return": avg_return,
                "avg_length_episode": avg_length_episode,
            }
        )

        if idx_iteration < p["n_iterations"] - 1:
            eval_episode_returns_per_iteration.append([0])
            eval_episode_lengths_per_iteration.append([0])

        save_data(
            p,
            eval_episode_returns_per_iteration,
            eval_episode_lengths_per_iteration,
            agent.get_model(),
        )
