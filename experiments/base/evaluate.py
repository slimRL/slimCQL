import os
import json
import pickle
from functools import partial

import jax
import jax.numpy as jnp

from slimdqn.networks.dqn import DQN
from slimdqn.networks.cql import CQL


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_eval"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_eval):
    uniform_key, action_key = jax.random.split(key, 2)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_eval,  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


def evaluate_one_epoch(key, agent_best_action, agent_params, env, p):
    eval_episode_returns_epoch = []
    eval_episode_lengths_epoch = []
    eval_episode_returns_epoch.append(0)
    eval_episode_lengths_epoch.append(0)

    n_evaluation_steps = 0
    env.reset()
    has_reset = False

    while n_evaluation_steps < p["n_evaluation_steps_per_epoch"] or not has_reset:
        key, action_key = jax.random.split(key)
        action = select_action(
            agent_best_action,
            agent_params,
            env.state,
            action_key,
            env.n_actions,
            p["epsilon_eval"],
        ).item()

        reward, absorbing = env.step(action)

        n_evaluation_steps += 1
        eval_episode_returns_epoch[-1] += reward
        eval_episode_lengths_epoch[-1] += 1

        has_reset = absorbing or env.n_steps >= p["horizon"]
        if has_reset:
            env.reset()

        if has_reset and n_evaluation_steps < p["n_evaluation_steps_per_epoch"]:
            eval_episode_returns_epoch.append(0)
            eval_episode_lengths_epoch.append(0)

    return eval_episode_returns_epoch, eval_episode_lengths_epoch


def evaluate(p, env):
    if p["algo_name"] == "dqn":
        agent = DQN(
            key=jax.random.PRNGKey(0),
            observation_dim=(env.state_height, env.state_width, env.n_stacked_frames),
            n_actions=env.n_actions,
            features=p["features"],
            architecture_type=p["architecture_type"],
            learning_rate=-1,
            gamma=-1,
            update_horizon=-1,
            target_update_frequency=-1,
            adam_eps=-1,
        )
    elif p["algo_name"] == "cql":
        agent = CQL(
            key=jax.random.PRNGKey(0),
            observation_dim=(env.state_height, env.state_width, env.n_stacked_frames),
            n_actions=env.n_actions,
            features=p["features"],
            architecture_type=p["architecture_type"],
            learning_rate=-1,
            gamma=-1,
            update_horizon=-1,
            target_update_frequency=-1,
            adam_eps=-1,
            alpha_cql=-1,
        )
    else:
        assert False, f"Invalid algorithm {p['algo_name']}"

    eval_episode_returns, eval_episode_lengths = evaluate_one_epoch(
        jax.random.PRNGKey(seed=p["seed"]),
        agent.best_action,
        pickle.load(
            open(
                f"{p['save_path']}/models/{p['seed']}/{p['epoch']}",
                "rb",
            )
        )["params"],
        env,
        p,
    )

    os.makedirs(
        f"{p['save_path']}/episode_returns_and_lengths/{p['seed']}",
        exist_ok=True,
    )
    json.dump(
        {"episode_returns": list(eval_episode_returns), "episode_lengths": list(eval_episode_lengths)},
        open(
            f"{p['save_path']}/episode_returns_and_lengths/{p['seed']}/{p['epoch']}_results.json",
            "w",
        ),
    )
