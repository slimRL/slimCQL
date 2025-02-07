import os
import sys
import argparse
import json
import pickle
import multiprocess
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import jax.numpy as jnp

from slimfqi.networks.dqn import DQN
from slimfqi.environments.atari import AtariEnv


@partial(jax.jit, static_argnames=("best_action_fn", "n_actions", "epsilon_eval"))
def select_action(best_action_fn, params, state, key, n_actions, epsilon_eval):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_eval,  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        best_action_fn(params, state),  # otherwise, take a greedy action
    )


def evaluate_one_iteration(
    key,
    agent_best_action,
    agent_params,
    env,
    p,
    args,
    idx_iteration,
    eval_episode_returns,
    eval_episode_lengths,
):
    
    eval_episode_returns_per_iteration = []
    eval_episode_lengths_per_iteration = []
    eval_episode_returns_per_iteration[idx_iteration].append(0)
    eval_episode_lengths_per_iteration[idx_iteration].append(0)
    
    n_evaluation_steps_iteration = 0
    env.reset()
    has_reset = False

    while n_evaluation_steps_iteration < args.n_evaluation_steps_per_iteration or not has_reset:
        key, action_key = jax.random.split(key)
        action = select_action(
            agent_best_action,
            agent_params,
            env.state,
            action_key,
            env.n_actions,
            args.epsilon_eval,
        ).item()
        # print(f"Eval action = {action}")

        reward, absorbing = env.step(action)

        n_evaluation_steps_iteration += 1
        eval_episode_returns_per_iteration[idx_iteration][-1] += reward
        eval_episode_lengths_per_iteration[idx_iteration][-1] += 1

        has_reset = absorbing or env.n_steps >= args.max_steps_per_episode
        if has_reset:
            env.reset()

        if has_reset and n_evaluation_steps_iteration < args.n_evaluation_steps_per_iteration:
            eval_episode_returns_per_iteration[idx_iteration].append(0)
            eval_episode_lengths_per_iteration[idx_iteration].append(0)
            
    eval_episode_returns[idx_iteration] = eval_episode_returns_per_iteration
    eval_episode_lengths[idx_iteration] = eval_episode_lengths_per_iteration


def evaluate(key, q, p, args):
    all_keys = jax.random.split(key, p[args.algo_name]["n_iterations"])

    processes = []
    manager = multiprocess.Manager()
    eval_episode_returns = manager.list([np.nan for _ in range(p[args.algo_name]["n_iterations"])])
    eval_episode_lengths = manager.list([np.nan for _ in range(p[args.algo_name]["n_iterations"])])

    for idx_iteration in range(p[args.algo_name]["n_iterations"]):
        processes.append(
            multiprocess.Process(
                target=evaluate_one_iteration,
                args=(
                    all_keys[idx_iteration],
                    q.best_action,
                    pickle.load(
                        open(
                            f"experiments/atari/exp_output/{args.experiment_name}/{args.algo_name}/models/{args.seed}/{idx_iteration}",
                            "rb",
                        )
                    )["params"],
                    AtariEnv(p["shared_parameters"]["experiment_name"].split("_")[-1]),
                    p,
                    args,
                    idx_iteration,
                    eval_episode_returns,
                    eval_episode_lengths,
                ),
            )
        )

    for process in processes:
        process.start()
    for process in processes:
        process.join()
    results = {"returns": eval_episode_returns, "episode_lengths": eval_episode_lengths}
    os.makedirs(f"experiments/{p['shared_parameters']['experiment_name'].split('_')[-1]}/exp_output/{args.experiment_name}/{args.algo_name}/results", exist_ok=True)
    pickle.dump(results, open(f"experiments/{p['shared_parameters']['experiment_name'].split('_')[-1]}/exp_output/{args.experiment_name}/{args.algo_name}/results/{args.seed}_results.pkl", "wb"))


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Evaluate offline agent.")
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-an",
        "--algo_name",
        help="Algorithm name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-mspe",
        "--max_steps_per_episode",
        help="Max. steps per episode in evaluation.",
        type=int,
        default=27_000,
    )
    parser.add_argument(
        "-nespi",
        "--n_evaluation_steps_per_iteration",
        help="Evaluation steps per iteration.",
        type=int,
        default=125_000,
    )
    parser.add_argument(
        "-ee",
        "--epsilon_eval",
        help="Epsilon to use for evaluation.",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "-wb",
        "--wandb_link",
        help="Wandb run link for the experiment for logging into same run.",
        type=str,
        default=None,
    )
    args = parser.parse_args(argvs)

    env_name = os.path.abspath(__file__).split("/")[-2]
    p_path = f"experiments/{env_name}/exp_output/{args.experiment_name}/parameters.json"
    p = json.load(open(p_path, "rb"))

    env = AtariEnv(p["shared_parameters"]["experiment_name"].split("_")[-1])

    key = jax.random.PRNGKey(args.seed)
    q_key, eval_key = jax.random.split(key)

    agent = DQN(
        key=jax.random.PRNGKey(0),
        observation_dim=(env.state_height, env.state_width, env.n_stacked_frames),
        n_actions=env.n_actions,
        features=p["shared_parameters"]["features"],
        architecture_type=p["shared_parameters"]["architecture_type"],
        learning_rate=p["shared_parameters"]["learning_rate"],
        gamma=p["shared_parameters"]["gamma"],
        update_horizon=p["shared_parameters"]["update_horizon"],
        update_to_data=-1,
        target_update_frequency=-1,
        adam_eps=-1,
    )

    evaluate(eval_key, agent, p, args)

if __name__ == "__main__":
    run()