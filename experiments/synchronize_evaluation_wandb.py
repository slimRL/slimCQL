import os
import shutil
import sys
import json
import argparse
import wandb
import numpy as np

from experiments.base.eval_parser_argument import add_eval_arguments


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Evaluate offline agent.")
    add_eval_arguments(parser)
    args = parser.parse_args(argvs)

    env_name = os.path.abspath(__file__).split("/")[-2]
    p = json.load(open(f"experiments/{env_name}/exp_output/{args.experiment_name}/parameters.json", "rb"))
    p = dict(list(p["shared_parameters"].items()) + list(p[args.algo_name].items()) + list(vars(args).items()))
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo_name']}",
    )

    api = wandb.Api()
    runs = api.runs(
        "slimCQL",
        filters={
            "config.experiment_name": p["experiment_name"],
            "config.algo_name": p["algo_name"],
            "config.seed": p["seed"],
        },
    )
    assert len(runs) == 1, f"There are multiple {p['experiment_name']} runs for {p['algo_name']} with seed {p['seed']}."

    run = wandb.init(project="slimCQL", id=runs[0].id, resume="must", settings=wandb.Settings(_disable_stats=True))
    name_iterations = "n_iterations" if "n_iterations" in run.config else "n_epochs"
    last_step = min(
        run.summary.get("_step"),
        (run.config["n_fitting_steps"] * run.config[name_iterations]) // run.config["target_update_frequency"],
    )
    all_results = {"episode_returns": [], "episode_lengths": []}
    for idx_epoch in range(p[name_iterations] + 1):
        epoch_results = json.load(
            open(
                f"{p['save_path']}/episode_returns_and_lengths/{p['seed']}/{idx_epoch}_results.json",
                "r",
            ),
        )
        all_results["episode_returns"].append(epoch_results["episode_returns"])
        all_results["episode_lengths"].append(epoch_results["episode_lengths"])
        run.log(
            {
                "epoch": idx_epoch,
                "n_training_steps": idx_epoch * p["n_fitting_steps"],
                "avg_return": np.mean(epoch_results["episode_returns"]),
                "avg_length_episode": np.mean(epoch_results["episode_lengths"]),
            },
            step=last_step + idx_epoch + 1,
        )
    wandb.finish()

    json.dump(
        all_results,
        open(
            f"{p['save_path']}/episode_returns_and_lengths/{p['seed']}_results.json",
            "w",
        ),
    )
    shutil.rmtree(f"{p['save_path']}/episode_returns_and_lengths/{p['seed']}")

    if p["delete_models"]:
        for idx_epoch in range(0, p[name_iterations]):
            os.remove(os.path.join(p["save_path"], f"models/{p['seed']}/{idx_epoch}"))


if __name__ == "__main__":
    run()
