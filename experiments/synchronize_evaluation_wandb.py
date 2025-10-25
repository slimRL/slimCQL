import os
import shutil
import sys
import json
import argparse
import wandb
import numpy as np

from experiments.base.eval_parser_argument import add_synchronization_arguments


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Evaluate offline agent.")
    add_synchronization_arguments(parser)
    args = parser.parse_args(argvs)

    p = json.load(open(f"experiments/{args.env_name}/exp_output/{args.experiment_name}/parameters.json", "rb"))

    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{args.env_name}/exp_output/{args.experiment_name}/{args.algo_name}",
    )

    api = wandb.Api()
    runs = api.runs(
        "slimCQL",
        filters={
            "config.experiment_name": args.experiment_name,
            "config.algo_name": args.algo_name,
            "config.seed": args.seed,
        },
    )
    assert len(runs) == 1, f"There are multiple {args.experiment_name} runs for {args.algo_name} with seed {args.seed}."

    run = wandb.init(project="slimCQL", id=runs[0].id, resume="must", settings=wandb.Settings(_disable_stats=True))
    last_step = min(
        run.summary.get("_step"),
        (run.config["n_fitting_steps"] * run.config["n_epochs"]) // run.config["target_update_period"],
    )
    all_results = {"episode_returns": [], "episode_lengths": []}
    for idx_epoch in range(p["shared_parameters"]["n_epochs"] + 1):
        epoch_results = json.load(
            open(
                f"{p['save_path']}/episode_returns_and_lengths/{args.seed}/{idx_epoch}_results.json",
                "r",
            ),
        )
        all_results["episode_returns"].append(epoch_results["episode_returns"])
        all_results["episode_lengths"].append(epoch_results["episode_lengths"])
        run.log(
            {
                "epoch": idx_epoch,
                "n_training_steps": idx_epoch * p["shared_parameters"]["n_fitting_steps"],
                "avg_return": np.mean(epoch_results["episode_returns"]),
                "avg_length_episode": np.mean(epoch_results["episode_lengths"]),
            },
            step=last_step + idx_epoch + 1,
        )
    wandb.finish()

    json.dump(
        all_results,
        open(
            f"{p['save_path']}/episode_returns_and_lengths/{args.seed}_results.json",
            "w",
        ),
    )
    shutil.rmtree(f"{p['save_path']}/episode_returns_and_lengths/{args.seed}")

    if args.delete_models:
        for idx_epoch in range(0, p["shared_parameters"]["n_epochs"]):
            os.remove(os.path.join(p["save_path"], f"models/{args.seed}/{idx_epoch}"))


if __name__ == "__main__":
    run()
