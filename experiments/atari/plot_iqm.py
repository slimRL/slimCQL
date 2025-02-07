import os
import json
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt
from experiments.base.compute_iqm import get_iqm_and_conf

env_name = "atari"
experiment_folders = ["slimfqi_Pong/fqi"]

base_path = os.path.join(os.path.abspath(""), "experiments", env_name, "exp_output")

experiment_data = {experiment: {} for experiment in experiment_folders}

for experiment in experiment_folders:
    experiment_path = os.path.join(base_path, experiment, "episode_returns_and_lengths")

    returns_experiment_ = []

    for experiment_file in os.listdir(experiment_path):
        list_episode_returns = json.load(open(os.path.join(experiment_path, experiment_file), "rb"))["returns"]

        returns_experiment_.append([np.mean(episode_returns) for episode_returns in list_episode_returns])

    returns_experiment = np.array(list(zip_longest(*returns_experiment_, fillvalue=np.nan))).T

    p = json.load(open(os.path.join(experiment_path, "../../parameters.json"), "rb"))

    print(f"Plot {experiment} with {returns_experiment.shape[0]} seeds.")
    if returns_experiment.shape[1] < p["fqi"]["n_iterations"]:
        print(f"!!! All the {returns_experiment.shape[0]} seeds are not complete !!!")
    elif np.isnan(returns_experiment).any():
        seeds = np.array(list(map(lambda path: int(path.strip(".json")), os.listdir(experiment_path))))
        print(f"!!! Seeds {seeds[np.isnan(returns_experiment).any(axis=1)]} are not complete !!!")

    experiment_data[experiment]["iqm"], experiment_data[experiment]["confidence"] = get_iqm_and_conf(
        returns_experiment
    )
    experiment_data[experiment]["x_values"] = (
        np.arange(1, returns_experiment.shape[1] + 1) * p["fqi"]["n_fitting_steps"]
    )
    
from experiments.atari import COLORS, ORDERS


plt.rc("font", family="serif", serif="Times New Roman", size=18)
plt.rc("lines", linewidth=3)

fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot(111)

for experiment in experiment_folders:
    ax.plot(
        experiment_data[experiment]["x_values"],
        experiment_data[experiment]["iqm"],
        label=experiment.split("/")[1].upper(),
        color=COLORS[experiment.split("/")[1]],
        zorder=ORDERS[experiment.split("/")[1]],
    )
    ax.fill_between(
        experiment_data[experiment]["x_values"],
        experiment_data[experiment]["confidence"][0],
        experiment_data[experiment]["confidence"][1],
        color=COLORS[experiment.split("/")[1]],
        alpha=0.3,
        zorder=ORDERS[experiment.split("/")[1]],
    )
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

ax.set_xlabel("Grad Steps")
ax.set_ylabel("IQM Return")

ax.grid()
ax.legend(ncols=1, frameon=False, loc="center", bbox_to_anchor=(1.25, 0.5))
ax.set_title("Pong - FQI")
fig.savefig(f"experiments/{env_name}/exp_output/performances.pdf", bbox_inches="tight")