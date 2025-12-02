# slimCQL - simple, minimal and flexible offline Deep RL

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![jax_badge][jax_badge_link]
![Static Badge](https://img.shields.io/badge/lines%20of%20code-1318-green)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`slimCQL`** provides a concise and customizable implementation of Deep Q-Network (DQN) and Conservative Q-Learning (CQL) algorithms in Reinforcement Learning⛳ for Atari environments. 
It enables to quickly code and run proof-of-concept type of experiments in off-policy Deep RL settings.

## User installation
GPU installation for Atari:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```
To verify the installation, run the tests as: ```pytest```

## Running experiments

### Dataset preparation

To train an offline RL agent on an Atari game for a given seed, we first download the dataset published in this work 👉 [RL Unplugged](https://arxiv.org/abs/2006.13888) (RLU). It contains the trajectories seen by a DQN agent trained for 50 million transitions.

To prepare the dataset for VideoPinball, seed 1:

1. Download the dataset from [RLU GCP bucket](https://console.cloud.google.com/storage/browser/rl_unplugged) (`run_1*` fetches all files corresponding to seed 1):
    ```bash
    mkdir -p experiments/atari/datasets/rlu_dataset/VideoPinball
    gsutil -m cp -R gs://rl_unplugged/atari_episodes_ordered/VideoPinball/run_1* experiments/atari/datasets/rlu_dataset/VideoPinball
    ```
    This stores the raw trajectories in `experiments/atari/datasets/rlu_dataset/VideoPinball`.


2. Convert the raw dataset into condensed numpy arrays (requires much less space, as it removes the redundant information), by setting the `GAME` and `RUN` variables in `experiments/atari/rlu_to_numpy.py` and running: `python3 experiments/atari/rlu_to_numpy.py`. Once complete, the arrays for the given game and run are stored in `experiments/atari/datasets/numpy_dataset/VideoPinball/1`.

3. Now you can prepare the replay buffers (used by the offline RL agent) for given values of $n$ and $\gamma$, by setting the `update_horizon` and `gamma` variables respectively (along with `GAME` and `RUN` variables) in `experiments/atari/prepare_replay_buffers.py`, and running: `python3 experiments/atari/prepare_replay_buffers.py`. Upon completion, the replay buffers are stored in `experiments/atari/datasets/slim_dataset/VideoPinball/1`.

The dataset for is now ready for learning! At this point, you can delete the downloaded dataset in `experiments/atari/datasets/rlu_dataset/VideoPinball` if you don't need them anymore.

### Training

To train a CQL agent on VideoPinball on your local system, run the launch file:\
`
launch_job/atari/launch.sh
`

It trains and evaluates a CQL agent on VideoPinball (seed 1) with CNN architecture, for 3.125 million gradient steps, using 10% of the dataset collected by the DQN agent.

- To see the stage of training, you can check the logs in `experiments/atari/logs/test_VideoPinball/cql` folder
- The models at the end of each epoch are stored in `experiments/atari/exp_output/test_VideoPinball/cql/models` folder
- To modify the percentage of DQN dataset to be used in training to $p$%, set the `replay_buffer_capacity` as $\lfloor p$\% $\times 1,000,000\rfloor$ in the launch file.

To train on cluster, change `launch_job/atari/local_cql.sh` in the launch file, to `launch_job/atari/cluster_cql.sh`, and run the launch file.


### Evaluation

The launch file organizes the evaluation upon completion of training as follows:

- `experiments/atari/evaluate.py` is used to evaluate the model at the end of every epoch.
- Once complete, `experiments/synchronize_evaluation_wandb.py` is used to:
    - Combine the evaluation returns into `experiments/atari/exp_output/test_VideoPinball/cql/episode_returns_and_lengths` folder,
    - Delete models corresponding to all epochs but the last (one at the end of the training),
    - Update wandb with evaluation results (if the flag --disable_wandb is not turned on).

[jax_badge_link]: https://tinyurl.com/5n8m53cy